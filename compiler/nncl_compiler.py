import os
import inspect
import re
import pyopencl as cl
import numpy as np

# Загружаем Си-код с классическими формулами
basecl_path = os.path.join(os.path.dirname(__file__), '..', 'cl_kernels', 'base_cl_functions.cl')
with open(basecl_path, 'r', encoding='utf-8') as f:
    BASE_CL_FUNCTIONS = f.read()

class JITProgramCache:
    _cache = {}
    @classmethod
    def get_or_compile(cls, ctx, kernel_code):
        if kernel_code not in cls._cache:
            cls._cache[kernel_code] = cl.Program(ctx, kernel_code).build()
        return cls._cache[kernel_code]

class LocalRunner:
    def __init__(self, ctx, queue, is_training=False):
        self.ctx = ctx
        self.queue = queue
        self.is_training = is_training
        self.backward_stack = []

    def __call__(self, lambda_func):
        try:
            raw_code = inspect.getsource(lambda_func).strip()
        except Exception:
            raise RuntimeError("LocalRunner не смог прочитать исходный код лямбды.")

        raw_code = " ".join(raw_code.split())
        match = re.search(r'(?:LocalRunner|runner)\(\s*lambda\s*:\s*(.*?)\s*\)\s*$', raw_code)
        if not match:
            match = re.search(r'lambda\s*:\s*([^)]*?)(?:\s*\)|$)', raw_code)
        
        expr_str = match.group(1).strip().replace('==', '=')
        expressions = [e.strip() for e in re.split(r'[;\n]', expr_str) if e.strip()]

        detected_tensors = {}
        saved_tensors_vram = {}  # Кэш клонированных буферов для текущей лямбды
        cl_body_forward = []
        cl_body_backward = []

        closure_vars = inspect.getclosurevars(lambda_func).nonlocals

        for expr in expressions:
            if '=' in expr:
                target, operation = expr.split('=', 1)
                target = target.strip()
                operation = operation.strip()

                cl_op = operation.replace('nnclloc.', 'nn_')
                while cl_op.count(')') > cl_op.count('('):
                    last_bracket = cl_op.rfind(')')
                    if last_bracket != -1:
                        cl_op = cl_op[:last_bracket] + cl_op[last_bracket+1:]

                tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', cl_op)
                
                # ИСПРАВЛЕНИЕ: Если операция IN-PLACE и включено обучение,
                # делаем изолированный физический клон буфера памяти во VRAM
                if target in tokens and self.is_training:
                    saved_name = f"saved_{target}"
                    tensor_obj = closure_vars[target]
                    
                    # Выделяем новую изолированную область VRAM
                    vram_copy = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, tensor_obj.nbytes)
                    cl.enqueue_copy(self.queue, vram_copy, tensor_obj.data_buffer)
                    saved_tensors_vram[saved_name] = vram_copy

                for t in tokens:
                    if not t.startswith('nn_') and t not in ['float', 'int']:
                        detected_tensors[t] = True
                        cl_op = re.sub(r'\b' + t + r'\b', f"{t}[tid]", cl_op)

                detected_tensors[target] = True
                cl_body_forward.append(f"{target}[tid] = {cl_op};")

                # 3. АВТО-BACKWARD (Полная изоляция через локальный upstream_grad)
                if self.is_training:
                    if "nn_sigmoid" in cl_op:
                        src_var = re.search(r'nn_sigmoid\((.*?)\)', cl_op).group(1).replace('[tid]', '')
                        use_src = f"saved_{src_var}" if f"saved_{src_var}" in saved_tensors_vram else src_var
                        # Для операции T = sigmoid(S): входящим градиентом является строго grad_T.
                        # Но чтобы не читать испорченный/накопленный grad_S, если T == S, 
                        # мы изолируем шаг вычисления.
                        cl_body_backward.append(
                            f"float upstream_{target} = grad_{target}[tid];\n"
                            f"    float back_{src_var} = nn_sigmoid_backward({use_src}[tid], upstream_{target});\n"
                            f"    grad_{src_var}[tid] = back_{src_var};" # Для листка графа заменяем накопление на чистое присвоение!
                        )
                    elif "nn_tanh" in cl_op:
                        src_var = re.search(r'nn_tanh\((.*?)\)', cl_op).group(1).replace('[tid]', '')
                        use_src = f"saved_{src_var}" if f"saved_{src_var}" in saved_tensors_vram else src_var
                        cl_body_backward.append(
                            f"float upstream_{target} = grad_{target}[tid];\n"
                            f"    float back_{src_var} = nn_tanh_backward({use_src}[tid], upstream_{target});\n"
                            f"    grad_{src_var}[tid] = back_{src_var};"
                        )
                    elif "*" in operation:
                        u, v = [item.strip() for item in operation.split('*')]
                        use_u = f"saved_{u}" if f"saved_{u}" in saved_tensors_vram else u
                        use_v = f"saved_{v}" if f"saved_{v}" in saved_tensors_vram else v
                        
                        # Для операции Y = U * V:
                        # Если это промежуточный шаг, мы должны аккумулировать через += только для входов, 
                        # которые не являются целью (target) текущего накопления.
                        cl_body_backward.append(
                            f"float upstream_{target} = grad_{target}[tid];\n"
                            f"    float back_{u} = upstream_{target} * {use_v}[tid];\n"
                            f"    float back_{v} = upstream_{target} * {use_u}[tid];\n"
                            f"    grad_{u}[tid] += back_{u};\n"
                            f"    grad_{v}[tid] += back_{v};"
                        )
                    elif "+" in operation:
                        vars_add = [v.strip() for v in operation.split('+')]
                        cl_body_backward.append(f"float upstream_{target} = grad_{target}[tid];")
                        for v in vars_add:
                            cl_body_backward.append(f"    grad_{v}[tid] += upstream_{target};")


        # Сборка аргументов ядра
        kernel_hash = abs(hash(expr_str))
        kernel_name = f"fused_forward_{kernel_hash}"

        kernel_args = []
        for tensor in detected_tensors.keys():
            kernel_args.append(f"__global float* {tensor}")
            if self.is_training:
                kernel_args.append(f"__global float* grad_{tensor}")
        
        for saved_name in saved_tensors_vram.keys():
            kernel_args.append(f"__global const float* {saved_name}")

        args_str = ", ".join(kernel_args)
        forward_statements = "\n    ".join(cl_body_forward)

        full_forward_code = f"""
        {BASE_CL_FUNCTIONS}
        __kernel void {kernel_name}({args_str}, const int total_elements) {{
            int tid = get_global_id(0);
            if (tid >= total_elements) return;
            {forward_statements}
        }}
        """

        program = JITProgramCache.get_or_compile(self.ctx, full_forward_code)
        kernel = getattr(program, kernel_name)

        first_tensor_name = list(detected_tensors.keys())[0]
        total_elements = closure_vars[first_tensor_name].size

        # --- ФИКСАЦИЯ БУФЕРОВ ДЛЯ FORWARD ВЫЗОВА ---
        cl_buffers_forward = []
        for tensor in detected_tensors.keys():
            tensor_obj = closure_vars[tensor]
            cl_buffers_forward.append(tensor_obj.data_buffer)
            if self.is_training:
                tensor_obj.require_grad()
                cl_buffers_forward.append(tensor_obj.grad_buffer)
        
        for vram_buffer in saved_tensors_vram.values():
            cl_buffers_forward.append(vram_buffer)

        # Выполняем прямой проход
        kernel(self.queue, (total_elements,), None, *cl_buffers_forward, np.int32(total_elements))
        self.queue.finish()

        # --- ИЗОЛИРОВАННАЯ ФИКСАЦИЯ БУФЕРОВ ДЛЯ BACKWARD ВЫЗОВА ---
        if self.is_training and cl_body_backward:
            backward_statements = "\n    ".join(cl_body_backward)
            backward_kernel_name = f"fused_backward_{kernel_hash}"
            
            full_backward_code = f"""
            {BASE_CL_FUNCTIONS}
            __kernel void {backward_kernel_name}({args_str}, const int total_elements) {{
                int tid = get_global_id(0);
                if (tid >= total_elements) return;
                {backward_statements}
            }}
            """
            
            # ВАЖНО: Мы сохраняем в стек КОПИЮ списка буферов (cl_buffers_forward[:]), 
            # которая намертво зафиксировала состояние указателей и кэшей VRAM именно на ТЕКУЩИЙ шаг!
            cl_buffers_backward = list(cl_buffers_forward)
            
            self.backward_stack.append((full_backward_code, backward_kernel_name, cl_buffers_backward, total_elements))

    def run_backward(self):
        if not self.backward_stack:
            print("-> Стек задач Backward пуст.")
            return

        print("\n" + "#"*70)
        print("🔍 ПОШАГОВАЯ ТРАССИРОВКА ОБРАТНОГО ПРОХОДА (DEEP DETAILED LOG)")
        print("#"*70)
        
        from compiler.host_returner import HostReturner
        import re

        task_idx = 1
        while self.backward_stack:
            code, name, buffers, total_elements = self.backward_stack.pop()
            
            print(f"\n[ЗАДАЧА НАЗАД №{task_idx}]: Вызов ядра {name}")
            print("-" * 50)
            
            # Извлекаем сигнатуру аргументов из сгенерированного Си-кода ядра
            # Нам нужно понять, в каком порядке идут указатели на буферы
            args_match = re.search(r'__kernel void \w+\((.*?)\)', code, re.DOTALL)
            if args_match:
                # Получаем список имён переменных из Си-кода, например: ['gate_A', 'grad_gate_A', 'saved_gate_A']
                arg_declarations = args_match.group(1).split(',')
                arg_names = [decl.strip().split()[-1].replace('*', '') for decl in arg_declarations if decl.strip()]
                # Последний аргумент обычно const int total_elements, отсекаем его
                if 'total_elements' in arg_names[-1]:
                    arg_names = arg_names[:-1]
            else:
                arg_names = [f"buffer_{i}" for i in range(len(buffers))]

            # Делаем дамп содержимого ВСЕХ задействованных буферов ДО выполнения текущего ядра
            print("Содержимое буферов во VRAM ПЕРЕД вычислением текущего шага:")
            with np.printoptions(precision=4, suppress=True, edgeitems=8, linewidth=80):
                for name_buf, cl_buf in zip(arg_names, buffers):
                    # Создаем временный суррогатный объект, чтобы HostReturner смог прочитать сырой cl.Buffer
                    class TargetWrapper:
                        def __init__(self, buf, size, queue):
                            self.data_buffer = buf
                            self.shape = (size,)
                            self.dtype = np.dtype(np.float32)
                            self.queue = queue
                    
                    wrapper = TargetWrapper(cl_buf, total_elements, self.queue)
                    cpu_data = HostReturner.to_cpu(wrapper)
                    print(f"  • {name_buf:18} -> {cpu_data}")

            # Компилируем и запускаем ядро на GPU
            program = JITProgramCache.get_or_compile(self.ctx, code)
            kernel = getattr(program, name)
            kernel(self.queue, (total_elements,), None, *buffers, np.int32(total_elements))
            self.queue.finish() # Синхронизируем, чтобы увидеть изменения
            
            # Делаем дамп результатов ПОСЛЕ выполнения ядра
            print("\nСодержимое буферов во VRAM ПОСЛЕ вычисления текущего шага:")
            with np.printoptions(precision=4, suppress=True, edgeitems=8, linewidth=80):
                for name_buf, cl_buf in zip(arg_names, buffers):
                    wrapper = TargetWrapper(cl_buf, total_elements, self.queue)
                    cpu_data = HostReturner.to_cpu(wrapper)
                    print(f"  • {name_buf:18} -> {cpu_data}")
            
            print("-" * 50)
            task_idx += 1
            
        print("#"*70)
        print("🔍 КОНЕЦ ТРАССИРОВКИ ОБРАТНОГО ПРОХОДА")
        print("#"*70 + "\n")

