import os
import sys
import re
import pyopencl as cl
import numpy as np
from core.tensor import Tensor

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
            
            # --- АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ РАЗМЕРА ВАРПА / ВЕЙВФРОНТА ---
            # Компилируем микроскопическое пустое ядро, чтобы спросить у драйвера устройства 
            # его предпочтительную кратность потоков (preferred work group size multiple)
            dummy_code = "__kernel void dummy() {}"
            dummy_prg = cl.Program(self.ctx, dummy_code).build()
            device = self.queue.device
            
            try:
                # Извлекаем аппаратную кратность для текущего девайса (Intel/AMD/NVIDIA/Apple)
                self.warp_size = dummy_prg.dummy.get_work_group_info(
                    cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                    device
                )
                print(f"⚙️ Аппаратная оптимизация: Обнаружен размер подгруппы (Warp/Wavefront) = {self.warp_size}")
            except Exception:
                # Безопасный фоллбэк на случай непредвиденных ограничений старых драйверов
                self.warp_size = 32
                print("⚠️ Не удалось определить PREFERRED_WORK_GROUP_SIZE_MULTIPLE. Фоллбэк на 32.")

    def __call__(self, program_text: str):
        """
        Принимает многострочный блок кода в виде строки Python.
        Парсит инструкции по '\n', извлекает объекты из области видимости вызова
        и компилирует монолитное Си-ядро.
        """
        # Разбираем переданный текст на массив отдельных инструкций
        expressions = [e.strip() for e in program_text.strip().split('\n') if e.strip()]
        
        detected_tensors = {}    # Имена настоящих тензоров
        detected_scalars = {}    # Имена внешних переменных (чисел)
        saved_tensors_vram = {}  # Кэш VRAM клонов для backward
        cl_body_forward = []
        cl_body_backward = []

        # ПОЛУЧЕНИЕ ОБЛАСТИ ВИДИМОСТИ (Вместо inspect.getclosurevars)
        # sys._getframe(1).f_locals автоматически заглядывает в переменные функции, которая вызвала runner()
        caller_locals = sys._getframe(1).f_locals
        caller_globals = sys._getframe(1).f_globals

        cl_keywords = ['float', 'int', 'tid', 'unsigned', 'p', 'seed', 'p_val', 'seed_val', 'DROPOUT_P', 'DYNAMIC_SEED']

        for expr in expressions:
            # Поддерживаем и '=' и '==' для удобства синтаксиса из ТЗ
            expr_clean = expr.replace('==', '=')
            if '=' not in expr_clean:
                continue
                
            target, operation = expr_clean.split('=', 1)
            target = target.strip()
            operation = operation.strip()

            cl_op = operation.replace('nnclloc.', 'nn_')
            while cl_op.count(')') > cl_op.count('('):
                last_bracket = cl_op.rfind(')')
                if last_bracket != -1:
                    cl_op = cl_op[:last_bracket] + cl_op[last_bracket+1:]

            # Находим все переменные в текущей строке инструкции
            tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', cl_op)
            
            for t in tokens:
                if t.startswith('nn_') or t in cl_keywords or t.isdigit():
                    continue
                
                # Ищем объект во фрейме вызова (сначала локальные, потом глобальные)
                val_obj = caller_locals.get(t, caller_globals.get(t, None))
                if val_obj is not None:
                    if isinstance(val_obj, Tensor) or hasattr(val_obj, 'data_buffer'):
                        detected_tensors[t] = True
                        cl_op = re.sub(r'\b' + t + r'\b', f"{t}[tid]", cl_op)
                    else:
                        detected_scalars[t] = True
                else:
                    detected_tensors[t] = True
                    cl_op = re.sub(r'\b' + t + r'\b', f"{t}[tid]", cl_op)

            detected_tensors[target] = True
            
            # Вшиваем in-place защиту в единый Си-контур
            if target in tokens and self.is_training:
                saved_name = f"saved_{target}"
                tensor_obj = caller_locals.get(target, caller_globals.get(target))
                vram_copy = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, tensor_obj.nbytes)
                cl.enqueue_copy(self.queue, vram_copy, tensor_obj.data_buffer)
                saved_tensors_vram[saved_name] = vram_copy

            # Прямой проход с проверкой границ
            cl_body_forward.append(f"if (tid < {target}Size) {{ {target}[tid] = {cl_op}; }}")

            # Обратный проход
            if self.is_training:
                func_match = re.search(r'(nn_[a-zA-Z0-9_]+)\((.*?)\)', cl_op)
                if func_match:
                    func_name = func_match.group(1)
                    src_var = func_match.group(2).replace('[tid]', '').strip()
                    use_src = f"saved_{src_var}" if f"saved_{src_var}" in saved_tensors_vram else src_var
                    cl_body_backward.append(
                        f"if (tid < {target}Size) {{\n"
                        f"        float upstream_{target} = grad_{target}[tid];\n"
                        f"        if (tid < {src_var}Size) grad_{src_var}[tid] = {func_name}_backward({use_src}[tid], upstream_{target});\n"
                        f"    }}"
                    )
                elif "*" in operation:
                    u, v = [item.strip() for item in operation.split('*')]
                    use_u = f"saved_{u}" if f"saved_{u}" in saved_tensors_vram else u
                    use_v = f"saved_{v}" if f"saved_{v}" in saved_tensors_vram else v
                    cl_body_backward.append(
                        f"if (tid < {target}Size) {{\n"
                        f"        float upstream_{target} = grad_{target}[tid];\n"
                        f"        if (tid < {u}Size) grad_{u}[tid] += upstream_{target} * {use_v}[tid];\n"
                        f"        if (tid < {v}Size) grad_{v}[tid] += upstream_{target} * {use_u}[tid];\n"
                        f"    }}"
                    )
                elif any(op in operation for op in ["+", "-", "/"]):
                    for t in tokens:
                        if t in detected_tensors and t != target:
                            cl_body_backward.append(
                                f"if (tid < {target}Size) {{\n"
                                f"        if (tid < {t}Size) grad_{t}[tid] += grad_{target}[tid];\n"
                                f"    }}"
                            )

        # Компиляция Си-кода ядра
        kernel_hash = abs(hash(program_text))
        kernel_name = f"text_forward_{kernel_hash}"

        kernel_args = []
        for tensor in detected_tensors.keys():
            kernel_args.append(f"__global float* {tensor}")
            kernel_args.append(f"const int {tensor}Size")
            if self.is_training:
                kernel_args.append(f"__global float* grad_{tensor}")
        
        for scalar in detected_scalars.keys():
            kernel_args.append(f"const float {scalar}")
        for saved_name in saved_tensors_vram.keys():
            kernel_args.append(f"__global const float* {saved_name}")

        args_str = ", ".join(kernel_args)
        forward_statements = "\n    ".join(cl_body_forward)

        full_forward_code = f"""
        {BASE_CL_FUNCTIONS}
        __kernel void {kernel_name}({args_str}) {{
            int tid = get_global_id(0) * {self.warp_size} + get_global_id(1);
            {forward_statements}
        }}
        """

        program = JITProgramCache.get_or_compile(self.ctx, full_forward_code)
        kernel = getattr(program, kernel_name)

        # Вычисляем максимальный размер рабочей группы и собираем буферы
        max_global_size = 1
        cl_buffers_forward = []
        
        for tensor in detected_tensors.keys():
            tensor_obj = caller_locals.get(tensor, caller_globals.get(tensor))
            if tensor_obj.size > max_global_size:
                max_global_size = tensor_obj.size
                
            cl_buffers_forward.append(tensor_obj.data_buffer)
            cl_buffers_forward.append(np.int32(tensor_obj.size))
            if self.is_training:
                tensor_obj.require_grad()
                cl_buffers_forward.append(tensor_obj.grad_buffer)
        # АППАРАТНОЕ ВЫРАВНИВАНИЕ ПО WARP (32)
        # Округляем общее число потоков вверх до кратного 32
        total_count = int(np.ceil(max_global_size / self.warp_size) * self.warp_size)
        
        # Задаем размеры 2D-сетки ND-Range
        global_size = (total_count // self.warp_size, self.warp_size)
        for scalar in detected_scalars.keys():
            scalar_val = caller_locals.get(scalar, caller_globals.get(scalar))
            cl_buffers_forward.append(np.float32(scalar_val))

        cl_buffers_call_fwd = list(cl_buffers_forward)
        for vram_buffer in saved_tensors_vram.values():
            cl_buffers_call_fwd.append(vram_buffer)

        # Запускаем монолитное сплавленное ядро на GPU
        kernel(self.queue, global_size, None, *cl_buffers_call_fwd)
        self.queue.finish()

        # Стек Backward
        if self.is_training and cl_body_backward:
            backward_statements = "\n    ".join(cl_body_backward)
            backward_kernel_name = f"text_backward_{kernel_hash}"
            
            full_backward_code = f"""
            {BASE_CL_FUNCTIONS}
            __kernel void {backward_kernel_name}({args_str}) {{
                int tid = get_global_id(0) * {self.warp_size} + get_global_id(1);
                {backward_statements}
            }}
            """
            cl_buffers_backward = list(cl_buffers_forward)
            for vram_buffer in saved_tensors_vram.values():
                cl_buffers_backward.append(vram_buffer)
            self.backward_stack.append((full_backward_code, backward_kernel_name, cl_buffers_backward, global_size))

    def run_backward(self):
        if not self.backward_stack:
            return
        while self.backward_stack:
            code, name, buffers, global_size = self.backward_stack.pop() # Принимаем кортеж global_size
            program = JITProgramCache.get_or_compile(self.ctx, code)
            kernel = getattr(program, name)
            # Запуск обратного прохода в 2D топологии
            kernel(self.queue, global_size, None, *buffers)
        self.queue.finish()




