import inspect
import re
import pyopencl as cl
import numpy as np
basecl_path = os.path.join(os.path.dirname(__file__), '..', 'cl_kernels', 'base_cl_functions.cl')
# Загружаем базовую математическую библиотеку OpenCL C, которую мы написали ранее
BASE_CL_FUNCTIONS = """
inline float nn_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float nn_tanh(float x) { return tanh(x); }
inline float nn_sigmoid_backward(float x, float go) { float s = nn_sigmoid(x); return go * s * (1.0f - s); }
inline float nn_tanh_backward(float x, float go) { float t = nn_tanh(x); return go * (1.0f - t * t); }
"""
with open(matmul_path, 'r', encoding='utf-8') as f:
    BASE_CL_FUNCTIONS=f.read()
class JITProgramCache:
    """Глобальный кэш скомпилированных OpenCL программ, чтобы избежать overhead драйвера"""
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
        
        # Списки для автоматического построения цепочки обратного прохода (Backward)
        self.backward_tasks = []

    def __call__(self, lambda_func):
        """
        Принимает lambda-выражение вида: lambda: get_A.assign(nnclloc.sigmoid(get_A))
        Парсит его тело, строит монолитный OpenCL Kernel и выполняет его за один проход VRAM.
        """
        # 1. Извлекаем исходный код лямбды через рефлексию Python
        raw_code = inspect.getsource(lambda_func)
        
        # Находим саму структуру выражения внутри скобок LocalRunner(...)
        match = re.search(r'LocalRunner\(\s*lambda\s*:\s*(.*?)\s*\)', raw_code, re.DOTALL)
        if not match:
            # Попробуем альтернативный синтаксис (мультистрочный или обычный)
            match = re.search(r'lambda\s*:\s*(.*)', raw_code)
            
        expr_str = match.group(1).strip()
        
        # Разделяем выражения, если они написаны через запятую или точку с запятой
        expressions = [e.strip() for e in re.split(r'[;\n]', expr_str) if e.strip()]
        
        # Словари для отслеживания задействованных в ядре буферов
        detected_tensors = {}
        cl_body_forward = []
        cl_body_backward = []

        # 2. Транслируем синтаксис псевдокода Python в нативный OpenCL C код
        for expr in expressions:
            # Парсим выражения вида: target = nnclloc.name(source) или target = source1 * source2
            # Шаблон парсинга присвоения
            if '=' in expr:
                target, operation = expr.split('=', 1)
                target = target.strip()
                operation = operation.strip()
                
                # Подставляем вызовы нативных инлайн-функций C++ OpenCL
                cl_op = operation.replace('nnclloc.', 'nn_')
                
                # Заменяем обращения к переменным на индексированные массивы в OpenCL
                # Пример: memory_state превратится в memory_state[tid]
                tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', cl_op)
                for t in tokens:
                    if not t.startswith('nn_') and t not in ['float', 'int']:
                        detected_tensors[t] = True
                        cl_op = re.sub(r'\b' + t + r'\b', f"{t}[tid]", cl_op)
                
                detected_tensors[target] = True
                cl_body_forward.append(f"{target}[tid] = {cl_op};")
                
                # 3. ЕСЛИ ВКЛЮЧЕН РЕЖИМ ОБУЧЕНИЯ: Строим параллельную цепочку Backward
                if self.is_training:
                    # Символьное автоматическое дифференцирование легких выражений
                    if "nn_sigmoid" in cl_op:
                        src_var = re.search(r'nn_sigmoid\((.*?)\)', cl_op).group(1)
                        cl_body_backward.append(f"grad_{src_var.replace('[tid]', '')}[tid] = nn_sigmoid_backward({src_var}, grad_{target}[tid]);")
                    elif "nn_tanh" in cl_op:
                        src_var = re.search(r'nn_tanh\((.*?)\)', cl_op).group(1)
                        cl_body_backward.append(f"grad_{src_var.replace('[tid]', '')}[tid] = nn_tanh_backward({src_var}, grad_{target}[tid]);")
                    elif "+" in cl_op:
                        # Производная суммы: градиент просто течет в оба слагаемых
                        vars_add = [v.strip() for v in operation.split('+')]
                        for v in vars_add:
                            cl_body_backward.append(f"grad_{v}[tid] += grad_{target}[tid];")

        # 4. Собираем финальный текст глобального OpenCL ядра
        kernel_name = f"fused_kernel_{abs(hash(expr_str))}"
        
        # Формируем аргументы функции ядра OpenCL
        kernel_args = []
        for tensor in detected_tensors.keys():
            kernel_args.append(f"__global float* {tensor}")
            if self.is_training:
                kernel_args.append(f"__global float* grad_{tensor}")
                
        args_str = ", ".join(kernel_args)
        forward_statements = "\n    ".join(cl_body_forward)

        full_kernel_code = f"""
        {BASE_CL_FUNCTIONS}
        
        __kernel void {kernel_name}({args_str}, const int total_elements) {{
            int tid = get_global_id(0);
            if (tid >= total_elements) return;
            
            // Прямой проход (Forward)
            {forward_statements}
        }}
        """
        
        # 5. Компиляция или извлечение из кэша
        program = JITProgramCache.get_or_compile(self.ctx, full_kernel_code)
        kernel = getattr(program, kernel_name)
        
        # 6. Подготовка аргументов для вызова (Буферы PyOpenCL)
        # Для демонстрации предполагаем, что объекты тензоров лежат в локальной области видимости лямбды
        closure_vars = inspect.getclosurevars(lambda_func).nonlocals
        
        # Определяем размер по первому попавшемуся буферу
        first_tensor_name = list(detected_tensors.keys())[0]
        total_elements = closure_vars[first_tensor_name].size 
        
        cl_buffers_call = []
        for tensor in detected_tensors.keys():
            tensor_obj = closure_vars[tensor]
            cl_buffers_call.append(tensor_obj.data_buffer) # Нативный cl.Buffer
            if self.is_training:
                cl_buffers_call.append(tensor_obj.grad_buffer)

        # 7. Запуск скомпилированного монолитного ядра на GPU
        kernel(self.queue, (total_elements,), None, *cl_buffers_call, np.int32(total_elements))
        
        # Если мы учим модель, сохраняем сгенерированный backward-текст в стек задач
        if self.is_training and cl_body_backward:
            backward_statements = "\n    ".join(cl_body_backward)
            backward_kernel_code = f"""
            {BASE_CL_FUNCTIONS}
            __kernel void grad_{kernel_name}({args_str}, const int total_elements) {{
                int tid = get_global_id(0);
                if (tid >= total_elements) return;
                {backward_statements}
            }}
            """
            self.backward_tasks.append((backward_kernel_code, cl_buffers_call, total_elements))
