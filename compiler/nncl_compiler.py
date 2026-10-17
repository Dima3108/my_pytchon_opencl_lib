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
        # --- НОВЫЕ АТРИБУТЫ ДЛЯ СВЯЗИ СО СЛОЯМИ ВЫШЕ ---
        self._captured_inputs = set()   # Имена тензоров-аргументов (справа от =)
        self._captured_outputs = set()  # Имена изменившихся тензоров (слева от =)
        # Автодетекция размера варпа
        dummy_code = "__kernel void dummy() {}"
        dummy_prg = cl.Program(self.ctx, dummy_code).build()
        try:
            self.warp_size = dummy_prg.dummy.get_work_group_info(
                cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
                self.queue.device
            )
        except Exception:
            self.warp_size = 32

    def __call__(self, program_text: str):
        """
        ГЛАВНЫЙ МЕТОД: Принимает многострочный блок кода Python в виде обычной строки.
        Разбивает по '\n', парсит операторы '=' и строит монолитный Fused OpenCL Kernel.
        """
        print("\n" + "·"*60)
        print("🔍 ТРАССИРОВКА ТЕКСТОВОГО JIT-БЛОКА (RUNNER LOG):")
        print("·"*60)
        
        # Получаем область видимости из фрейма вызова (метод forward модели)
        caller_locals = sys._getframe(1).f_locals
        caller_globals = sys._getframe(1).f_globals
        
        # СОХРАНЯЕМ ИНДУСТРИАЛЬНОЕ ЛОГИРОВАНИЕ ОБЪЕКТОВ
        print("Обнаруженные объекты в области видимости forward():")
        for name, val in list(caller_locals.items()):
            if isinstance(val, Tensor) or hasattr(val, 'data_buffer'):
                print(f"  • [Tensor] {name:15} -> size: {val.size}, shape: {val.shape}")
            elif isinstance(val, (int, float)):
                print(f"  • [Scalar] {name:15} -> value: {val}")

        # Разбиваем текст на отдельные чистые инструкции
        expressions = [e.strip() for e in program_text.strip().split('\n') if e.strip()]
        # Сбрасываем кэш имен для текущего вызова блока
        self._captured_inputs.clear()
        self._captured_outputs.clear()       
        detected_tensors = {}    
        detected_scalars = {}    
        saved_tensors_vram = {}  
        cl_body_forward = []
        cl_body_backward = []

        cl_keywords = ['float', 'int', 'tid', 'unsigned', 'p', 'seed', 'p_val', 'seed_val', 'DROPOUT_P', 'DYNAMIC_SEED']

        print("\nКомпиляция строк выражений в Си-код:")
        for expr in expressions:
            # Убираем возможные двойные '==', оставляя строго чистые '='
            expr_clean = expr.replace('==', '=')
            if '=' not in expr_clean: continue
                
            # Легко отделяем целевую переменную (target) от функции (operation)
            target, operation = expr_clean.split('=', 1)
            target, operation = target.strip(), operation.strip()
            print(f"  • {target:12} = {operation}")

            cl_op = operation.replace('nnclloc.', 'nn_')
            while cl_op.count(')') > cl_op.count('('):
                last_bracket = cl_op.rfind(')')
                if last_bracket != -1: cl_op = cl_op[:last_bracket] + cl_op[last_bracket+1:]

            # =================================================================
            # ИЗОЛИРОВАННЫЙ БЛОК ДЛЯ DROPOUT С ДИНАМИЧЕСКИМИ РАЗМЕРАМИ
            # =================================================================
            if "nn_dropout" in cl_op:
                drop_match = re.search(r'nn_dropout\((.*?), (.*?), tid, (.*?)\)', cl_op)
                if drop_match:
                    inner_expr = drop_match.group(1).strip()
                    p_val = drop_match.group(2).strip()
                    seed_val = drop_match.group(3).strip()

                    tokens_inner = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', inner_expr)
                    cl_inner_fwd = inner_expr
                    for t in tokens_inner:
                        if t not in ['DROPOUT_P', 'DYNAMIC_SEED'] and not t.isdigit():
                            val_obj = caller_locals.get(t, caller_globals.get(t, None))
                            if val_obj is not None and (isinstance(val_obj, Tensor) or hasattr(val_obj, 'data_buffer')):
                                detected_tensors[t] = True
                                cl_inner_fwd = re.sub(r'\b' + t + r'\b', f"{t}[tid]", cl_inner_fwd)
                            else:
                                detected_scalars[t] = True

                    detected_tensors[target] = True
                    cl_body_forward.append(f"if (tid < {target}Size) {{ {target}[tid] = nn_dropout({cl_inner_fwd}, {p_val}, tid, {seed_val}); }}")

                    if self.is_training:
                        base_target_tensor = inner_expr.split(',')[0].strip()
                        cl_body_backward.append(f"if (tid < {target}Size) {{")
                        cl_body_backward.append(f"        float upstream_{target} = grad_{target}[tid];")
                        cl_body_backward.append(f"        float back_drop_{target} = nn_dropout_backward(upstream_{target}, {p_val}, tid, {seed_val});")
                        
                        if "*" in base_target_tensor:
                            u, v = [item.strip() for item in base_target_tensor.split('*')]
                            detected_tensors[u], detected_tensors[v] = True, True
                            cl_body_backward.append(f"        if (tid < {u}Size) grad_{u}[tid] += back_drop_{target} * {v}[tid];")
                            cl_body_backward.append(f"        if (tid < {v}Size) grad_{v}[tid] += back_drop_{target} * {u}[tid];")
                        elif "+" in base_target_tensor:
                            vars_add = [v.strip() for v in base_target_tensor.split('+')]
                           
                            for v in vars_add:
                                detected_tensors[v] = True
                                cl_body_backward.append(f"        if (tid < {v}Size) grad_{v}[tid] += back_drop_{target};")
                        else:
                            detected_tensors[base_target_tensor] = True
                            cl_body_backward.append(f"        if (tid < {base_target_tensor}Size) grad_{base_target_tensor}[tid] += back_drop_{target};")
                        cl_body_backward.append(f"    }}")
                    continue

            # =================================================================
            # СТАНДАРТНЫЙ БЛОК ОПЕРАЦИЙ (Сигмоиды, Произведения)
            # =================================================================
            tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', cl_op)
            if target in tokens and self.is_training:
                saved_name = f"saved_{target}"
                tensor_obj = caller_locals.get(target, caller_globals.get(target))
                vram_copy = cl.Buffer(self.ctx, cl.mem_flags.READ_WRITE, tensor_obj.nbytes)
                cl.enqueue_copy(self.queue, vram_copy, tensor_obj.data_buffer)
                saved_tensors_vram[saved_name] = vram_copy
            # Проверяем, что токен — это реальный Tensor в области видимости forward()
            
            # Регистрируем выходной тензор (изменился)
            if target in caller_locals and isinstance(caller_locals[target], Tensor):
                self._captured_outputs.add(target)
            # Строка 143 остается как есть (начало цикла):
            for t in tokens:
                if t.startswith('nn_') or t in cl_keywords or t.isdigit(): continue
                if t in caller_locals and isinstance(caller_locals[t], Tensor):
                    # Если тензор не является результатом текущей строки, значит он пришел на вход
                    if t != target:
                        self._autograd_ctx_inputs_helper(t, caller_locals)
                val_obj = caller_locals.get(t, caller_globals.get(t, None))
                if val_obj is not None and (isinstance(val_obj, Tensor) or hasattr(val_obj, 'data_buffer')):
                    detected_tensors[t] = True
                    # СТРОКА 149 — ИСПРАВЛЕНИЕ (Корректное регулярное выражение замены на f-строку с [tid]):
                    cl_op = re.sub(r'\b' + t + r'\b', f"{t}[tid]", cl_op)
                else:
                    detected_scalars[t] = True

            detected_tensors[target] = True
            # СТРОКА 154 — ИСПРАВЛЕНИЕ (Убираем лишние фигурные скобки, мешающие Си-компиляции):
            cl_body_forward.append(f"if (tid < {target}Size) {{ {target}[tid] = {cl_op}; }}")


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


        print("·"*60 + "\n")

        # Компиляция Си-кода
        kernel_hash = abs(hash(program_text))
        kernel_name = f"text_forward_{kernel_hash}"
        kernel_args = []
        # Объединяем весь сгенерированный Си-код в одну строку для поиска подстрок
        compiled_code_str = "\n".join(cl_body_forward) + "\n".join(cl_body_backward)

        for tensor in detected_tensors.keys():
            # ИСПРАВЛЕНИЕ: Проверяем, что имя тензора действительно используется в кодовой строке ядра!
            if re.search(r'\b' + tensor + r'\b', compiled_code_str):
                kernel_args.append(f"__global float* {tensor}")
                kernel_args.append(f"const int {tensor}Size")
                if self.is_training:
                    kernel_args.append(f"__global float* grad_{tensor}")
        
        for scalar in detected_scalars.keys():
            # ИСПРАВЛЕНИЕ: Проверяем использование скаляра
            if re.search(r'\b' + scalar + r'\b', compiled_code_str):
                kernel_args.append(f"const float {scalar}")

        for saved_name in saved_tensors_vram.keys():
            if re.search(r'\b' + saved_name + r'\b', compiled_code_str):
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
        if 'DROPOUT_P' in caller_locals: 
            full_forward_code = full_forward_code.replace('DROPOUT_P', f"{float(caller_locals['DROPOUT_P'])}f")
        if 'DYNAMIC_SEED' in caller_locals: 
            full_forward_code = full_forward_code.replace('DYNAMIC_SEED', f"{int(caller_locals['DYNAMIC_SEED'])}")
        program = JITProgramCache.get_or_compile(self.ctx, full_forward_code)
        kernel = getattr(program, kernel_name)
        max_elements = 1
        cl_buffers_forward = []

        # === ВНУТРИ compiler/nncl_compiler.py (Сборка буферов для вызова ядра) ===
        max_elements = 1
        cl_buffers_forward = []
        
        for tensor in detected_tensors.keys():
            # ИСПРАВЛЕНИЕ: Передаем буфер на GPU только если он есть в кодовой строке ядра!
            if re.search(r'\b' + tensor + r'\b', compiled_code_str):
                tensor_obj = caller_locals.get(tensor, caller_globals.get(tensor))
                if tensor_obj.size > max_elements:
                    max_elements = tensor_obj.size
                cl_buffers_forward.append(tensor_obj.data_buffer)
                cl_buffers_forward.append(np.int32(tensor_obj.size))
                if self.is_training:
                    tensor_obj.require_grad()
                    cl_buffers_forward.append(tensor_obj.grad_buffer)

        for scalar in detected_scalars.keys():
            # ИСПРАВЛЕНИЕ: Передаем скаляр только если он есть в кодовой строке ядра!
            if re.search(r'\b' + scalar + r'\b', compiled_code_str):
                scalar_val = caller_locals.get(scalar, caller_globals.get(scalar))
                cl_buffers_forward.append(np.float32(scalar_val))

        cl_buffers_call_fwd = list(cl_buffers_forward)
        for saved_name, vram_buffer in saved_tensors_vram.items():
            # ИСПРАВЛЕНИЕ: Проверяем сохраненные in-place кэши
            if re.search(r'\b' + saved_name + r'\b', compiled_code_str):
                cl_buffers_call_fwd.append(vram_buffer)


        
        total_count = int(np.ceil(max_elements / float(self.warp_size)) * self.warp_size)
        global_size = (total_count // self.warp_size, self.warp_size)
        kernel(self.queue, global_size, None, *cl_buffers_call_fwd)
        self.queue.finish()
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
            # Проверяем, зарегистрирован ли у выходных тензоров autograd-контекст (очередь модели)
            # Нам достаточно вытащить ссылку на autograd_ctx из любого захваченного тензора
            active_autograd_ctx = None
            for out_name in self._captured_outputs:
                tensor_obj = caller_locals[out_name]
                if hasattr(tensor_obj, 'autograd_ctx') and tensor_obj.autograd_ctx is not None:
                    active_autograd_ctx = tensor_obj.autograd_ctx
                    break
            if active_autograd_ctx is not None:
                            # Оборачиваем выполнение конкретно ЭТОГО шага в ленивый callback
                            # Замыкание захватывает self (текущий экземпляр runner), который хранит свой backward_stack
                            def runner_step_backward_closure():
                                # Запускаем ваш оригинальный run_backward(), который достанет 
                                # из backward_stack именно это Си-ядро и его физические OpenCL-буферы!
                                self.run_backward()

                            # Помещаем callback в общую ленту вычислений модели
                            active_autograd_ctx.register_backward(runner_step_backward_closure)            
        if 'DROPOUT_P' in caller_locals: 
            full_backward_code = full_backward_code.replace('DROPOUT_P', f"{float(caller_locals['DROPOUT_P'])}f")
        if 'DYNAMIC_SEED' in caller_locals: 
            full_backward_code = full_backward_code.replace('DYNAMIC_SEED', f"{int(caller_locals['DYNAMIC_SEED'])}")
        cl_buffers_backward = list(cl_buffers_forward)

        for saved_name, vram_buffer in saved_tensors_vram.items():
            if re.search(r'\b' + saved_name + r'\b', compiled_code_str):
                cl_buffers_backward.append(vram_buffer)

        self.backward_stack.append((full_backward_code, backward_kernel_name, cl_buffers_backward, global_size))


    def run_backward(self):
        if not self.backward_stack: 
            return
        while self.backward_stack:
            code, name, buffers, global_size = self.backward_stack.pop()
            program = JITProgramCache.get_or_compile(self.ctx, code)
            kernel = getattr(program, name)
            kernel(self.queue, global_size, None, *buffers)
            self.queue.finish()
    def get_tensor(self, name: str) -> Tensor:
            """
            Возвращает объект Tensor по его текстовому имени, 
            который участвовал в вычислениях последнего JIT-ядра.
            """
            # Сначала ищем в сохраненном кэше локальных переменных вызова модели
            caller_locals = getattr(self, '_active_caller_locals', None)
            if caller_locals is None:
                # Если кэша нет, заглядываем на один фрейм вверх напрямую
                caller_locals = sys._getframe(1).f_locals
                
            val_obj = caller_locals.get(name, None)
            if val_obj is None:
                raise NameError(f"Тензор с именем '{name}' не найден в области видимости вызова forward().")
            return val_obj
    def _autograd_ctx_inputs_helper(self, token_name, caller_locals):
        """Вспомогательный метод для фильтрации входных тензоров, требующих градиент"""
        tensor_obj = caller_locals[token_name]
        if tensor_obj.grad_buffer is not None:
            self._captured_inputs.add(token_name)

    def __call2__(self, program_text: str):
        # 1. Захватываем область видимости из фрейма вызова forward()
        caller_locals = sys._getframe(1).f_locals
        caller_globals = sys._getframe(1).f_globals
        self._captured_locals = caller_locals

        # Сбрасываем кэш захваченных тензоров
        self._captured_inputs.clear()
        self._captured_outputs.clear()

        # Разделяем входную строку по символу переноса "\n"
        raw_lines = program_text.strip().split('\n')
        compiled_statements = []
        detected_tensors_set = set()

        for line in raw_lines:
            cleaned_line = line.strip()
            if not cleaned_line or cleaned_line.startswith('#'):
                continue

            # Заменяем все "nnclloc." на "nn_"
            cleaned_line = cleaned_line.replace('nnclloc.', 'nn_')

            # Вплавляем значения скалярных параметров (например, q = 0.23 -> /=0.23)
            # Ищем потенциальные имена переменных в строке
            tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', cleaned_line)
            for token in tokens:
                if token in ['nn_', 'sigmoid', 'tanh', 'dropout', 'tid']:
                    continue
                # Если токен найден в локальных переменных и это число (int, float)
                if token in caller_locals and isinstance(caller_locals[token], (int, float)):
                    val = caller_locals[token]
                    # Безопасно заменяем имя переменной на её числовое значение
                    cleaned_line = re.sub(r'\b' + token + r'\b', f"{val}", cleaned_line)

            # --- БЛОК ВАЛИДАЦИИ ИНСТРУКЦИИ ---
            
            # Шаг 1: Проверяем наличие хотя бы одного реального тензора в строке
            line_tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', cleaned_line)
            has_tensor = False
            for t in line_tokens:
                if t in caller_locals and isinstance(caller_locals[t], Tensor):
                    has_tensor = True
                    detected_tensors_set.add(t)
            
            if not has_tensor:
                raise ValueError(f"Ошибка валидации: Строка '{line}' не содержит ни одного тензора.")

            target_tensor_name = None

            # Шаг 2: Проверяем наличие знака равенства "="
            if '=' not in cleaned_line:
                # Если знака равенства нет, должно быть строго ++ или --
                if '++' not in cleaned_line and '--' not in cleaned_line:
                    raise SyntaxError(f"Ошибка валидации: Строка '{line}' не содержит знака '=' или инкремента/декремента (++ / --).")
                
                # Извлекаем имя тензора, к которому применен инкремент/декремент
                match_inc_dec = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b\s*(?:\+\+|--)', cleaned_line)
                if match_inc_dec:
                    target_tensor_name = match_inc_dec.group(1)
            else:
                # Шаг 3: Разделяем строку по знаку равенства
                left_part, right_part = cleaned_line.split('=', 1)
                left_part = left_part.strip()
                right_part = right_part.strip()

                # Проверяем комбинированные операторы присваивания (+=, -=, *=, /=, %=)
                # Если перед '=' стоит один из знаков арифметики
                if left_part and left_part[-1] in ['+', '-', '*', '/', '%']:
                    target_tensor_name = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', left_part)[-1]
                else:
                    # Шаг 4: Чистое присваивание '=', проверяем правую часть на валидность данных
                    target_tensor_name = left_part
                    
                    # Извлекаем все буквенные токены из правой части, кроме вызовов функций nn_
                    right_tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', right_part)
                    for rt in right_tokens:
                        if rt.startswith('nn_') or rt in ['sigmoid', 'tanh', 'dropout', 'tid']:
                            continue
                        # Если в правой части осталась строка, которая не является тензором и не была вплавлена как число
                        if rt in caller_locals and isinstance(caller_locals[rt], Tensor):
                            continue
                        raise TypeError(f"Ошибка валидации: В правой части выражения '{line}' обнаружено недопустимое строковое значение '{rt}'.")

            if not target_tensor_name:
                raise ValueError(f"Ошибка валидации: Не удалось определить целевой тензор в строке '{line}'.")

            self._captured_outputs.add(target_tensor_name)

        # === РЕАЛИЗАЦИЯ УМНОГО ВЕКТОРНОГО ФИЛЬТРА ЗАМЕН ДЛЯ ТЕНЗОРОВ ===
        # Набор операторов и символов из вашего ТЗ (включая пробел)
        operators = [' ', '-', '+', '=', ',', '(', '/', '%', ')']
        
        # Обходим уникальные имена тензоров, обнаруженные в строке
        for t in detected_tensors_set:
            # 1. Формируем вектор пар: имя + оператор/символ
            pattern_vector = [f"{t}{op}" for op in operators]
            
            # 2. Формируем параллельный вектор замен: имя[tid] + оператор/символ
            replacement_vector = [f"{t}[tid]{op}" for op in operators]
            
            # 3. Выполняем алгоритм контекстной замены:
            # Пока комбинация имя+оператор встречается в строке, производим замену
            for pattern, replacement in zip(pattern_vector, replacement_vector):
                while pattern in cleaned_line:
                    cleaned_line = cleaned_line.replace(pattern, replacement)

        # Оборачиваем валидную строку в Си-конструкцию с проверкой границ
        compiled_statement = f"if (tid < {target_tensor_name}Size) {{ {cleaned_line}; }}"
        compiled_statements.append(compiled_statement)

        # Собираем входные тензоры-аргументы (все задействованные тензоры минус те, что перезаписаны)
        for t in detected_tensors_set:
            if t in caller_locals and caller_locals[t].grad_buffer is not None:
                if t not in self._captured_outputs:
                    self._captured_inputs.add(t)

        # --- СБОРКА ФИНАЛЬНОГО ТЕКСТА КЕРНЕЛА ---
        kernel_hash = abs(hash(program_text))
        kernel_name = f"text_forward_{kernel_hash}"

        # Формируем список аргументов ядра динамически на основе обнаруженных тензоров
        kernel_args = []
        cl_buffers_call_fwd = []
        max_elements = 1

        for tensor_name in sorted(list(detected_tensors_set)):
            tensor_obj = caller_locals[tensor_name]
            if tensor_obj.size > max_elements:
                max_elements = tensor_obj.size

            # Добавляем указатель на буфер данных и переменную размера
            kernel_args.append(f"__global float* {tensor_name}")
            kernel_args.append(f"const int {tensor_name}Size")
            cl_buffers_call_fwd.append(tensor_obj.data_buffer)
            cl_buffers_call_fwd.append(np.int32(tensor_obj.size))

        args_str = ", ".join(kernel_args)
        forward_statements_str = "\n    ".join(compiled_statements)

        # Собираем монолитный Си-код
        full_forward_code = f"""
{BASE_CL_FUNCTIONS}

__kernel void {kernel_name}({args_str}) {{
    int tid = {self.warp_size} * get_global_id(0) + get_global_id(1);
    {forward_statements_str}
}}
"""
        # Сборка сетки потоков и выполнение на GPU
        total_count = int(np.ceil(max_elements / float(self.warp_size)) * self.warp_size)
        global_size = (total_count // self.warp_size, self.warp_size)

        program = JITProgramCache.get_or_compile(self.ctx, full_forward_code)
        kernel = getattr(program, kernel_name)
        kernel(self.queue, global_size, None, *cl_buffers_call_fwd)
        self.queue.finish()






