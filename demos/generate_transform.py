#!/usr/bin/env python3
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re

@dataclass
class TilingInfo:
    dimension: str  # "K", "M", "N"
    size: int

@dataclass
class CacheOperation:
    matrix: str           # "A", "B", "C"
    source_memory: str    # "ddr", "gsm", "am", "sm"
    target_memory: str    # "gsm", "am", "sm", "ddr"
    multi_buffer: bool
    is_result: bool = False  # 是否是result操作(用于cache_write)

@dataclass
class ControlOperation:
    type: str    # "unroll", "parallel"
    factor: int

@dataclass
class LayerInfo:
    original_line: str  # 保留原始输入行
    tiling: Optional[TilingInfo]
    cache_ops: List[CacheOperation]
    controls: List[ControlOperation]

class ScheduleParser:
    def parse_schedule_from_file(self, filename: str) -> List[LayerInfo]:
        """从文件读取并解析调度描述"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.parse_schedule(content)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found!")
            return []
        except Exception as e:
            print(f"Error reading file '{filename}': {e}")
            return []
    
    def parse_schedule(self, text: str) -> List[LayerInfo]:
        lines = []
        for line in text.strip().split('\n'):
            line = line.strip()
            # 忽略空行和以//开头的注释行
            if line and not line.startswith('//'):
                lines.append(line)
        return [self._parse_layer(line) for line in lines]
    
    def _parse_layer(self, line: str) -> LayerInfo:
        tiling = self._extract_tiling(line)
        cache_ops = self._extract_cache_operations(line)
        controls = self._extract_controls(line)
        return LayerInfo(line, tiling, cache_ops, controls)
    
    def _extract_tiling(self, line: str) -> Optional[TilingInfo]:
        match = re.match(r'^([KMN])\s+(\d+)', line)
        if match:
            return TilingInfo(match.group(1), int(match.group(2)))
        return None
    
    def _extract_cache_operations(self, line: str) -> List[CacheOperation]:
        # 匹配 A->A_gsm(multi_buffer) 或 C_am->C(multi_buffer) 等
        cache_pattern = r'([ABC](?:_\w+)?)->([ABC](?:_\w+)?)\(([^)]*)\)'
        matches = re.findall(cache_pattern, line)
        
        ops = []
        for source_full, target_full, flags in matches:
            multi_buffer = 'multi_buffer' in flags
            
            # 解析矩阵和内存层次
            matrix = source_full[0]  # A, B, C
            source_memory = self._extract_memory(source_full)
            target_memory = self._extract_memory(target_full)
            
            # 判断是否是result操作(target是原始矩阵名)
            is_result = len(target_full) == 1  # C_am->C 中的 C
            
            ops.append(CacheOperation(matrix, source_memory, target_memory, multi_buffer, is_result))
        
        return ops
    
    def _extract_memory(self, var_name: str) -> str:
        if '_' in var_name:
            return var_name.split('_')[1]
        return 'ddr'  # 默认内存层次
    
    def _extract_controls(self, line: str) -> List[ControlOperation]:
        controls = []
        
        # unroll(factor)
        unroll_match = re.search(r'unroll\((\d+)\)', line)
        if unroll_match:
            controls.append(ControlOperation('unroll', int(unroll_match.group(1))))
        
        # parallel
        if 'parallel' in line:
            controls.append(ControlOperation('parallel', 8))  # 默认8线程
        
        return controls

class VariableNamer:
    def __init__(self):
        # 跟踪每个维度的当前分块大小，None表示未分块
        self.current_sizes = {"M": None, "N": None, "K": None}
    
    def get_matmul_op_name(self) -> str:
        if all(size is None for size in self.current_sizes.values()):
            return "%0"
        
        # 构建名称，未分块的维度显示维度名，已分块的显示大小
        parts = []
        for dim in ["M", "N", "K"]:
            if self.current_sizes[dim] is None:
                parts.append(dim)
            else:
                parts.append(f"{dim}{self.current_sizes[dim]}")
        
        return f"%matmul_{'_'.join(parts)}"
    
    def get_loop_name(self, dimension: str, size: int) -> str:
        return f"%{dimension}{size}"
    
    def get_matrix_var_name(self, matrix: str, memory: str, is_result: bool = False) -> str:
        if is_result:
            return f"%{matrix}_result_{memory}"
        return f"%{matrix}_{memory}"
    
    def update_tiling(self, dimension: str, size: int):
        self.current_sizes[dimension] = size

class MemoryTracker:
    def __init__(self):
        self.current_memory = {"A": "ddr", "B": "ddr", "C": "ddr"}
    
    def get_current_memory(self, matrix: str) -> str:
        return self.current_memory[matrix]
    
    def update_memory(self, matrix: str, new_memory: str):
        self.current_memory[matrix] = new_memory

class MLIRCodeGenerator:
    def __init__(self, namer: VariableNamer, memory_tracker: MemoryTracker):
        self.namer = namer
        self.memory_tracker = memory_tracker
    
    def generate_module_header(self) -> str:
        return """module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op"""
    
    def generate_module_footer(self) -> str:
        return """    transform.yield
  }
}"""
    
    def generate_tiling_code(self, current_op: str, tiling: TilingInfo, original_line: str) -> Tuple[str, str]:
        # 计算tile_sizes [M, N, K]
        tile_sizes = [0, 0, 0]
        if tiling.dimension == "M":
            tile_sizes[0] = tiling.size
        elif tiling.dimension == "N":
            tile_sizes[1] = tiling.size
        elif tiling.dimension == "K":
            tile_sizes[2] = tiling.size
        
        # 生成变量名
        loop_name = self.namer.get_loop_name(tiling.dimension, tiling.size)
        self.namer.update_tiling(tiling.dimension, tiling.size)
        new_op_name = self.namer.get_matmul_op_name()
        
        code = f"""    // {original_line}
    {new_op_name}, {loop_name} = transform.structured.tile_using_for {current_op} tile_sizes {tile_sizes} interchange = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)"""
        
        return new_op_name, loop_name, code
    
    def generate_cache_operations(self, current_op: str, cache_ops: List[CacheOperation]) -> str:
        code_parts = []
        
        for cache_op in cache_ops:
            if cache_op.is_result:
                # cache_write操作
                result_var = self.namer.get_matrix_var_name(cache_op.matrix, cache_op.source_memory, True)
                target_var = self.namer.get_matrix_var_name(cache_op.matrix, cache_op.target_memory)
                multi_buffer_str = "true" if cache_op.multi_buffer else "false"
                
                code_parts.append(f"    {result_var} = transform.get_result {current_op}[0] : (!transform.any_op) -> !transform.any_value")
                code_parts.append(f"    transform.structured.cache_write {result_var}, {target_var} multi_buffer = {multi_buffer_str} {{memory_space = #mtdsp.address_space<{cache_op.target_memory}>}} : (!transform.any_value, !transform.any_value) -> !transform.any_op")
            else:
                # cache_read操作
                operand_index = {"A": 0, "B": 1, "C": 2}[cache_op.matrix]
                source_var = self.namer.get_matrix_var_name(cache_op.matrix, cache_op.source_memory)
                multi_buffer_str = "true" if cache_op.multi_buffer else "false"
                
                code_parts.append(f"    {source_var} = transform.get_operand {current_op}[{operand_index}] : (!transform.any_op) -> !transform.any_value")
                code_parts.append(f"    transform.structured.cache_read {source_var} multi_buffer = {multi_buffer_str} {{memory_space = #mtdsp.address_space<{cache_op.target_memory}>}} : (!transform.any_value) -> !transform.any_op")
                
                # 更新内存状态
                self.memory_tracker.update_memory(cache_op.matrix, cache_op.target_memory)
        
        return '\n'.join(code_parts)
    
    def generate_control_operations(self, loop_var: str, controls: List[ControlOperation]) -> str:
        code_parts = []
        
        for control in controls:
            if control.type == "unroll":
                code_parts.append(f"    transform.mark_unroll {loop_var} unroll_factor = {control.factor} : (!transform.any_op) -> !transform.any_op")
            elif control.type == "parallel":
                code_parts.append(f"    transform.mark_parallel {loop_var} num_threads = {control.factor} : (!transform.any_op) -> !transform.any_op")
        
        return '\n'.join(code_parts)

class TransformGenerator:
    def __init__(self):
        self.parser = ScheduleParser()
        self.namer = VariableNamer()
        self.memory_tracker = MemoryTracker()
        self.code_gen = MLIRCodeGenerator(self.namer, self.memory_tracker)
    
    def generate(self, schedule_text: str) -> str:
        layers = self.parser.parse_schedule(schedule_text)
        
        code_parts = [self.code_gen.generate_module_header()]
        current_op = "%0"
        current_loop = None
        
        for layer in layers:
            # 1. 生成分块代码
            if layer.tiling:
                current_op, current_loop, tiling_code = self.code_gen.generate_tiling_code(current_op, layer.tiling, layer.original_line)
                code_parts.append(tiling_code)
            
            # 2. 生成缓存代码
            if layer.cache_ops:
                cache_code = self.code_gen.generate_cache_operations(current_op, layer.cache_ops)
                if cache_code:
                    code_parts.append(cache_code)
            
            # 3. 生成控制代码
            if layer.controls and current_loop:
                control_code = self.code_gen.generate_control_operations(current_loop, layer.controls)
                if control_code:
                    code_parts.append(control_code)
        
        code_parts.append(self.code_gen.generate_module_footer())
        return '\n'.join(code_parts)
    
    def generate_from_file(self, filename: str) -> str:
        """从文件生成transform dialect代码"""
        layers = self.parser.parse_schedule_from_file(filename)
        if not layers:
            return ""
        
        code_parts = [self.code_gen.generate_module_header()]
        current_op = "%0"
        current_loop = None
        
        for layer in layers:
            # 1. 生成分块代码
            if layer.tiling:
                current_op, current_loop, tiling_code = self.code_gen.generate_tiling_code(current_op, layer.tiling, layer.original_line)
                code_parts.append(tiling_code)
            
            # 2. 生成缓存代码
            if layer.cache_ops:
                cache_code = self.code_gen.generate_cache_operations(current_op, layer.cache_ops)
                if cache_code:
                    code_parts.append(cache_code)
            
            # 3. 生成控制代码
            if layer.controls and current_loop:
                control_code = self.code_gen.generate_control_operations(current_loop, layer.controls)
                if control_code:
                    code_parts.append(control_code)
        
        code_parts.append(self.code_gen.generate_module_footer())
        return '\n'.join(code_parts)
        layers = self.parser.parse_schedule(schedule_text)
        
        code_parts = [self.code_gen.generate_module_header()]
        current_op = "%0"
        current_loop = None
        
        for layer in layers:
            # 1. 生成分块代码
            if layer.tiling:
                current_op, current_loop, tiling_code = self.code_gen.generate_tiling_code(current_op, layer.tiling, layer.original_line)
                code_parts.append(tiling_code)
            
            # 2. 生成缓存代码
            if layer.cache_ops:
                cache_code = self.code_gen.generate_cache_operations(current_op, layer.cache_ops)
                if cache_code:
                    code_parts.append(cache_code)
            
            # 3. 生成控制代码
            if layer.controls and current_loop:
                control_code = self.code_gen.generate_control_operations(current_loop, layer.controls)
                if control_code:
                    code_parts.append(control_code)
        
        code_parts.append(self.code_gen.generate_module_footer())
        return '\n'.join(code_parts)

def main():
    generator = TransformGenerator()
    result = generator.generate_from_file("demos/schedule.mlir")
    
    if not result:
        print("Failed to generate transform dialect. Please check the input file.")
        return
    
    # 输出到文件
    with open("demos/transform.mlir", "w") as f:
        f.write(result)
    
    print("Transform dialect generated successfully!")
    # print("Input file: schedule.mlir")
    # print("Output file: transform.mlir")
    # print("\n" + "="*50)
    # print(result)

if __name__ == "__main__":
    main()