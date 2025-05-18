# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.20)

# 定义项目信息
project(
  "compiler"
  DESCRIPTION "dsp ai compiler"
  LANGUAGES CXX C
)

# 设置编译选项
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# 保存原始模块路径
set(ORIGINAL_CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}")
set(LLVM_DIR "/opt/llvm-19/lib/cmake/llvm")
set(MLIR_DIR "/opt/llvm-19/lib/cmake/mlir")
message(STATUS "LLVM_DIR: ${LLVM_DIR}")
message(STATUS "MLIR_DIR: ${MLIR_DIR}")

# 查找包
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
message(STATUS "Found MLIR ${MLIR_VERSION_MAJOR}.${MLIR_VERSION_MINOR}.${MLIR_VERSION_PATCH}")
message(STATUS "LLVM_CMAKE_DIR: ${LLVM_CMAKE_DIR}")
message(STATUS "MLIR_CMAKE_DIR: ${MLIR_CMAKE_DIR}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")

# 重置并重新设置模块路径
set(CMAKE_MODULE_PATH "${ORIGINAL_CMAKE_MODULE_PATH}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
message(STATUS "Final CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# 包含模块
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# 全局包含目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/lib)

# 移除不需要的编译选项
string(REPLACE "-fno-lifetime-dse" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE "-fno-lifetime-dse" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

# 添加子目录
add_subdirectory(include)
add_subdirectory(lib)

# 创建主程序
add_executable(compiler driver.cpp conv2d.cpp)

get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# 设置主程序的链接库
target_link_libraries(compiler
  PRIVATE
  ${MLIR_DIALECT_LIBS}
  ${MLIR_CONVERSION_LIBS}
  ${MLIR_EXTENSION_LIBS}
  ${MLIR_TRANSLATION_LIBS}
  MLIRIR
  MLIRParser
  MLIRAnalysis
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRAffineAnalysis
  MLIRCastInterfaces
  MLIRDialect
  MLIROptLib
  MLIRSupport
  # LLVM 组件
  LLVMPasses
)