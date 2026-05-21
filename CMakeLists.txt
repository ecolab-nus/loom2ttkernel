cmake_minimum_required(VERSION 3.20.0)

project(loom2ttkernel LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(MLIR_DIR "/opt/ttmlir-toolchain/lib/cmake/mlir" CACHE PATH
    "Path to MLIRConfig.cmake")
set(LLVM_DIR "/opt/ttmlir-toolchain/lib/cmake/llvm" CACHE PATH
    "Path to LLVMConfig.cmake")
set(LLVM_USE_LINKER "lld" CACHE STRING "Linker to use")

set(LOOM_SOURCE_DIR "/root/loom-dataflow" CACHE PATH
    "Path to the Loom source checkout")
set(LOOM_BUILD_DIR "${LOOM_SOURCE_DIR}/build" CACHE PATH
    "Path to an already configured/built Loom build directory")
set(TTMLIR_SOURCE_DIR "/root/tt-mlir" CACHE PATH
    "Path to the tt-mlir source checkout")
set(TTMLIR_BUILD_DIR "${TTMLIR_SOURCE_DIR}/build" CACHE PATH
    "Path to an already configured/built tt-mlir build directory")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using Loom source dir: ${LOOM_SOURCE_DIR}")
message(STATUS "Using Loom build dir: ${LOOM_BUILD_DIR}")
message(STATUS "Using tt-mlir source dir: ${TTMLIR_SOURCE_DIR}")
message(STATUS "Using tt-mlir build dir: ${TTMLIR_BUILD_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

function(require_path path description)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR
      "Missing ${description}: ${path}\n"
      "Set the matching CMake cache variable or build the dependency first.")
  endif()
endfunction()

function(import_static_library target archive)
  require_path("${archive}" "${target} static library")
  add_library(${target} STATIC IMPORTED GLOBAL)
  set_target_properties(${target} PROPERTIES IMPORTED_LOCATION "${archive}")
endfunction()

set(TTMLIR_INCLUDE_DIR "${TTMLIR_SOURCE_DIR}/include")
set(TTMLIR_BUILD_INCLUDE_DIR "${TTMLIR_BUILD_DIR}/include")
set(TTMLIR_LIB_DIR "${TTMLIR_BUILD_DIR}/lib")
set(LOOM_DIALECT_INC_DIR "${LOOM_BUILD_DIR}/lib/loom-dialect/IR")
set(ADL_DIALECT_INC_DIR "${LOOM_BUILD_DIR}/lib/adl-dialect/IR")
set(LOOM_LIB_DIR "${LOOM_BUILD_DIR}/lib")

require_path("${TTMLIR_INCLUDE_DIR}/ttmlir/Dialect/TTKernel/IR/TTKernel.h"
             "tt-mlir TTKernel source headers")
require_path("${TTMLIR_BUILD_INCLUDE_DIR}/ttmlir/Dialect/TTKernel/IR/TTKernelOpsDialect.h.inc"
             "tt-mlir generated TTKernel headers")
require_path("${LOOM_DIALECT_INC_DIR}/LoomOps.h.inc"
             "generated Loom dialect headers")
require_path("${LOOM_DIALECT_INC_DIR}/LoomInterfaces.h.inc"
             "generated Loom interface headers")
require_path("${ADL_DIALECT_INC_DIR}/ADLDialect.h.inc"
             "generated ADL dialect headers")

import_static_library(loomDialect "${LOOM_LIB_DIR}/libloomDialect.a")
import_static_library(loomADLDialect "${LOOM_LIB_DIR}/libloomADLDialect.a")
import_static_library(TTMLIRTTKernelDialect
                      "${TTMLIR_LIB_DIR}/libMLIRTTKernelDialect.a")
import_static_library(TTMLIRTTCoreDialect
                      "${TTMLIR_LIB_DIR}/libMLIRTTCoreDialect.a")
import_static_library(TTMLIRTTMetalDialect
                      "${TTMLIR_LIB_DIR}/libMLIRTTMetalDialect.a")

target_include_directories(loomDialect INTERFACE "${LOOM_DIALECT_INC_DIR}")
target_include_directories(loomADLDialect INTERFACE "${ADL_DIALECT_INC_DIR}")
target_include_directories(TTMLIRTTKernelDialect INTERFACE
  "${TTMLIR_INCLUDE_DIR}"
  "${TTMLIR_BUILD_INCLUDE_DIR}"
)
target_include_directories(TTMLIRTTCoreDialect INTERFACE
  "${TTMLIR_INCLUDE_DIR}"
  "${TTMLIR_BUILD_INCLUDE_DIR}"
)
target_include_directories(TTMLIRTTMetalDialect INTERFACE
  "${TTMLIR_INCLUDE_DIR}"
  "${TTMLIR_BUILD_INCLUDE_DIR}"
)

set(LOOM2TTKERNEL_SOURCES
  src/TTKernelUtils.cpp
  src/RuntimeArgLayout.cpp
  src/MemoryOpToTTKernel.cpp
  src/ComputeOpToTTKernel.cpp
  src/SCFOpToTTKernel.cpp
  src/FuncOpToTTKernel.cpp
  src/TileLoomToTTKernel.cpp
)

set(LOOM2TTKERNEL_MLIR_LIBS
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsmParser
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRBytecodeReader
  MLIRDestinationStyleOpInterface
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTransforms
  MLIRViewLikeInterface
)

set(LOOM2TTKERNEL_AVAILABLE_MLIR_LIBS)
foreach(lib IN LISTS LOOM2TTKERNEL_MLIR_LIBS)
  if(TARGET ${lib})
    list(APPEND LOOM2TTKERNEL_AVAILABLE_MLIR_LIBS ${lib})
  endif()
endforeach()

add_library(loomTileLoomToTTKernel STATIC ${LOOM2TTKERNEL_SOURCES})

target_include_directories(loomTileLoomToTTKernel PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
  ${MLIR_INCLUDE_DIRS}
  ${TTMLIR_INCLUDE_DIR}
  ${TTMLIR_BUILD_INCLUDE_DIR}
  ${LOOM_DIALECT_INC_DIR}
)

target_link_libraries(loomTileLoomToTTKernel PUBLIC
  ${LOOM2TTKERNEL_AVAILABLE_MLIR_LIBS}
  LLVMSupport
  loomDialect
  TTMLIRTTKernelDialect
  TTMLIRTTCoreDialect
  TTMLIRTTMetalDialect
)

target_compile_options(loomTileLoomToTTKernel PRIVATE -fno-rtti)
mlir_check_all_link_libraries(loomTileLoomToTTKernel)

add_executable(tileloom_to_ttkernel_opt
  tool/tileloom_to_ttkernel_opt.cpp
)

target_include_directories(tileloom_to_ttkernel_opt PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/inc
  ${MLIR_INCLUDE_DIRS}
  ${TTMLIR_INCLUDE_DIR}
  ${TTMLIR_BUILD_INCLUDE_DIR}
  ${LOOM_DIALECT_INC_DIR}
  ${ADL_DIALECT_INC_DIR}
)

target_link_libraries(tileloom_to_ttkernel_opt PRIVATE
  loomTileLoomToTTKernel
  loomADLDialect
  loomDialect
  MLIRMlirOptMain
  ${LOOM2TTKERNEL_AVAILABLE_MLIR_LIBS}
  LLVMSupport
  TTMLIRTTKernelDialect
  TTMLIRTTCoreDialect
  TTMLIRTTMetalDialect
)

target_compile_options(tileloom_to_ttkernel_opt PRIVATE -fno-rtti)
mlir_check_all_link_libraries(tileloom_to_ttkernel_opt)

set_target_properties(tileloom_to_ttkernel_opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
