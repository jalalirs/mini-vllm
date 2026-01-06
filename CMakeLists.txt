cmake_minimum_required(VERSION 3.18)
project(mini_vllm_cuda LANGUAGES CXX CUDA)

# =============================================================================
# H100 CUDA Configuration
# =============================================================================
set(CMAKE_CUDA_ARCHITECTURES 90)  # H100 = SM90
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Find PyTorch
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX})
find_package(Torch REQUIRED)

# =============================================================================
# Compiler Flags
# =============================================================================
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90,code=sm_90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_90a,code=sm_90a")

# Half precision support
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUDA_HAS_FP16=1")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -U__CUDA_NO_HALF_OPERATORS__")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -U__CUDA_NO_HALF_CONVERSIONS__")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -U__CUDA_NO_BFLOAT16_CONVERSIONS__")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -U__CUDA_NO_HALF2_OPERATORS__")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# =============================================================================
# Include Directories
# =============================================================================
set(CSRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mini_vllm/csrc)

include_directories(
    ${CSRC_DIR}
    ${CSRC_DIR}/attention
    ${CSRC_DIR}/core
    ${CSRC_DIR}/moe
    ${CSRC_DIR}/quantization
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# =============================================================================
# CUDA Source Files
# =============================================================================
set(CUDA_SOURCES
    # Core ops
    ${CSRC_DIR}/torch_bindings.cpp
    ${CSRC_DIR}/layernorm_kernels.cu
    ${CSRC_DIR}/activation_kernels.cu
    ${CSRC_DIR}/pos_encoding_kernels.cu
    ${CSRC_DIR}/cache_kernels.cu
    ${CSRC_DIR}/cuda_utils_kernels.cu
    
    # Attention
    ${CSRC_DIR}/attention/paged_attention_v1.cu
    ${CSRC_DIR}/attention/paged_attention_v2.cu
    ${CSRC_DIR}/attention/merge_attn_states.cu
    
    # MoE
    ${CSRC_DIR}/moe/topk_softmax_kernels.cu
    ${CSRC_DIR}/moe/moe_align_sum_kernels.cu
    ${CSRC_DIR}/moe/grouped_topk_kernels.cu
)

# =============================================================================
# Build Shared Library
# =============================================================================
add_library(mini_vllm_cuda SHARED ${CUDA_SOURCES})

target_link_libraries(mini_vllm_cuda
    ${TORCH_LIBRARIES}
    ${Python3_LIBRARIES}
    CUDA::cudart
)

# Set output name and location
set_target_properties(mini_vllm_cuda PROPERTIES
    OUTPUT_NAME "_C"
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mini_vllm
)

# For torch extension compatibility
target_compile_definitions(mini_vllm_cuda PRIVATE
    TORCH_EXTENSION_NAME=mini_vllm_ops
)

# =============================================================================
# Install
# =============================================================================
install(TARGETS mini_vllm_cuda
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/mini_vllm
)

# =============================================================================
# Print Configuration
# =============================================================================
message(STATUS "")
message(STATUS "=== Mini-vLLM CUDA Build Configuration ===")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "PyTorch: ${TORCH_CMAKE_PREFIX}")
message(STATUS "Python: ${Python3_EXECUTABLE}")
message(STATUS "Output: ${CMAKE_CURRENT_SOURCE_DIR}/mini_vllm/_C.so")
message(STATUS "==========================================")
message(STATUS "")

