cmake_minimum_required(VERSION 3.26)

# =============================================================================
# Mini-vLLM CMakeLists.txt
# =============================================================================
# Simplified CMake for mini-vLLM focusing on H100 (SM90) with Flash Attention
# Build: mkdir build && cd build && cmake -G Ninja -DVLLM_PYTHON_EXECUTABLE=$(which python3) ..
#        cmake --build . && cmake --install .
# =============================================================================

project(vllm_extensions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA by default
set(VLLM_TARGET_DEVICE "cuda" CACHE STRING "Target device backend for vLLM")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target device: ${VLLM_TARGET_DEVICE}")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

# Suppress potential warnings about unused manually-specified variables
set(ignoreMe "${VLLM_PYTHON_PATH}")

# Prevent installation of dependencies (cutlass) by default
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

# Supported python versions
set(PYTHON_SUPPORTED_VERSIONS "3.10" "3.11" "3.12" "3.13")

# Supported torch version
set(TORCH_SUPPORTED_VERSION_CUDA "2.9.1")

# Find Python
if (VLLM_PYTHON_EXECUTABLE)
  find_python_from_executable(${VLLM_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set VLLM_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()

# Update cmake's CMAKE_PREFIX_PATH with torch location
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

# Ensure nvcc is in PATH
find_program(NVCC_EXECUTABLE nvcc)
if (CUDA_FOUND AND NOT NVCC_EXECUTABLE)
    message(FATAL_ERROR "nvcc not found")
endif()

# Import torch cmake configuration (this also imports CUDA)
find_package(Torch REQUIRED)

# Supported NVIDIA architectures for CUDA 12.9
set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")

# Only support CUDA for mini-vLLM
if (NOT VLLM_TARGET_DEVICE STREQUAL "cuda")
    message(FATAL_ERROR "Mini-vLLM only supports CUDA target device")
endif()

# Set up CUDA
set(VLLM_GPU_LANG "CUDA")

if (NOT Torch_VERSION VERSION_EQUAL ${TORCH_SUPPORTED_VERSION_CUDA})
  message(WARNING "Pytorch version ${TORCH_SUPPORTED_VERSION_CUDA} "
    "expected for CUDA build, saw ${Torch_VERSION} instead.")
endif()

# Extract CUDA architectures
clear_cuda_arches(CUDA_ARCH_FLAGS)
extract_unique_cuda_archs_ascending(CUDA_ARCHS "${CUDA_ARCH_FLAGS}")
message(STATUS "CUDA target architectures: ${CUDA_ARCHS}")

# Filter by supported archs
cuda_archs_loose_intersection(CUDA_ARCHS "${CUDA_SUPPORTED_ARCHS}" "${CUDA_ARCHS}")
message(STATUS "CUDA supported target architectures: ${CUDA_ARCHS}")

# Get torch GPU compiler flags
get_torch_gpu_compiler_flags(VLLM_GPU_FLAGS ${VLLM_GPU_LANG})

# Set nvcc parallelism
if(NVCC_THREADS AND VLLM_GPU_LANG STREQUAL "CUDA")
  list(APPEND VLLM_GPU_FLAGS "--threads=${NVCC_THREADS}")
endif()

# Set CUDA include flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include")
if(CUDA_VERSION VERSION_GREATER_EQUAL 13.0)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
endif()

# FetchContent for dependencies
include(FetchContent)
file(MAKE_DIRECTORY ${FETCHCONTENT_BASE_DIR})
message(STATUS "FetchContent base directory: ${FETCHCONTENT_BASE_DIR}")

# =============================================================================
# CUTLASS
# =============================================================================
SET(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
set(CUTLASS_REVISION "v4.2.1")

if (DEFINED ENV{VLLM_CUTLASS_SRC_DIR})
  set(VLLM_CUTLASS_SRC_DIR $ENV{VLLM_CUTLASS_SRC_DIR})
endif()

if(VLLM_CUTLASS_SRC_DIR)
  if(NOT IS_ABSOLUTE VLLM_CUTLASS_SRC_DIR)
    get_filename_component(VLLM_CUTLASS_SRC_DIR "${VLLM_CUTLASS_SRC_DIR}" ABSOLUTE)
  endif()
  message(STATUS "Using CUTLASS from ${VLLM_CUTLASS_SRC_DIR}")
  FetchContent_Declare(cutlass SOURCE_DIR ${VLLM_CUTLASS_SRC_DIR})
else()
  FetchContent_Declare(
      cutlass
      GIT_REPOSITORY https://github.com/nvidia/cutlass.git
      GIT_TAG ${CUTLASS_REVISION}
      GIT_PROGRESS TRUE
      GIT_SHALLOW TRUE
  )
endif()
FetchContent_MakeAvailable(cutlass)

# =============================================================================
# cumem_allocator extension
# =============================================================================
set(VLLM_CUMEM_EXT_SRC "csrc/cumem_allocator.cpp")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_CUMEM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

message(STATUS "Enabling cumem allocator extension.")
list(APPEND CUMEM_LIBS CUDA::cuda_driver)

define_extension_target(
  cumem_allocator
  DESTINATION vllm
  LANGUAGE CXX
  SOURCES ${VLLM_CUMEM_EXT_SRC}
  LIBRARIES ${CUMEM_LIBS}
  USE_SABI 3.8
  WITH_SOABI)

# =============================================================================
# _C extension - Core CUDA kernels
# =============================================================================
# Only include source files that actually exist in mini-vllm
set(VLLM_EXT_SRC
  "csrc/cache_kernels.cu"
  "csrc/attention/paged_attention_v1.cu"
  "csrc/attention/paged_attention_v2.cu"
  "csrc/attention/merge_attn_states.cu"
  "csrc/attention/vertical_slash_index.cu"
  "csrc/pos_encoding_kernels.cu"
  "csrc/activation_kernels.cu"
  "csrc/layernorm_kernels.cu"
  "csrc/fused_qknorm_rope_kernel.cu"
  "csrc/layernorm_quant_kernels.cu"
  "csrc/sampler.cu"
  "csrc/cuda_view.cu"
  "csrc/quantization/w8a8/int8/scaled_quant.cu"
  "csrc/quantization/w8a8/fp8/common.cu"
  "csrc/quantization/fused_kernels/fused_layernorm_dynamic_per_token_quant.cu"
  "csrc/quantization/activation_kernels.cu"
  "csrc/cuda_utils_kernels.cu"
  "csrc/custom_all_reduce.cu"
  "csrc/torch_bindings.cpp")

# Add CUTLASS-based kernels
list(APPEND VLLM_EXT_SRC
  "csrc/permute_cols.cu"
  "csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu"
  "csrc/quantization/fp4/nvfp4_quant_entry.cu"
  "csrc/quantization/fp4/nvfp4_scaled_mm_entry.cu"
  "csrc/sparse/cutlass/sparse_scaled_mm_entry.cu"
  "csrc/cutlass_extensions/common.cpp"
  "csrc/quantization/w8a8/fp8/per_token_group_quant.cu"
  "csrc/quantization/w8a8/int8/per_token_group_quant.cu")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

# SM90 (Hopper/H100) specific kernels
cuda_archs_loose_intersection(SCALED_MM_ARCHS "9.0a;" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.0 AND SCALED_MM_ARCHS)
  set(SRCS
     "csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm90.cu"
     "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8.cu"
     "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_int8.cu"
     "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_azp_sm90_int8.cu"
     "csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SCALED_MM_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SCALED_MM_SM90=1")
  message(STATUS "Building scaled_mm_c3x_sm90 for archs: ${SCALED_MM_ARCHS}")
endif()

# 2:4 Sparse kernels for SM90
cuda_archs_loose_intersection(SPARSE_ARCHS "9.0a;" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.2 AND SPARSE_ARCHS)
  set(SRCS "csrc/sparse/cutlass/sparse_scaled_mm_c3x.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${SPARSE_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_SPARSE_SCALED_MM_C3X=1")
  message(STATUS "Building sparse_scaled_mm_c3x for archs: ${SPARSE_ARCHS}")
endif()

# CUTLASS MoE kernels for SM90
cuda_archs_loose_intersection(MOE_ARCHS "9.0a" "${CUDA_ARCHS}")
if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.3 AND MOE_ARCHS)
  set(SRCS
    "csrc/quantization/w8a8/cutlass/moe/grouped_mm_c3x_sm90.cu"
    "csrc/quantization/w8a8/cutlass/moe/moe_data.cu")
  set_gencode_flags_for_srcs(
    SRCS "${SRCS}"
    CUDA_ARCHS "${MOE_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_CUTLASS_MOE_SM90=1")
  message(STATUS "Building CUTLASS MoE for archs: ${MOE_ARCHS}")
endif()

# =============================================================================
# GPTQ Marlin kernels (needed for MXFP4)
# =============================================================================
# Marlin arches - need SM80+ for H100 compatibility
cuda_archs_loose_intersection(MARLIN_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_OTHER_ARCHS "8.0+PTX" "${CUDA_ARCHS}")

if (MARLIN_OTHER_ARCHS)
  message(STATUS "Building Marlin kernels for archs: ${MARLIN_OTHER_ARCHS}")

  # Generate Marlin template kernels
  set(MARLIN_GEN_SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/quantization/gptq_marlin/generate_kernels.py)

  if(EXISTS ${MARLIN_GEN_SCRIPT})
    list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
    message(STATUS "Running Marlin kernel generation script for archs: ${CUDA_ARCHS_STR}")
    execute_process(
      COMMAND ${Python_EXECUTABLE} ${MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE marlin_generation_result
      OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
      ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/marlin_generation.log
    )
    if (NOT marlin_generation_result EQUAL 0)
      message(WARNING "Marlin kernel generation returned: ${marlin_generation_result}")
    endif()
  endif()

  # Collect generated kernel files
  file(GLOB MARLIN_TEMPLATE_KERNEL_SRC "csrc/quantization/gptq_marlin/sm80_kernel_*.cu")
  if(MARLIN_TEMPLATE_KERNEL_SRC)
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_TEMPLATE_KERNEL_SRC}"
      CUDA_ARCHS "${MARLIN_ARCHS}")
    list(APPEND VLLM_EXT_SRC "${MARLIN_TEMPLATE_KERNEL_SRC}")
    message(STATUS "Adding ${CMAKE_CURRENT_LIST_LENGTH} Marlin template kernels")
  endif()

  # Core Marlin sources
  set(MARLIN_SRCS
    "csrc/quantization/gptq_marlin/gptq_marlin.cu"
    "csrc/quantization/gptq_marlin/marlin_int4_fp8_preprocess.cu"
    "csrc/quantization/gptq_marlin/gptq_marlin_repack.cu"
    "csrc/quantization/gptq_marlin/awq_marlin_repack.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_SRCS}"
    CUDA_ARCHS "${MARLIN_OTHER_ARCHS}")
  list(APPEND VLLM_EXT_SRC "${MARLIN_SRCS}")
  list(APPEND VLLM_GPU_FLAGS "-DENABLE_GPTQ_MARLIN=1")
  message(STATUS "Adding GPTQ Marlin core sources")
endif()

message(STATUS "Enabling _C extension with ${CMAKE_CUDA_COMPILER_VERSION}")
define_extension_target(
  _C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

target_compile_definitions(_C PRIVATE CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1)

# =============================================================================
# _moe_C extension
# =============================================================================
set(VLLM_MOE_EXT_SRC
  "csrc/moe/torch_bindings.cpp"
  "csrc/moe/moe_align_sum_kernels.cu"
  "csrc/moe/topk_softmax_kernels.cu"
  "csrc/moe/moe_wna16.cu"
  "csrc/moe/grouped_topk_kernels.cu"
  "csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu"
  "csrc/moe/moe_permute_unpermute_op.cu")

set_gencode_flags_for_srcs(
  SRCS "${VLLM_MOE_EXT_SRC}"
  CUDA_ARCHS "${CUDA_ARCHS}")

# =============================================================================
# Marlin MOE kernels (needed for MXFP4 MoE models)
# =============================================================================
cuda_archs_loose_intersection(MARLIN_MOE_ARCHS "8.0+PTX" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_MOE_SM75_ARCHS "7.5" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_MOE_FP8_ARCHS "8.9;12.0" "${CUDA_ARCHS}")
cuda_archs_loose_intersection(MARLIN_MOE_OTHER_ARCHS "7.5;8.0+PTX" "${CUDA_ARCHS}")

if (MARLIN_MOE_OTHER_ARCHS)
  # Generate Marlin MOE kernels
  set(MOE_MARLIN_GEN_SCRIPT
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/moe/marlin_moe_wna16/generate_kernels.py)

  if(EXISTS ${MOE_MARLIN_GEN_SCRIPT})
    list(JOIN CUDA_ARCHS "," CUDA_ARCHS_STR)
    message(STATUS "Running Marlin MOE kernel generation script for archs: ${CUDA_ARCHS_STR}")
    execute_process(
      COMMAND ${Python_EXECUTABLE} ${MOE_MARLIN_GEN_SCRIPT} ${CUDA_ARCHS_STR}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE moe_marlin_generation_result
      OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
      ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/moe_marlin_generation.log
    )
    if (NOT moe_marlin_generation_result EQUAL 0)
      message(WARNING "Marlin MOE kernel generation returned: ${moe_marlin_generation_result}")
    else()
      message(STATUS "Marlin MOE generation completed successfully.")
    endif()
  endif()

  # Collect generated SM80 kernels
  if (MARLIN_MOE_ARCHS)
    file(GLOB MARLIN_MOE_SRC "csrc/moe/marlin_moe_wna16/sm80_kernel_*.cu")
    if(MARLIN_MOE_SRC)
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_ARCHS}")
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SRC})
      message(STATUS "Adding Marlin MOE SM80 kernels")
    endif()
  endif()

  # Collect generated SM75 kernels
  if (MARLIN_MOE_SM75_ARCHS)
    file(GLOB MARLIN_MOE_SM75_SRC "csrc/moe/marlin_moe_wna16/sm75_kernel_*.cu")
    if(MARLIN_MOE_SM75_SRC)
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_SM75_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_SM75_ARCHS}")
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_SM75_SRC})
      message(STATUS "Adding Marlin MOE SM75 kernels")
    endif()
  endif()

  # Collect generated FP8 kernels (SM89)
  if (MARLIN_MOE_FP8_ARCHS)
    file(GLOB MARLIN_MOE_FP8_SRC "csrc/moe/marlin_moe_wna16/sm89_kernel_*.cu")
    if(MARLIN_MOE_FP8_SRC)
      set_gencode_flags_for_srcs(
        SRCS "${MARLIN_MOE_FP8_SRC}"
        CUDA_ARCHS "${MARLIN_MOE_FP8_ARCHS}")
      list(APPEND VLLM_MOE_EXT_SRC ${MARLIN_MOE_FP8_SRC})
      message(STATUS "Adding Marlin MOE FP8 kernels")
    endif()
  endif()

  # Main ops.cu file
  set(MARLIN_MOE_OTHER_SRC "csrc/moe/marlin_moe_wna16/ops.cu")
  set_gencode_flags_for_srcs(
    SRCS "${MARLIN_MOE_OTHER_SRC}"
    CUDA_ARCHS "${MARLIN_MOE_OTHER_ARCHS}")
  list(APPEND VLLM_MOE_EXT_SRC "${MARLIN_MOE_OTHER_SRC}")

  list(APPEND VLLM_GPU_FLAGS "-DENABLE_MARLIN_MOE=1")
  message(STATUS "Building Marlin MOE kernels for archs: ${MARLIN_MOE_OTHER_ARCHS}")
else()
  message(STATUS "Not building Marlin MOE kernels - no compatible archs")
endif()

message(STATUS "Enabling _moe_C extension.")
define_extension_target(
  _moe_C
  DESTINATION vllm
  LANGUAGE ${VLLM_GPU_LANG}
  SOURCES ${VLLM_MOE_EXT_SRC}
  COMPILE_FLAGS ${VLLM_GPU_FLAGS}
  ARCHITECTURES ${VLLM_GPU_ARCHES}
  INCLUDE_DIRECTORIES ${CUTLASS_INCLUDE_DIR}
  INCLUDE_DIRECTORIES ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR}
  USE_SABI 3
  WITH_SOABI)

# =============================================================================
# External Projects: triton_kernels and Flash Attention
# =============================================================================
include(cmake/external_projects/triton_kernels.cmake)

# vllm-flash-attn should be last as it overwrites some CMake functions
include(cmake/external_projects/vllm_flash_attn.cmake)
