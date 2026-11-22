cmake_minimum_required(VERSION 3.18)
project(nbody VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
endif()

option(NBODY_BUILD_TESTS "Compila os testes (CTest)" ON)
option(NBODY_ENABLE_CUDA "Habilita o backend Barnes-Hut em CUDA" ON)
option(NBODY_USE_FLOAT   "Usa float (FP32) em vez de double (FP64)" ON)

# ---------------------------------------------------------------------------
# Nucleo: fisica, integrador, condicoes iniciais, IO e backend CUDA
# ---------------------------------------------------------------------------
set(NBODY_CORE_SOURCES
  src/body_system.cpp
  src/solver.cpp
  src/brute_force.cpp
  src/barnes_hut.cpp
  src/simulation.cpp
  src/diagnostics.cpp
  src/initial_conditions.cpp
  src/io.cpp
)

if(NBODY_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "Arquitetura CUDA (75 = RTX 2060 SUPER)" FORCE)
  endif()
  list(APPEND NBODY_CORE_SOURCES src/cuda/barnes_hut_cuda.cu)
else()
  list(APPEND NBODY_CORE_SOURCES src/cuda/cuda_stub.cpp)
endif()

add_library(nbody_core ${NBODY_CORE_SOURCES})
target_include_directories(nbody_core PUBLIC include)
target_compile_features(nbody_core PUBLIC cxx_std_17)

target_compile_options(nbody_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)

if(NBODY_USE_FLOAT)
  target_compile_definitions(nbody_core PUBLIC NBODY_USE_FLOAT)
endif()

if(NBODY_ENABLE_CUDA)
  target_compile_definitions(nbody_core PUBLIC NBODY_ENABLE_CUDA)
  set_target_properties(nbody_core PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endif()

# ---------------------------------------------------------------------------
# Executavel
# ---------------------------------------------------------------------------
add_executable(nbody src/main.cpp)
target_link_libraries(nbody PRIVATE nbody_core)

# ---------------------------------------------------------------------------
# Testes (CTest, sem dependencias externas)
# ---------------------------------------------------------------------------
if(NBODY_BUILD_TESTS)
  enable_testing()
  add_executable(nbody_tests tests/test_all.cpp)
  target_link_libraries(nbody_tests PRIVATE nbody_core)
  add_test(NAME nbody_tests COMMAND nbody_tests)
endif()