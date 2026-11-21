cmake_minimum_required(VERSION 3.16)
project(nbody VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
endif()

option(NBODY_BUILD_TESTS "Compila os testes (CTest)" ON)

# ---------------------------------------------------------------------------
# Nucleo: fisica, integrador, condicoes iniciais e IO
# ---------------------------------------------------------------------------
add_library(nbody_core
  src/body_system.cpp
  src/solver.cpp
  src/brute_force.cpp
  src/barnes_hut.cpp
  src/simulation.cpp
  src/diagnostics.cpp
  src/initial_conditions.cpp
  src/io.cpp
)
target_include_directories(nbody_core PUBLIC include)
target_compile_features(nbody_core PUBLIC cxx_std_17)
target_compile_options(nbody_core PRIVATE -Wall -Wextra -Wpedantic)

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