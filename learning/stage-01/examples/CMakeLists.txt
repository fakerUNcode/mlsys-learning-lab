# Stage 1 示例的统一构建入口。每个子目录是一个边界清楚的主题工程。
cmake_minimum_required(VERSION 3.16)
project(stage1_learning_examples LANGUAGES CXX)

option(ENABLE_SANITIZERS "Enable address and undefined behavior sanitizers" OFF)

enable_testing()
add_subdirectory(00-runtime-foundations)
add_subdirectory(02-smart-pointer-ownership)
add_subdirectory(03-stl-containers-and-algorithms)
