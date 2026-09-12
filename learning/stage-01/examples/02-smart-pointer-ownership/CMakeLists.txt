# 智能指针专题的构建目标和测试。
add_executable(stage1_smart_pointer_demo src/main.cpp)
target_include_directories(stage1_smart_pointer_demo PRIVATE include)
target_compile_features(stage1_smart_pointer_demo PRIVATE cxx_std_17)

add_executable(stage1_smart_pointer_test tests/smart_pointer_test.cpp)
target_include_directories(stage1_smart_pointer_test PRIVATE include)
target_compile_features(stage1_smart_pointer_test PRIVATE cxx_std_17)

if(ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target stage1_smart_pointer_demo stage1_smart_pointer_test)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()

add_test(NAME stage1_smart_pointer_test COMMAND stage1_smart_pointer_test)
