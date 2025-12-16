cmake_minimum_required(VERSION 3.10)
project(PlanetLLM)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)
include_directories(deps/llama.cpp/include)
include_directories(deps/llama.cpp/ggml/include)
include_directories(/home/linuxbrew/.linuxbrew/Cellar/nlohmann-json/3.11.3/include)
include_directories(src)

set(LLAMA_SOURCES
    deps/llama.cpp/ggml.c
    deps/llama.cpp/llama.cpp
)
set(planet_llm_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/node)
set(planet_llm_llm_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/llm)
set(planet_llm_encrypt_p2p_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/encrypt_p2p)

set(TEST_MODEL_PREPARE_SOURCES
    tests/test_model_prepare_standalone.cpp
    src/llm/llama_wrapper.cpp  
)

set(SRC_FILES
    src/node/user_node.cpp
    src/node/model_node.cpp
    src/node/verification_node.cpp
    src/llm/llama_wrapper.cpp
)

add_library(planet_llm_library STATIC ${SRC_FILES})
target_include_directories(planet_llm_library PUBLIC ${planet_llm_PUBLIC_INCLUDE_DIR})
target_include_directories(planet_llm_library PUBLIC ${planet_llm_llm_INCLUDE_DIR})
target_include_directories(planet_llm_library PUBLIC ${planet_llm_encrypt_p2p_INCLUDE_DIR})
set_target_properties(planet_llm_library PROPERTIES EXCLUDE_FROM_ALL TRUE)

add_executable(test_model_prepare_standalone ${TEST_MODEL_PREPARE_SOURCES})

target_link_libraries(test_model_prepare_standalone pthread ssl crypto)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_model_prepare_standalone OpenMP::OpenMP_CXX)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libllama.so")
    target_link_libraries(test_model_prepare_standalone ${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libllama.so)
else()
    message(WARNING "Could not find pre-compiled libllama.so. Please build llama.cpp first.")
endif()


add_executable(test_sida tests/test_sida.cpp)
target_include_directories(test_sida PRIVATE ${planet_llm_encrypt_p2p_INCLUDE_DIR})
target_link_libraries(test_sida pthread ssl crypto)

add_custom_target(run_sida_tests
    COMMAND test_sida
    DEPENDS test_sida
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)


find_package(CURL REQUIRED)

add_executable(verification_node
    src/node/verification_node_main.cpp
    src/node/verification_node.cpp
)
target_include_directories(verification_node PRIVATE 
    ${planet_llm_PUBLIC_INCLUDE_DIR}
    ${planet_llm_encrypt_p2p_INCLUDE_DIR}
    ${planet_llm_llm_INCLUDE_DIR}
)
# verification_node uses llama-server HTTP API, no need for llama.cpp
target_link_libraries(verification_node 
    pthread 
    ssl 
    crypto 
    ${CURL_LIBRARIES}
)
target_include_directories(verification_node PRIVATE ${CURL_INCLUDE_DIRS})

# Model Node - serves LLM inference with S-IDA support
# Requires llama.cpp library
if(EXISTS "${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libllama.so")
    add_executable(model_node
        src/node/model_node.cpp
        src/llm/llama_wrapper.cpp
    )
    target_include_directories(model_node PRIVATE 
        ${planet_llm_PUBLIC_INCLUDE_DIR}
        ${planet_llm_llm_INCLUDE_DIR}
        ${planet_llm_encrypt_p2p_INCLUDE_DIR}
    )
    target_link_libraries(model_node 
        pthread 
        ssl 
        crypto 
        ${CURL_LIBRARIES}
        ${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libllama.so
    )
    if(OpenMP_CXX_FOUND)
        target_link_libraries(model_node OpenMP::OpenMP_CXX)
    endif()
else()
    message(STATUS "Model node executable not built - llama.cpp library not found")
endif()


add_executable(user_node
    src/node/user_node.cpp
)
target_include_directories(user_node PRIVATE 
    ${planet_llm_PUBLIC_INCLUDE_DIR}
    ${planet_llm_encrypt_p2p_INCLUDE_DIR}
)
target_link_libraries(user_node 
    pthread 
    ssl 
    crypto 
    ${CURL_LIBRARIES}
)

find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ libzmq)
    if(ZMQ_FOUND)
        target_link_libraries(user_node ${ZMQ_LIBRARIES})
        target_include_directories(user_node PRIVATE ${ZMQ_INCLUDE_DIRS})
    endif()
endif()


set_target_properties(user_node PROPERTIES EXCLUDE_FROM_ALL TRUE)

#=============================================================================
# Demo Executables - Local Multi-Process Testing
#=============================================================================

set(DEMO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/demo)

# Find ZeroMQ for demo
find_package(PkgConfig)
if(PkgConfig_FOUND)
    pkg_check_modules(ZMQ libzmq)
endif()

# Relay Node - Simple message forwarder
add_executable(relay_node
    demo/relay_node.cpp
)
target_include_directories(relay_node PRIVATE 
    ${DEMO_INCLUDE_DIR}
    ${planet_llm_encrypt_p2p_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(relay_node pthread ssl crypto)
if(ZMQ_FOUND)
    target_link_libraries(relay_node ${ZMQ_LIBRARIES})
    target_include_directories(relay_node PRIVATE ${ZMQ_INCLUDE_DIRS})
endif()

# Demo User Node - Sends prompts via S-IDA relay network
add_executable(demo_user_node
    demo/demo_user_node.cpp
)
target_include_directories(demo_user_node PRIVATE 
    ${DEMO_INCLUDE_DIR}
    ${planet_llm_encrypt_p2p_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(demo_user_node pthread ssl crypto)
if(ZMQ_FOUND)
    target_link_libraries(demo_user_node ${ZMQ_LIBRARIES})
    target_include_directories(demo_user_node PRIVATE ${ZMQ_INCLUDE_DIRS})
endif()

# Demo Verifier Node - Sends challenges via S-IDA relay network
# Now includes LLM for perplexity-based verification
if(EXISTS "${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libllama.so")
    add_executable(demo_verifier_node
        demo/demo_verifier_node.cpp
        src/llm/llama_wrapper.cpp
    )
    target_include_directories(demo_verifier_node PRIVATE 
        ${DEMO_INCLUDE_DIR}
        ${planet_llm_encrypt_p2p_INCLUDE_DIR}
        ${planet_llm_llm_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(demo_verifier_node 
        pthread 
        ssl 
        crypto 
        ${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libllama.so
    )
    if(ZMQ_FOUND)
        target_link_libraries(demo_verifier_node ${ZMQ_LIBRARIES})
        target_include_directories(demo_verifier_node PRIVATE ${ZMQ_INCLUDE_DIRS})
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(demo_verifier_node OpenMP::OpenMP_CXX)
    endif()
else()
    # Build without LLM (heuristic verification only)
    add_executable(demo_verifier_node
        demo/demo_verifier_node.cpp
    )
    target_include_directories(demo_verifier_node PRIVATE 
        ${DEMO_INCLUDE_DIR}
        ${planet_llm_encrypt_p2p_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(demo_verifier_node pthread ssl crypto)
    if(ZMQ_FOUND)
        target_link_libraries(demo_verifier_node ${ZMQ_LIBRARIES})
        target_include_directories(demo_verifier_node PRIVATE ${ZMQ_INCLUDE_DIRS})
    endif()
    message(STATUS "demo_verifier_node built without LLM support - llama.cpp not found")
endif()

# Demo Model Node - Receives via S-IDA, runs LLM inference
if(EXISTS "${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libllama.so")
    add_executable(demo_model_node
        demo/demo_model_node.cpp
        src/llm/llama_wrapper.cpp
    )
    target_include_directories(demo_model_node PRIVATE 
        ${DEMO_INCLUDE_DIR}
        ${planet_llm_encrypt_p2p_INCLUDE_DIR}
        ${planet_llm_llm_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(demo_model_node 
        pthread 
        ssl 
        crypto 
        ${CMAKE_SOURCE_DIR}/deps/llama.cpp/build/bin/libllama.so
    )
    if(ZMQ_FOUND)
        target_link_libraries(demo_model_node ${ZMQ_LIBRARIES})
        target_include_directories(demo_model_node PRIVATE ${ZMQ_INCLUDE_DIRS})
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(demo_model_node OpenMP::OpenMP_CXX)
    endif()
else()
    # Build demo_model_node without LLM for testing network only
    add_executable(demo_model_node
        demo/demo_model_node.cpp
        src/llm/llama_wrapper.cpp
    )
    target_include_directories(demo_model_node PRIVATE 
        ${DEMO_INCLUDE_DIR}
        ${planet_llm_encrypt_p2p_INCLUDE_DIR}
        ${planet_llm_llm_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(demo_model_node pthread ssl crypto)
    if(ZMQ_FOUND)
        target_link_libraries(demo_model_node ${ZMQ_LIBRARIES})
        target_include_directories(demo_model_node PRIVATE ${ZMQ_INCLUDE_DIRS})
    endif()
    message(STATUS "demo_model_node built without LLM support - llama.cpp not found")
endif()

# Aggregate target to build all demo executables
add_custom_target(demo_all
    DEPENDS relay_node demo_user_node demo_verifier_node demo_model_node
    COMMENT "Building all demo executables..."
)
