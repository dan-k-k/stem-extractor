# plugin/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(SmartStemExtractor VERSION 0.0.1)

# 1. Tell CMake where the JUCE framework is
add_subdirectory(JUCE)

# 2. Define the metadata for your audio plugin
juce_add_plugin(SmartStemExtractor
    COMPANY_NAME "DanK"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE Dank
    PLUGIN_CODE Sstm
    FORMATS VST3 AU Standalone
    PRODUCT_NAME "Smart Stem Extractor"
)

# 3. Generate the JUCE header so our C++ files can easily include the framework
juce_generate_juce_header(SmartStemExtractor)

# 4. Tell the compiler which C++ files to build
target_sources(SmartStemExtractor PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

# 5. Link the core JUCE audio and GUI libraries
target_link_libraries(SmartStemExtractor PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
)

# 6. FIX: Tell JUCE this VST3 does not replace a legacy VST2
target_compile_definitions(SmartStemExtractor PUBLIC
    JUCE_VST3_CAN_REPLACE_VST2=0
)

# 7. Tell the compiler where the ONNX Runtime headers are
target_include_directories(SmartStemExtractor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
)

# 8. Link the pre-compiled ONNX Runtime library for macOS Apple Silicon
target_link_libraries(SmartStemExtractor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.dylib
)

