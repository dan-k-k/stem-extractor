# plugin/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(SmartStemExtractor VERSION 0.0.1)

add_subdirectory(JUCE)

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

juce_generate_juce_header(SmartStemExtractor)

target_sources(SmartStemExtractor PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_link_libraries(SmartStemExtractor PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
)

target_compile_definitions(SmartStemExtractor PUBLIC
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_include_directories(SmartStemExtractor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
)

target_link_libraries(SmartStemExtractor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.dylib
)

