# plugin/CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(StemExtractor VERSION 0.0.1)

add_subdirectory(JUCE)

juce_add_plugin(StemExtractor
    COMPANY_NAME "DanK"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACTURER_CODE Dank
    PLUGIN_CODE Sstm
    FORMATS VST3 AU Standalone
    PRODUCT_NAME "Stem Extractor"
)

juce_generate_juce_header(StemExtractor)

target_sources(StemExtractor PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
)

target_link_libraries(StemExtractor PRIVATE
    juce::juce_audio_utils
    juce::juce_dsp
)

target_compile_definitions(StemExtractor PUBLIC
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_include_directories(StemExtractor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
)

target_link_libraries(StemExtractor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.dylib
)

# --- Testing ---
enable_testing()

# Create a test executable
add_executable(RunCPPTests ../tests/CPP/test_fifo.cpp)

# Link it to JUCE (and whatever else your FIFO relies on)
target_link_libraries(RunCPPTests PRIVATE 
    juce::juce_audio_utils 
    juce::juce_dsp
)

# Register it with CTest
add_test(NAME FIFOTest COMMAND RunCPPTests)

