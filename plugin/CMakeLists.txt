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
    COPY_PLUGIN_AFTER_BUILD FALSE #
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
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.dylib
)

target_compile_definitions(StemExtractor PUBLIC
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_include_directories(StemExtractor PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include
)

set_target_properties(StemExtractor_VST3 PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path/../Frameworks"
)

add_custom_command(TARGET StemExtractor_VST3 POST_BUILD
    # 1. Create the Frameworks directory inside the built .vst3 bundle
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:StemExtractor_VST3>/../Frameworks"
    
    # 2. Copy the ONNX library AND force it to have the exact versioned name dyld expects
    COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib/libonnxruntime.dylib"
        "$<TARGET_FILE_DIR:StemExtractor_VST3>/../Frameworks/libonnxruntime.1.24.2.dylib"
        
    # 3. Strip any Apple quarantine flags from the bundled files
    COMMAND xattr -rc "$<TARGET_FILE_DIR:StemExtractor_VST3>/../.."
    
    # 4. Ad-hoc sign the completely assembled bundle in the build folder
    COMMAND codesign --force --deep --sign - "$<TARGET_FILE_DIR:StemExtractor_VST3>/../.."

    # 5. Remove any old versions in your local Mac plugin folder to avoid conflicts
    COMMAND ${CMAKE_COMMAND} -E remove_directory "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/Stem Extractor.vst3"

    # 6. Copy the perfected, signed bundle to your local Mac for Ableton testing
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "$<TARGET_FILE_DIR:StemExtractor_VST3>/../.."
        "$ENV{HOME}/Library/Audio/Plug-Ins/VST3/Stem Extractor.vst3"
)

enable_testing()

add_executable(RunCPPTests ../tests/CPP/test_fifo.cpp)

target_link_libraries(RunCPPTests PRIVATE 
    juce::juce_audio_utils 
    juce::juce_dsp
)

add_test(NAME FIFOTest COMMAND RunCPPTests)

