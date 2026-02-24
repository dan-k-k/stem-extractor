// plugin/Source/PluginProcessor.h
#pragma once
#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

// 1. INHERIT FROM juce::Thread
class SmartStemExtractorProcessor : public juce::AudioProcessor, public juce::Thread
{
public:
    SmartStemExtractorProcessor();
    ~SmartStemExtractorProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    // We must override the run() function for our background thread
    void run() override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Smart Stem Extractor"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override {}
    void setStateInformation (const void*, int) override {}

private:
    juce::AudioParameterFloat* gainParam; // <-- ADD THIS LINE
    juce::AudioParameterChoice* stemParam; // <-- ADD THIS LINE

    Ort::Env onnxEnv{ORT_LOGGING_LEVEL_WARNING, "SmartStemExtractor"};
    std::unique_ptr<Ort::Session> onnxSession;

    static constexpr int fftOrder = 10; 
    static constexpr int fftSize = 1 << fftOrder; 
    static constexpr int hopSize = 256;           
    
    juce::dsp::FFT forwardFFT;
    juce::dsp::FFT inverseFFT;
    juce::dsp::WindowingFunction<float> window;

    // --- SEPARATED CIRCULAR BUFFERS ---
    juce::AudioBuffer<float> inputFifo;  
    juce::AudioBuffer<float> outputFifo; 
    int inputWriteIdx = 0;  // Tracks where we are writing incoming audio
    int outputReadIdx = 0;  // Tracks where Ableton is reading outgoing audio
    int hopCounter = 0; 

    // --- AI TENSOR MEMORY ---
    std::vector<float> inputTensorData;
    std::vector<float> outputTensorData;

    static constexpr int aiTimeFrames = 512;
    int frameCounter = 0; 
    
    std::vector<float> complexHistoryL;
    std::vector<float> complexHistoryR;
    
    // --- THREAD SAFE COPIES ---
    // The background thread will use these so the audio thread can keep running
    std::vector<float> inputTensorDataCopy;
    std::vector<float> complexHistoryLCopy;
    std::vector<float> complexHistoryRCopy;
    
    std::vector<float> fftWorkspaceL;
    std::vector<float> fftWorkspaceR;

    int threadWriteStartIdx = 0;

    void processFFTFrame();
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (SmartStemExtractorProcessor)
};

