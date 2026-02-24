// plugin/Source/PluginProcessor.h
#pragma once
#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>

class StemExtractorProcessor : public juce::AudioProcessor, public juce::Thread
{
public:
    StemExtractorProcessor();
    ~StemExtractorProcessor() override;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    void run() override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "Stem Extractor"; }
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
    juce::AudioParameterFloat* gainParam;
    juce::AudioParameterChoice* stemParam;

    Ort::Env onnxEnv{ORT_LOGGING_LEVEL_WARNING, "StemExtractor"};
    std::unique_ptr<Ort::Session> onnxSession;

    static constexpr int fftOrder = 10; 
    static constexpr int fftSize = 1 << fftOrder; 
    static constexpr int hopSize = 256;           
    
    juce::dsp::FFT forwardFFT;
    juce::dsp::FFT inverseFFT;
    juce::dsp::WindowingFunction<float> window;

    juce::AudioBuffer<float> inputFifo;  
    juce::AudioBuffer<float> outputFifo; 
    int inputWriteIdx = 0;  // Tracks where we are writing incoming audio
    int outputReadIdx = 0;  // Tracks where Ableton is reading outgoing audio
    int hopCounter = 0; 

    // AI TENSOR MEMORY
    std::vector<float> inputTensorData;
    std::vector<float> outputTensorData;

    static constexpr int aiTimeFrames = 512;
    int frameCounter = 0; 
    
    std::vector<float> complexHistoryL;
    std::vector<float> complexHistoryR;
    
    // THREAD SAFE COPIES
    std::vector<float> inputTensorDataCopy;
    std::vector<float> complexHistoryLCopy;
    std::vector<float> complexHistoryRCopy;
    
    std::vector<float> fftWorkspaceL;
    std::vector<float> fftWorkspaceR;

    int threadWriteStartIdx = 0;

    void processFFTFrame();
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (StemExtractorProcessor)
};

