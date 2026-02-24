// plugin/Source/PluginProcessor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"

SmartStemExtractorProcessor::SmartStemExtractorProcessor()
     : AudioProcessor (BusesProperties().withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                                        .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
       juce::Thread("ONNX_Inference_Thread"), 
       forwardFFT (fftOrder), 
       inverseFFT (fftOrder),
       window (fftSize, juce::dsp::WindowingFunction<float>::hann)
{
    // --- ADD THESE TWO LINES ---
    gainParam = new juce::AudioParameterFloat ("gain", "Stem Gain", 0.0f, 2.0f, 1.0f);
    addParameter (gainParam);

    juce::StringArray stemChoices {"Full Mix", "Vocals", "Drums", "Bass", "Other"};
    stemParam = new juce::AudioParameterChoice ("stem", "Stem Selection", stemChoices, 0); // 0 defaults to Full Mix
    addParameter (stemParam);
    // ---------------------------

    // Update the path to look inside the new dedicated folder
    std::string modelPath = "/Users/Shared/SmartStemExtractor/smart_stem_extractor.onnx";

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1); 
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        onnxSession = std::make_unique<Ort::Session>(onnxEnv, modelPath.c_str(), sessionOptions);
        juce::Logger::writeToLog("✅ AI Model Loaded Successfully!");
    } catch (const Ort::Exception& e) {
        juce::Logger::writeToLog("❌ ONNX Load Error: " + juce::String(e.what()));
    }
}

SmartStemExtractorProcessor::~SmartStemExtractorProcessor() 
{
    stopThread(2000); // Safely kill the background thread when closing the plugin
}

void SmartStemExtractorProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // REPORT LATENCY TO ABLETON
    int aiLatency = aiTimeFrames * hopSize;
    int computeBuffer = (int)(sampleRate * 2); 
    setLatencySamples (aiLatency + computeBuffer);
    // --------------------------------------------------

    // Give the output FIFO a massive 10-second buffer so it never wraps prematurely
    int tenSecondBuffer = sampleRate * 10.0;
    inputFifo.setSize (getTotalNumInputChannels(), fftSize * 2);
    outputFifo.setSize (getTotalNumOutputChannels(), tenSecondBuffer);
    
    inputFifo.clear();
    outputFifo.clear();
    inputWriteIdx = 0;
    outputReadIdx = 0;
    hopCounter = 0;
    frameCounter = 0;

    int inputDataSize = 2 * 512 * aiTimeFrames;
    inputTensorData.assign(inputDataSize, 0.0f);
    inputTensorDataCopy.assign(inputDataSize, 0.0f);

    int outputDataSize = 4 * 2 * 512 * aiTimeFrames;
    outputTensorData.assign(outputDataSize, 0.0f);

    complexHistoryL.assign(aiTimeFrames * fftSize * 2, 0.0f);
    complexHistoryR.assign(aiTimeFrames * fftSize * 2, 0.0f);
    complexHistoryLCopy.assign(aiTimeFrames * fftSize * 2, 0.0f);
    complexHistoryRCopy.assign(aiTimeFrames * fftSize * 2, 0.0f);

    fftWorkspaceL.assign(fftSize * 2, 0.0f);
    fftWorkspaceR.assign(fftSize * 2, 0.0f);
}

void SmartStemExtractorProcessor::releaseResources() {}

void SmartStemExtractorProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals; 
    auto numInputChannels = getTotalNumInputChannels();
    auto numSamples = buffer.getNumSamples();

    for (auto i = numInputChannels; i < getTotalNumOutputChannels(); ++i)
        buffer.clear (i, 0, numSamples);

    if (numInputChannels < 2) return;

    const float* inL = buffer.getReadPointer(0);
    const float* inR = buffer.getReadPointer(1);
    float* outL = buffer.getWritePointer(0);
    float* outR = buffer.getWritePointer(1);

    for (int i = 0; i < numSamples; ++i)
    {
        // 1. Push new audio into the Input FIFO
        inputFifo.setSample(0, inputWriteIdx, inL[i]);
        inputFifo.setSample(1, inputWriteIdx, inR[i]);

        // 2. Pull audio from the Output FIFO (this will be silence until the AI finishes)
        float aiL = outputFifo.getSample(0, outputReadIdx);
        float aiR = outputFifo.getSample(1, outputReadIdx);

        // AI output
        outL[i] = aiL;
        outR[i] = aiR;

        outputFifo.setSample(0, outputReadIdx, 0.0f);
        outputFifo.setSample(1, outputReadIdx, 0.0f);

        // 4. Advance our separated pointers safely
        inputWriteIdx = (inputWriteIdx + 1) % inputFifo.getNumSamples();
        outputReadIdx = (outputReadIdx + 1) % outputFifo.getNumSamples();

        hopCounter++;
        if (hopCounter >= hopSize) 
        {
            hopCounter = 0; 
            processFFTFrame(); 
        }
    }
}

void SmartStemExtractorProcessor::processFFTFrame()
{
    for (int i = 0; i < fftSize; ++i) {
        // Read backwards from our current write index
        int readIndex = (inputWriteIdx - fftSize + i + inputFifo.getNumSamples()) % inputFifo.getNumSamples();
        fftWorkspaceL[i] = inputFifo.getSample(0, readIndex);
        fftWorkspaceR[i] = inputFifo.getSample(1, readIndex);
    }
    window.multiplyWithWindowingTable (fftWorkspaceL.data(), fftSize);
    window.multiplyWithWindowingTable (fftWorkspaceR.data(), fftSize);

    forwardFFT.performRealOnlyForwardTransform (fftWorkspaceL.data());
    forwardFFT.performRealOnlyForwardTransform (fftWorkspaceR.data());

    int historyOffset = frameCounter * fftSize * 2;
    for (int bin = 0; bin < 512; ++bin) 
    {
        complexHistoryL[historyOffset + (bin * 2)]     = fftWorkspaceL[bin * 2];
        complexHistoryL[historyOffset + (bin * 2) + 1] = fftWorkspaceL[bin * 2 + 1];
        complexHistoryR[historyOffset + (bin * 2)]     = fftWorkspaceR[bin * 2];
        complexHistoryR[historyOffset + (bin * 2) + 1] = fftWorkspaceR[bin * 2 + 1];

        float magL = std::sqrt(std::pow(fftWorkspaceL[bin*2], 2) + std::pow(fftWorkspaceL[bin*2+1], 2));
        float magR = std::sqrt(std::pow(fftWorkspaceR[bin*2], 2) + std::pow(fftWorkspaceR[bin*2+1], 2));
        
        inputTensorData[(0 * 512 * aiTimeFrames) + (bin * aiTimeFrames) + frameCounter] = magL;
        inputTensorData[(1 * 512 * aiTimeFrames) + (bin * aiTimeFrames) + frameCounter] = magR;
    }

    frameCounter++;

    // WHEN WE HAVE 512 FRAMES...
    if (frameCounter >= aiTimeFrames) 
    {
        frameCounter = 0; 

        if (! isThreadRunning()) {
            inputTensorDataCopy = inputTensorData;
            complexHistoryLCopy = complexHistoryL;
            complexHistoryRCopy = complexHistoryR;

            // FIX: Target exactly 0.5 seconds ahead, aligning perfectly with Ableton's latency
            int computeBuffer = (int)(getSampleRate() * 2.0); // <-- CHANGE 0.5 to 2.0 HERE
            threadWriteStartIdx = (outputReadIdx + computeBuffer) % outputFifo.getNumSamples();

            startThread();
        }
    }
}

// --- THE BACKGROUND THREAD ---
void SmartStemExtractorProcessor::run()
{
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> inputShape = {1, 2, 512, aiTimeFrames};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorDataCopy.data(), inputTensorDataCopy.size(), inputShape.data(), inputShape.size());
        
    std::vector<int64_t> outputShape = {1, 4, 2, 512, aiTimeFrames};
    Ort::Value outputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorData.data(), outputTensorData.size(), outputShape.data(), outputShape.size());

    const char* inputNames[] = {"input_spectrogram"};
    const char* outputNames[] = {"output_masks"};
    
    try {
        // --- ADD THESE LINES TO TIME THE AI ---
        auto startTime = juce::Time::getMillisecondCounterHiRes();

        onnxSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, &outputTensor, 1);
        
        auto endTime = juce::Time::getMillisecondCounterHiRes();
        juce::Logger::writeToLog("⏱ AI Inference Took: " + juce::String(endTime - startTime) + " ms");
        // --------------------------------------

        // Target writing the reconstructed audio 0.5 seconds ahead of Ableton's CURRENT playhead
        int writeStartIdx = threadWriteStartIdx;

        // Use thread-local workspaces so we don't accidentally touch the Audio thread's variables
        std::vector<float> threadWorkspaceL(fftSize * 2, 0.0f);
        std::vector<float> threadWorkspaceR(fftSize * 2, 0.0f);

        int selectedMode = stemParam->getIndex(); 

        for (int frame = 0; frame < aiTimeFrames; ++frame) {
            if (threadShouldExit()) return; 
            int offset = frame * fftSize * 2;

            // 1. Apply AI Masks to the Positive Frequencies (Bins 0 to 511)
            for (int bin = 0; bin < 512; ++bin) {
                float maskL = 0.0f;
                float maskR = 0.0f;

                if (selectedMode == 0) {
                    // FULL MIX: Loop through all 4 stems and add them together
                    for (int stem = 0; stem < 4; ++stem) {
                        int idxL = (stem * 2 * 512 * aiTimeFrames) + (0 * 512 * aiTimeFrames) + (bin * aiTimeFrames) + frame;
                        int idxR = (stem * 2 * 512 * aiTimeFrames) + (1 * 512 * aiTimeFrames) + (bin * aiTimeFrames) + frame;
                        maskL += outputTensorData[idxL];
                        maskR += outputTensorData[idxR];
                    }
                } else {
                    // ISOLATED STEM: Just grab the one the user selected
                    // We subtract 1 because "Vocals" is selectedMode 1, but stem index 0
                    int stem = selectedMode - 1; 
                    int idxL = (stem * 2 * 512 * aiTimeFrames) + (0 * 512 * aiTimeFrames) + (bin * aiTimeFrames) + frame;
                    int idxR = (stem * 2 * 512 * aiTimeFrames) + (1 * 512 * aiTimeFrames) + (bin * aiTimeFrames) + frame;
                    maskL = outputTensorData[idxL];
                    maskR = outputTensorData[idxR];
                }

                // Apply the calculated mask to the audio history
                threadWorkspaceL[bin * 2]     = complexHistoryLCopy[offset + (bin * 2)]     * maskL;
                threadWorkspaceL[bin * 2 + 1] = complexHistoryLCopy[offset + (bin * 2) + 1] * maskL;
                threadWorkspaceR[bin * 2]     = complexHistoryRCopy[offset + (bin * 2)]     * maskR;
                threadWorkspaceR[bin * 2 + 1] = complexHistoryRCopy[offset + (bin * 2) + 1] * maskR;
            }

            // 2. Zero out the center Nyquist bin (Bin 512)
            threadWorkspaceL[512 * 2] = 0.0f; threadWorkspaceL[512 * 2 + 1] = 0.0f;
            threadWorkspaceR[512 * 2] = 0.0f; threadWorkspaceR[512 * 2 + 1] = 0.0f;

            // Notice: The Step 3 Mirroring loop is completely gone! 

            // 4. Transform back to Audio
            inverseFFT.performRealOnlyInverseTransform (threadWorkspaceL.data());
            inverseFFT.performRealOnlyInverseTransform (threadWorkspaceR.data());

            // 5. Synthesis Window & Overlap Normalization
            window.multiplyWithWindowingTable (threadWorkspaceL.data(), fftSize);
            window.multiplyWithWindowingTable (threadWorkspaceR.data(), fftSize);
            
            // The JUCE Factor of 4 * WOLA Window sum of 1.5 = 6.0
            float overlapNorm = 1.0f / 6.0f; 
            
            // Grab the live parameter value from Ableton/UI
            float userGain = gainParam->get();

            for (int i = 0; i < fftSize; ++i) {
                int writeIndex = (writeStartIdx + (frame * hopSize) + i) % outputFifo.getNumSamples();
                
                // Multiply by both the unity math normalizer AND the user's gain knob
                outputFifo.addSample(0, writeIndex, threadWorkspaceL[i] * overlapNorm * userGain);
                outputFifo.addSample(1, writeIndex, threadWorkspaceR[i] * overlapNorm * userGain);
            }
        }
    } catch (const Ort::Exception& e) {
        juce::Logger::writeToLog("❌ Inference Error: " + juce::String(e.what()));
    }
}

juce::AudioProcessorEditor* SmartStemExtractorProcessor::createEditor() { return new SmartStemExtractorEditor (*this); }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new SmartStemExtractorProcessor(); }

