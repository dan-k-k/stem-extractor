// tests/CPP/test_fifo.cpp
#include <juce_core/juce_core.h>
#include <juce_audio_basics/juce_audio_basics.h>

class FifoTest : public juce::UnitTest
{
public:
    FifoTest() : juce::UnitTest ("FIFO Wraparound Test") {}

    void runTest() override
    {
        beginTest ("Circular Buffer Wraparound Logic");

        juce::AudioBuffer<float> testFifo (2, 512);
        int writeIdx = 510; // Near the end of the buffer

        for (int i = 0; i < 4; ++i) {
            testFifo.setSample(0, writeIdx, 1.0f);
            writeIdx = (writeIdx + 1) % testFifo.getNumSamples();
        }

        expectEquals (writeIdx, 2);
    }
};

static FifoTest fifoTest; // Static instance so JUCE registers the test

int main (int argc, char* argv[])
{
    juce::UnitTestRunner runner;
    runner.runAllTests();

    for (int i = 0; i < runner.getNumResults(); ++i) {
        if (runner.getResult(i)->failures > 0)
            return 1; 
    }
    
    return 0; 
}

