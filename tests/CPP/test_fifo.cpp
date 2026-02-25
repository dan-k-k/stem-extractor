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
        int writeIdx = 510; // Start right near the end of the buffer

        // Write 4 samples (this will force it to wrap around to index 0 and 1)
        for (int i = 0; i < 4; ++i) {
            testFifo.setSample(0, writeIdx, 1.0f);
            writeIdx = (writeIdx + 1) % testFifo.getNumSamples();
        }

        // Assert that the math worked and it wrapped correctly
        expectEquals (writeIdx, 2);
    }
};

// Create a static instance so JUCE registers the test
static FifoTest fifoTest;

// The entry point for our test executable
int main (int argc, char* argv[])
{
    juce::UnitTestRunner runner;
    runner.runAllTests();

    // Check if any tests failed. If they did, return a non-zero exit code
    // so that CTest and GitHub Actions register the pipeline step as failed.
    for (int i = 0; i < runner.getNumResults(); ++i) {
        if (runner.getResult(i)->failures > 0)
            return 1; 
    }
    
    return 0; // Success!
}

