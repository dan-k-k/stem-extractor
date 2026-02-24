// plugin/Source/PluginEditor.h
#pragma once
#include <JuceHeader.h>
#include "PluginProcessor.h"

class StemExtractorEditor : public juce::AudioProcessorEditor
{
public:
    StemExtractorEditor (StemExtractorProcessor&);
    ~StemExtractorEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    StemExtractorProcessor& audioProcessor;

    juce::Slider gainSlider;
    juce::Label gainLabel;
    std::unique_ptr<juce::SliderParameterAttachment> gainAttachment;

    juce::ComboBox stemDropdown;
    juce::Label stemLabel;
    std::unique_ptr<juce::ComboBoxParameterAttachment> stemAttachment;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (StemExtractorEditor)
};

