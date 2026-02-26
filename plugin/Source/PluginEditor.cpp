// plugin/Source/PluginEditor.cpp
#include "PluginProcessor.h"
#include "PluginEditor.h"

StemExtractorEditor::StemExtractorEditor (StemExtractorProcessor& p)
    : AudioProcessorEditor (&p), audioProcessor (p)
{
    // Gain Knob
    gainSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    gainSlider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 50, 20);
    addAndMakeVisible(gainSlider);

    gainLabel.setText("Stem Gain", juce::dontSendNotification);
    gainLabel.setJustificationType(juce::Justification::centred);
    gainLabel.attachToComponent(&gainSlider, false); 
    addAndMakeVisible(gainLabel);

    // Stem Dropdown Menu
    stemDropdown.addItemList({"Full Mix", "Vocals", "Drums", "Bass", "Other"}, 1);
    addAndMakeVisible(stemDropdown);

    stemLabel.setText("Extraction Target", juce::dontSendNotification);
    stemLabel.setJustificationType(juce::Justification::centred);
    stemLabel.attachToComponent(&stemDropdown, false);
    addAndMakeVisible(stemLabel);

    // Attach UI to Backend Parameters
    const auto& parameters = audioProcessor.getParameters(); 
    
    // Gain is at index 0
    if (auto* floatParam = dynamic_cast<juce::RangedAudioParameter*>(parameters[0]))
        gainAttachment = std::make_unique<juce::SliderParameterAttachment>(*floatParam, gainSlider, nullptr);
        
    // Stem Choice is at index 1
    if (auto* choiceParam = dynamic_cast<juce::RangedAudioParameter*>(parameters[1]))
        stemAttachment = std::make_unique<juce::ComboBoxParameterAttachment>(*choiceParam, stemDropdown, nullptr);

    setSize (400, 300);
}

StemExtractorEditor::~StemExtractorEditor() {}

void StemExtractorEditor::paint (juce::Graphics& g)
{
    // Dark grey background
    g.fillAll (juce::Colour::fromRGB (30, 30, 30));
}

void StemExtractorEditor::resized()
{
    int uiWidth = getWidth();
    int uiHeight = getHeight();

    // x, y, width, height
    stemDropdown.setBounds (50, uiHeight / 2 - 12, 150, 24);
    gainSlider.setBounds (250, uiHeight / 2 - 50, 100, 100);
}

