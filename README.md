# Emotion Detection from Voice Recordings

This project enables emotion classification from audio recordings by combining speech-to-text transcription with NLP-based sentiment analysis. It segments the recording, identifies emotional content in each section, and outputs an overall emotion profile.

---

## Features

- Upload or record voice clips
- Whisper-based speech-to-text
- DistilBERT-based emotion classification
- Segment-wise and overall emotion analysis
- Streamlit web app interface
- Hosted publicly on Hugging Face and Streamlit

---

##  Tech Stack

- **Speech-to-Text**: [OpenAI Whisper](https://github.com/openai/whisper)
- **NLP Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased)
- **Training Framework**: Hugging Face Transformers + Datasets
- **Deployment**: Streamlit Cloud
- **GPU Platform**: Google Colab (T4)

---

##  Model Details

- **Architecture**: DistilBERT (uncased)
- **Dataset**: Hugging Face native emotion dataset
  - 16,000 utterances labeled across 6 emotions
  - Balanced class distribution
  - Average sentence length: 10 tokens
- **Training**:
  - Fine-tuned for 3 epochs
  - Achieved **93% F1-score**
- **Model Hosted At**:  
  [Hugging Face – emotion-distilbert-final](https://huggingface.co/Akhil199797/emotion-distilbert-final)

---

## Web Application

Try out the working demo here:  
[Streamlit App](https://emotiondetection-wpkuapstxvqrkyb68ec4wk.streamlit.app/)

### App Flow

1. Upload or record a voice clip
2. Convert audio to text using Whisper
3. Segment the text to match model input limits
4. Predict emotion per segment using DistilBERT
5. Display:
   - Overall dominant emotion
   - Segment-wise breakdown
   - Transcribed text with audio playback

---

## Future Improvements

- **Dataset Expansion**: Integrate MELD and IEMOCAP datasets for better diversity
- **Audio-Based Modeling**: Fine-tune `Wav2Vec2` for tone-based emotional detection
- **Multimodal Fusion**: Combine text and audio features for enhanced prediction accuracy

---

## Repository Structure
emotion_detection/
│
├── app/ # Streamlit app files
├── data/ # Scripts for loading/preprocessing datasets
├── model/ # Training and inference code
├── utils/ # Helper functions
├── requirements.txt # Required Python packages
├── README.md # Project overview
└── inference_pipeline.py # End-to-end inference logic

