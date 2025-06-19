import streamlit as st
import numpy as np
import soundfile as sf
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# Load models (cache to avoid reloading on every rerun)
@st.cache_resource
def load_models():
    whisper_model = whisper.load_model("base")
    model_path = "Akhil199797/emotion-distilbert-final"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    emotion_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emotion_model.to(device)
    return whisper_model, tokenizer, emotion_model, device


LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'neutral']

# Audio chunking and inference
@st.cache_data(show_spinner=False)
def process_audio_in_chunks(audio_path, _whisper_model, _tokenizer, _emotion_model, _device, chunk_sec=10):
    waveform, sr = sf.read(audio_path)
    if len(waveform.shape) == 1:
        waveform = waveform.reshape(1, -1)
    waveform = torch.from_numpy(waveform)
    chunk_samples = chunk_sec * sr
    total_samples = waveform.shape[1]
    results = []

    for i in range(0, total_samples, chunk_samples):
        segment_waveform = waveform[:, i:i+chunk_samples]
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_seg:
            segment_path = tmp_seg.name
        sf.write(segment_path, segment_waveform.squeeze(0).numpy(), sr)
        result = _whisper_model.transcribe(segment_path)
        text = result["text"].strip()
        os.remove(segment_path)
        if not text:
            continue
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = _emotion_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        results.append({
            "time": f"{i//sr:.2f}s - {(i+chunk_samples)//sr:.2f}s",
            "text": text,
            "emotion": LABELS[pred],
            "confidence": round(probs[0][pred].item(), 2)
        })
    return results

st.title("Emotion Detection from Audio (Streamlit)")

st.write("Upload or record an audio file (wav, mp3, ogg, webm) to analyze emotions segment-wise and overall.")

# File uploader
audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg", "webm"])

# Audio recorder
audio_bytes = audio_recorder()

# Determine which audio to process
if audio_file is not None:
    audio_bytes = audio_file.read()

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    # Save to temp file and process as before
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    whisper_model, tokenizer, emotion_model, device = load_models()
    segments = process_audio_in_chunks(tmp_path, whisper_model, tokenizer, emotion_model, device)
    os.remove(tmp_path)

    if not segments:
        st.error("No valid speech found.")
    else:
        # Compute overall score using weighted sum of confidence scores
        emotion_scores = {label: 0.0 for label in LABELS}
        for seg in segments:
            emotion_scores[seg["emotion"]] += seg["confidence"]
        overall = max(emotion_scores, key=emotion_scores.get)
        overall_score = round(emotion_scores[overall], 2)

        st.success(f"**Overall Emotion:** {overall} (Score: {overall_score})")
        st.write("### Segment-wise Analysis:")
        st.table([{**seg, "confidence": f"{seg['confidence']:.2f}"} for seg in segments]) 