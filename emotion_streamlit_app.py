import sys
import streamlit as st
import numpy as np
import soundfile as sf
import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tempfile
import os
from audio_recorder_streamlit import audio_recorder
import subprocess
import ffmpeg

st.set_page_config(page_title="Emotion Detection from Audio", layout="centered")

st.sidebar.text(f"Python version: {sys.version}")

def verify_ffmpeg():
    try:
        # Check if ffmpeg is available
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        st.error("FFmpeg is not installed. Please install FFmpeg to use this application.")
        return False

# Verify FFmpeg at startup
if not verify_ffmpeg():
    st.stop()

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


def convert_to_wav(audio_bytes, source_filename):
    """
    Converts audio data to a temporary WAV file using ffmpeg,
    resampling to 16kHz and converting to mono.
    """
    try:
        # Give a default extension if none is provided.
        input_suffix = os.path.splitext(source_filename)[-1] if source_filename else ".wav"

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=input_suffix
        ) as tmp_in:
            tmp_in.write(audio_bytes)
            input_path = tmp_in.name

        # Output path for the converted file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_path = tmp_out.name

        command = [
            "ffmpeg",
            "-i", input_path,
            "-ac", "1",              # Mono channel
            "-ar", "16000",           # 16kHz sample rate
            "-acodec", "pcm_s16le", # 16-bit PCM WAV
            "-y",                   # Overwrite output file
            output_path,
        ]

        # Hide ffmpeg startup banner and logs unless there is an error
        subprocess.run(
            command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )

    except FileNotFoundError:
        st.error(
            "ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH."
        )
        return None
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode()
        st.error(
            f"Error converting audio. Please upload a valid audio file. Details: {error_message}"
        )
        return None
    finally:
        # Clean up the temporary input file
        if "input_path" in locals() and os.path.exists(input_path):
            os.remove(input_path)

    return output_path


LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise', 'neutral']
# Audio chunking and inference
@st.cache_data(show_spinner=False)
def process_audio_in_chunks(audio_path, _whisper_model, _tokenizer, _emotion_model, _device, chunk_sec=10):
    waveform, sr = sf.read(audio_path)
    if len(waveform.shape) == 1:
        waveform = waveform.reshape(1, -1)  # mono: [1, samples]
    else:
        waveform = waveform.T  # stereo: [channels, samples]
    waveform = torch.from_numpy(waveform)
    chunk_samples = chunk_sec * sr
    total_samples = waveform.shape[1]
    segments = []

    def predict_emotion(text):
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = _emotion_model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        return LABELS[pred], round(probs[0][pred].item(), 2)

    for i in range(0, total_samples, chunk_samples):
        print(f"i: {i}, chunk_samples: {chunk_samples}, total_samples: {total_samples}")
        print(f"Chunk time: {i/sr:.2f}s–{(i+chunk_samples)/sr:.2f}s")
        segment_waveform = waveform[:, i:i+chunk_samples]
        print(f"Segment waveform shape: {segment_waveform.shape}")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_seg:
            segment_path = tmp_seg.name
        data = segment_waveform.numpy().T  # shape: (samples, channels)
        sf.write(segment_path, data, sr)

        result = _whisper_model.transcribe(segment_path)
        transcript = result["text"].strip()

        if len(_tokenizer.tokenize(transcript)) > 128:
            sentences = transcript.split(". ")
            for s in sentences:
                if len(_tokenizer.tokenize(s)) > 0:
                    label, conf = predict_emotion(s)
                    segments.append({
                        "time": f"{i/sr:.2f}s–{(i+chunk_samples)/sr:.2f}s",
                        "text": s,
                        "emotion": label,
                        "confidence": conf
                    })
        else:
            label, conf = predict_emotion(transcript)
            segments.append({
                "time": f"{i/sr:.2f}s–{(i+chunk_samples)/sr:.2f}s",
                "text": transcript,
                "emotion": label,
                "confidence": conf
            })

        os.remove(segment_path)
    print(f"Total segments: {len(segments)}")
    return segments

st.title(":microphone: Emotion Detection from Audio")

st.markdown("""
Welcome! This app detects emotions from your speech using state-of-the-art AI models.

**How to use:**
- **Option 1:** Upload an audio file (WAV, MP3, OGG, WEBM)
- **Option 2:** Record your voice directly in the browser
- After uploading or recording, the app will transcribe your speech, analyze it in segments, and show the detected emotions.

:bulb: **Tips:**
- For best results, speak clearly and avoid background noise.
- The recording will stop automatically if nothing is spoken for **5 seconds**.
- Segment-wise analysis means your audio is split into chunks (default: 10 seconds each) and each chunk is analyzed separately.
""")

st.info("""
**Upload Audio**
- Click below to upload a file from your device.
- Supported formats: WAV, MP3, OGG, WEBM
""")
audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "ogg", "webm"])

st.info("""
**Or Record Audio**
- Click the microphone button to start recording.
- To stop recording, click the microphone button again. Otherwise, it will stop automatically after 30 seconds of recording.
""")
audio_bytes = None
recorded_audio = audio_recorder(
    energy_threshold=(-1.0, 1.0),
    pause_threshold=30.0,
)

if audio_file is not None:
    audio_bytes = audio_file.read()
elif recorded_audio is not None:
    audio_bytes = recorded_audio

if audio_bytes:
    st.audio(audio_bytes)
    
    source_filename = audio_file.name if audio_file else "recording.wav"
    converted_path = None
    
    try:
        with st.spinner("Converting audio to standard WAV format..."):
            converted_path = convert_to_wav(audio_bytes, source_filename)

        if converted_path:
            with st.spinner("Processing audio and running inference..."):
                whisper_model, tokenizer, emotion_model, device = load_models()
                segments = process_audio_in_chunks(converted_path, whisper_model, tokenizer, emotion_model, device)
            
            if not segments:
                st.warning("Could not detect any speech in the audio. Please try again.")
            else:
                # Compute overall score
                emotion_scores = {label: 0.0 for label in LABELS}
                total_confidence = 0.0
                for seg in segments:
                    emotion_scores[seg["emotion"]] += seg["confidence"]
                    total_confidence += seg["confidence"]
                
                if total_confidence > 0:
                    overall_emotion = max(emotion_scores, key=emotion_scores.get)
                    # Normalize the score to be a percentage of total confidence
                    overall_score = (emotion_scores[overall_emotion] / total_confidence) * 100
                    st.success(f"**Overall Emotion:** {overall_emotion} ({overall_score:.2f}%)")
                else:
                    st.info("No dominant emotion could be determined.")

                st.write("### Segment-wise Analysis:")
                st.table([{**seg, "confidence": f"{seg['confidence']:.2f}"} for seg in segments])
    finally:
        # Clean up the converted file
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)