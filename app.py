from flask import Flask, request, render_template, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
import torch
import torchaudio
import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import uuid
import subprocess
import tempfile

# Setup
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
model_path = "Akhil199797/emotion-distilbert-final"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Whisper and Emotion Model
whisper_model = whisper.load_model("base")
tokenizer = AutoTokenizer.from_pretrained(model_path)
emotion_model = AutoModelForSequenceClassification.from_pretrained(model_path)
emotion_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model.to(device)

label_names = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise', 'love', 'confusion', 'boredom']

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        return label_names[pred_class], round(probs[0][pred_class].item(), 4)

# Audio Chunking and Inference
def process_audio_in_chunks(audio_path, chunk_sec=10):
    #waveform, sr = torchaudio.load(audio_path)
    waveform, sr = sf.read(audio_path)
    if len(waveform.shape) == 1:
        waveform = waveform.reshape(1, -1)
    waveform = torch.from_numpy(waveform)
    chunk_samples = chunk_sec * sr
    total_samples = waveform.shape[1]
    segments = []

    for i in range(0, total_samples, chunk_samples):
        segment_waveform = waveform[:, i:i+chunk_samples]
        segment_path = f"chunk_{i//chunk_samples}.wav"
        #torchaudio.save(segment_path, segment_waveform, sample_rate=sr)
        sf.write(segment_path, segment_waveform.squeeze(0).numpy(), sr)

        result = whisper_model.transcribe(segment_path)
        transcript = result["text"].strip()

        if len(tokenizer.tokenize(transcript)) > 128:
            sentences = transcript.split(". ")
            for s in sentences:
                if len(tokenizer.tokenize(s)) > 0:
                    label, conf = predict_emotion(s)
                    segments.append({
                        "time": f"{i//sr}s–{(i+chunk_samples)//sr}s",
                        "text": s,
                        "emotion": label,
                        "confidence": conf
                    })
        else:
            label, conf = predict_emotion(transcript)
            segments.append({
                "time": f"{i//sr}s–{(i+chunk_samples)//sr}s",
                "text": transcript,
                "emotion": label,
                "confidence": conf
            })

        os.remove(segment_path)
    return segments


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(filepath)

    # Convert to WAV using ffmpeg
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        wav_path = tmp_wav.name
    try:
        # ffmpeg -y -i input -ar 16000 -ac 1 output.wav
        subprocess.run([
            'ffmpeg', '-y', '-i', filepath, '-ar', '16000', '-ac', '1', wav_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        os.remove(filepath)
        return f"Audio conversion failed: {e.stderr.decode()}", 400

    segments = process_audio_in_chunks(wav_path, chunk_sec=10)
    os.remove(filepath)
    os.remove(wav_path)

    if not segments:
        return render_template("results.html", error="No valid speech found.")

    # Compute overall score using weighted sum of confidence scores
    emotion_scores = {label: 0.0 for label in label_names}
    for seg in segments:
        emotion_scores[seg["emotion"]] += seg["confidence"]
    overall = max(emotion_scores, key=emotion_scores.get)
    overall_score = round(emotion_scores[overall], 2)
    print(segments)
    print(overall)
    print(overall_score)
    return render_template("results.html", segments=segments, overall=overall, overall_score=overall_score)

if __name__ == "__main__":
    app.run(debug=True)
