import os
import tempfile
from flask import Flask, request, jsonify, send_file
from faster_whisper import WhisperModel
from gtts import gTTS
import google.generativeai as genai

# Use ONNX and tiny model
model = WhisperModel("tiny.en", compute_type="int8", cpu_threads=2)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
chat_model = genai.GenerativeModel("models/gemini-1.5-flash-8b")

app = Flask(__name__)

@app.route('/')
def home():
    return "AI Voice Chatbot is running."

@app.route('/ask', methods=['POST'])
def ask():
    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio_file.save(f.name)
        audio_path = f.name

    # Transcribe audio
    segments, _ = model.transcribe(audio_path)
    question = " ".join([seg.text.strip() for seg in segments])

    # Get Gemini response (limit to ~100 words)
    response = chat_model.generate_content(question)
    answer = response.text.strip()
    answer = " ".join(answer.split()[:100])  # Trim to ~100 words

    # Convert to speech
    tts = gTTS(text=answer, lang='en')
    tts_path = tempfile.mktemp(suffix=".mp3")
    tts.save(tts_path)

    os.remove(audio_path)

    return jsonify({
        "question": question,
        "answer": answer,
        "audio_url": "/audio_response?path=" + tts_path
    })

@app.route('/audio_response')
def audio_response():
    path = request.args.get('path')
    return send_file(path, mimetype="audio/mpeg")

if __name__ == "__main__":
    app.run(debug=True)



