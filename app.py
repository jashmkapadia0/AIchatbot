import os
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
from faster_whisper import WhisperModel
import google.generativeai as genai
from gtts import gTTS

app = Flask(__name__)

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Replace with your key
model = WhisperModel("base", compute_type="int8")

@app.route('/')
def index():
    return open("index.html", encoding="utf-8").read()

@app.route('/ask', methods=['POST'])
def ask():
    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio_file.save(f.name)
        audio_path = f.name

    segments, _ = model.transcribe(audio_path)
    question = " ".join([seg.text.strip() for seg in segments])

    # Gemini response
    response = genai.GenerativeModel("models/gemini-1.5-flash-8b").generate_content(question)
    full_answer = response.text.strip()

    # Limit to 100 words
    words = full_answer.split()
    answer = " ".join(words[:100])

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
    path = request.args.get("path")
    return send_file(path, mimetype='audio/mpeg')

if __name__ == '__main__':
    app.run(debug=True)


