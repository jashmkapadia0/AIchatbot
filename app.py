import os
import tempfile
import time
import requests
from flask import Flask, request, jsonify, send_file
from faster_whisper import WhisperModel
from gtts import gTTS
import google.generativeai as genai

# Load Supabase credentials from environment variables

# SUPABASE_URL = "https://agslqbilarldedalvqgx.supabase.co"
# SUPABASE_KEY = os.environ.get("SUPABASE_API_KEY", "").strip()
# SUPABASE_BUCKET = "audios"

# def upload_to_supabase(file_path, filename):
#     with open(file_path, 'rb') as f:
#         headers = {
#             "apikey": SUPABASE_KEY,
#             "Authorization": f"Bearer {SUPABASE_KEY}",
#             "Content-Type": "application/octet-stream"
#         }
#         url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{filename}"
#         res = requests.post(url, headers=headers, data=f)

#     if res.status_code in [200, 201]:
#         print(f"✅ Uploaded to Supabase: {filename}")
#     else:
#         print(f"❌ Upload failed: {res.status_code} - {res.text}")

# Use ONNX and tiny model
model = WhisperModel("tiny.en", compute_type="int8", cpu_threads=2)

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
chat_model = genai.GenerativeModel("models/gemini-1.5-flash-8b")

app = Flask(__name__)

@app.route("/")
def index():
    return send_file("index.html")

@app.route('/ask', methods=['POST'])
def ask():
    audio_file = request.files['audio']
    timestamp = int(time.time())
    filename = f"user_question_{timestamp}.wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        audio_file.save(f.name)
        audio_path = f.name

    # Upload original audio to Supabase
    # upload_to_supabase(audio_path, filename)

    # Transcribe audio
    segments, _ = model.transcribe(audio_path)
    question = " ".join([seg.text.strip() for seg in segments])

    # Get Gemini response (limit to ~100 words)
    response = chat_model.generate_content(question)
    answer = response.text.strip().replace('*', '')
    answer = " ".join(answer.split()[:100])  # limit length

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
