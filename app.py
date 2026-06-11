from flask import Flask, render_template, request, jsonify
import os
import uuid

# ---------------- Services ----------------
from services.groq_model import ask_llm
from services.speech_to_text import speech_to_text
from services.tts_model import speak
from services.language_detect import detect_language
from services.culture_text_model import detect_topic
from services.image_model import predict_image

app = Flask(__name__)

# ---------------- AI Pipeline ----------------
def process_question(text, audio_path=None):
    """
    Process user input:
    1. Detect language from audio (optional)
    2. Detect cultural topic
    3. Ask Groq LLM
    """
    lang = "en"
    if audio_path and os.path.exists(audio_path):
        lang = detect_language(audio_path)

    # Detect cultural topic
    topic = detect_topic(text)

    # Ask LLM
    response = ask_llm(text)

    return response, topic, lang

# ---------------- Text Chat API ----------------
@app.route("/chat", methods=["POST"])
def chat():
    if not request.is_json:
        return jsonify({"error": "Expected JSON request"}), 415

    data = request.get_json()
    user_text = data.get("message")
    if not user_text:
        return jsonify({"error": "No message provided"}), 400

    response, topic, lang = process_question(user_text)

    audio_file = speak(response)

    return jsonify({
        "response": response,
        "topic": topic,
        "audio": audio_file,
        "lang": lang
    })

# ---------------- Voice Input API ----------------
@app.route("/voice", methods=["POST"])
def voice():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["audio"]
    audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
    file.save(audio_path)

    # Speech → text
    text = speech_to_text(audio_path)

    response, topic, lang = process_question(text, audio_path=audio_path)

    audio_file = speak(response)

    os.remove(audio_path)

    return jsonify({
        "text": text,
        "response": response,
        "topic": topic,
        "audio": audio_file,
        "lang": lang
    })

# ---------------- Image Input API ----------------
@app.route("/image", methods=["POST"])
def image():

    file = request.files["image"]

    path = "temp.jpg"
    file.save(path)

    label, confidence = predict_image(path)

    response = ask_llm(
        f"Explain the Indian festival {label} in detail including history, rituals and cultural significance."
    )

    return jsonify({
        "label": label,
        "confidence": confidence,
        "response": response
    })
# ---------------- Home Page ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(debug=True)