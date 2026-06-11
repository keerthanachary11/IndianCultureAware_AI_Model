# tts_model.py
import os
from gtts import gTTS

def speak(text, lang="en"):
    if not os.path.exists("static"):
        os.makedirs("static")

    file = f"static/response_{lang}.mp3"
    tts = gTTS(text=text, lang=lang)
    tts.save(file)
    return file