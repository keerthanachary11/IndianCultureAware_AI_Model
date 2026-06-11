import whisper

model = whisper.load_model("tiny")

def speech_to_text(audio_path):

    result = model.transcribe(audio_path, fp16=False)

    return result["text"]