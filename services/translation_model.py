# translation_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

models = {}
tokenizers = {}

languages = ["as","bn","gu","hi","kn","ml","mr","or","pa","ta","te"]

for lang in languages:
    path = f"models/translation_model_en_{lang}"
    try:
        tokenizers[lang] = AutoTokenizer.from_pretrained(path)
        models[lang] = AutoModelForSeq2SeqLM.from_pretrained(path)
    except Exception as e:
        print(f"[WARN] Translation model for {lang} not found: {e}")

def translate(text, src, tgt):
    # fallback if model missing
    if src == "en" and tgt not in models:
        return text
    if tgt == "en" and src not in models:
        return text

    if src == "en":
        tokenizer = tokenizers[tgt]
        model = models[tgt]
        inputs = tokenizer(f"translate English to {tgt}: {text}", return_tensors="pt")
    elif tgt == "en":
        tokenizer = tokenizers[src]
        model = models[src]
        inputs = tokenizer(f"translate {src} to English: {text}", return_tensors="pt")
    else:
        # src → tgt via English pivot
        english = translate(text, src, "en")
        return translate(english, "en", tgt)

    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)