def load_knowledge():
    with open("services/knowledge.txt", "r", encoding="utf-8") as f:
        return f.read()