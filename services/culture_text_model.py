# culture_text_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np

class CultureTextModel(nn.Module):
    def __init__(self, input_dim=384, num_classes=5):
        super(CultureTextModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

model = CultureTextModel()
checkpoint = torch.load("models/culture_text_model.pt", map_location="cpu")
model_dict = model.state_dict()
filtered_ckpt = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(filtered_ckpt)
model.load_state_dict(model_dict)
model.eval()

topics = ["Festival", "Architecture", "Art Form", "Ritual", "Performing Arts"]

def detect_topic(text):
    embedding = embed_model.encode([text])
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)  # normalize
    x = torch.tensor(embedding).float()
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()
    return topics[pred]