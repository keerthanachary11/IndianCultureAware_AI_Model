# language_detect.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

class SmallCNN(nn.Module):
    def __init__(self, input_dim=40, num_classes=10):
        super(SmallCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

languages = ["bn","gu","hi","kn","ml","mr","pa","ta","te","ur"]

model = SmallCNN(40, 10)
model.load_state_dict(
    torch.load("models/indian_languages_cnn.pth", map_location="cpu")
)
model.eval()

def detect_language(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # normalize to match training
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    x = torch.tensor(mfcc).float().unsqueeze(0)

    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).item()

    return languages[pred]