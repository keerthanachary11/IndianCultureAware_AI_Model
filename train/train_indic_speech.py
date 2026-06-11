# ===============================
# Lightweight CNN for Indian Languages (Fixed Version)
# ===============================

# Install required packages if not already installed:
# pip install torch torchaudio librosa scikit-learn numpy

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ===============================
# Configuration
# ===============================
DATASET_PATH = r"C:\projects\ai_culture\Indian_Languages_Audio_Dataset"  # <-- Windows raw string
SAMPLE_RATE = 16000
N_MFCC = 40
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Load audio paths recursively
# ===============================
def load_audio_files(dataset_path):
    X = []
    y = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith((".wav", ".mp3")):
                full_path = os.path.join(root, f)
                # The language name is assumed to be the immediate parent folder
                language = os.path.basename(os.path.dirname(full_path))
                X.append(full_path)
                y.append(language)
    return X, y

audio_paths, labels = load_audio_files(DATASET_PATH)
print("Total audio files found:", len(audio_paths))
print("Sample files:", audio_paths[:5])

if len(audio_paths) == 0:
    raise ValueError("No audio files found! Check DATASET_PATH and folder structure.")

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
languages = label_encoder.classes_
print("Languages detected:", list(languages))

# Split train/test
train_paths, test_paths, train_labels, test_labels = train_test_split(
    audio_paths, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"Training samples: {len(train_paths)}, Testing samples: {len(test_paths)}")

# ===============================
# Dataset class
# ===============================
class AudioDataset(Dataset):
    def __init__(self, paths, labels, n_mfcc=40, sample_rate=16000):
        self.paths = paths
        self.labels = labels
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(path, sr=self.sample_rate)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        mfcc = np.mean(mfcc.T, axis=0)  # fixed-size feature vector
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Dataloaders
train_dataset = AudioDataset(train_paths, train_labels, N_MFCC, SAMPLE_RATE)
test_dataset = AudioDataset(test_paths, test_labels, N_MFCC, SAMPLE_RATE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# Define small CNN (MLP for MFCC)
# ===============================
class SmallCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SmallCNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SmallCNN(input_dim=N_MFCC, num_classes=len(languages)).to(DEVICE)

# ===============================
# Loss and optimizer
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===============================
# Training loop
# ===============================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

# ===============================
# Evaluation
# ===============================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy*100:.2f}%")

# ===============================
# Save model
# ===============================
torch.save(model.state_dict(), "indian_languages_cnn.pth")
print("Model saved successfully!")