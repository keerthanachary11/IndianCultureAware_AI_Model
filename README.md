# AI Model That Learns Humans Across Cultures

## Overview

AI Model That Learns Humans Across Cultures is a culturally aware multimodal artificial intelligence system designed to understand and interpret human interactions across diverse Indian cultural contexts. The system combines speech processing, multilingual text understanding, image recognition, retrieval-based reasoning, and large language models to generate culturally grounded and contextually accurate responses.

Unlike conventional AI systems that rely primarily on globally pretrained datasets, this project incorporates Indian cultural datasets and retrieval-augmented generation to improve cultural relevance, reduce hallucinations, and support multilingual communication.

## Key Features

* Multilingual speech recognition
* Language identification for Indian languages
* Cultural topic classification
* Festival and cultural image recognition
* Retrieval-Augmented Generation (RAG)
* Multimodal understanding (Speech + Text + Image)
* Context-aware response generation
* Cultural knowledge grounding using FAISS
* Support for 10+ Indian languages

## System Architecture

User Input (Speech/Text/Image)

↓

Speech Processing (Whisper)

↓

Language Detection (CNN + MFCC)

↓

Text Embedding (MiniLM)

↓

Topic Classification (Logistic Regression)

↓

Image Classification (ResNet-50)

↓

Multimodal Alignment (CLIP)

↓

Knowledge Retrieval (FAISS)

↓

Response Generation (Gemini 1.5 Pro)

↓

Final Cultural Response


## Datasets Used

### 1. Indian Culture Dataset

* Source: Hugging Face
* Size: ~50.5 KB
* Samples: <1000
* Contains:

  * Festivals
  * Rituals
  * Traditions
  * Arts
  * Architecture
  * Heritage practices

### 2. DRISHTIKON Dataset

* Source: Hugging Face
* Size: ~20.1 GB
* Samples: ~64,288 image-text pairs
* Languages: 15+
* Contains:

  * Cultural images
  * Visual Question Answering
  * Festivals
  * Cuisine
  * Rituals
  * State-wise cultural information

### 3. Indian Festival Classification Dataset

* Source: Kaggle
* Size: ~8,000–10,000 images
* Contains:

  * Diwali
  * Holi
  * Pongal
  * Ganesh Chaturthi
  * Durga Puja
  * Other Indian festivals

### 4. Indian Languages Audio Dataset

* Samples: ~10,000 audio clips
* Languages: 10 Indian languages
* Duration: 5 seconds per clip
* Used for:

  * Speech recognition
  * Language identification

### 5. Samanantar Dataset

* Source: AI4Bharat
* Size: ~49.7 million sentence pairs
* Languages: 11 Indic languages
* Used for:

  * Multilingual learning
  * Cross-lingual understanding
  * Translation support

## Models Used

### Whisper

Purpose:
Automatic Speech Recognition

Input:
Audio

Output:
Text

Why Used:
Supports multilingual speech recognition and handles accents effectively.

### CNN + MFCC

Purpose:
Language Detection

Input:
MFCC Features from Audio

Output:
Detected Language

Why Used:
Efficiently captures phonetic characteristics of Indian languages.

### MiniLM (Sentence Transformer)

Purpose:
Text Embedding

Input:
Text Query

Output:
Semantic Embeddings

Why Used:
Lightweight and effective semantic representation.

### Logistic Regression

Purpose:
Topic Classification

Input:
Text Embeddings

Output:
Cultural Category

Categories:

* Festivals
* Rituals
* Architecture
* Arts
* Heritage

### ResNet-50

Purpose:
Image Classification

Input:
Festival Images

Output:
Festival Label

Why Used:
High accuracy in image recognition tasks.

### CLIP

Purpose:
Multimodal Understanding

Input:
Image + Text

Output:
Shared Semantic Representation

Why Used:
Aligns textual and visual cultural information.

### FAISS

Purpose:
Knowledge Retrieval

Input:
Query Embedding

Output:
Relevant Cultural Documents

Why Used:
Fast similarity search for retrieval-augmented generation.

## Gemini 1.5 Pro

Purpose:
Response Generation

Input:
User Query + Retrieved Knowledge + Context

Output:
Culturally Aware Response

Why Used:
Provides natural, multilingual, and context-aware responses.

## Technologies Used

* Python
* TensorFlow
* PyTorch
* Scikit-Learn
* OpenCV
* Librosa
* Hugging Face Transformers
* Sentence Transformers
* FAISS
* Google Gemini API
* NumPy
* Pandas
* Flask

## Project Workflow

1. User submits speech, text, or image input.
2. Speech is converted to text using Whisper.
3. Language is identified using CNN with MFCC features.
4. Text is transformed into embeddings using MiniLM.
5. Query topic is classified using Logistic Regression.
6. Images are classified using ResNet-50.
7. CLIP aligns visual and textual information.
8. FAISS retrieves relevant cultural knowledge.
9. Gemini 1.5 Pro generates a culturally grounded response.
10. Final response is presented to the user.

## Future Enhancements

* Support for additional Indian languages
* Real-time multimodal interaction
* Regional dialect adaptation
* Cultural recommendation system
* Cultural sentiment analysis
* Mobile and cloud deployment

## Conclusion

This project demonstrates how multimodal AI, retrieval-based reasoning, and culturally curated datasets can be combined to build an inclusive AI system capable of understanding speech, text, and images while preserving cultural context. By integrating Whisper, MiniLM, ResNet-50, CLIP, FAISS, and Gemini 1.5 Pro, the system delivers culturally aware and contextually accurate responses tailored to Indian cultural diversity.
