import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# load dataset
data = pd.read_csv("data/indian_culture.csv")

# combine important fields into one text
documents = []

for _, row in data.iterrows():

    text = f"""
Topic: {row['topic']}
Description: {row['description']}
Region: {row['region']}
Category: {row['category']}
Subcategory: {row['subcategory']}
Source: {row['source']}
"""

    documents.append(text)

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# create embeddings
embeddings = model.encode(documents, convert_to_numpy=True)

# create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


def search(query, k=3):

    query_vector = model.encode([query], convert_to_numpy=True)

    distances, indices = index.search(query_vector, k)

    results = []

    for i in indices[0]:
        results.append(documents[i])

    return "\n\n".join(results)