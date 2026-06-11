# groq_model.py
from groq import Groq
from services.knowledge import load_knowledge
from services.rag_model import search

# ---------------- Initialize Groq client ----------------
client = Groq(api_key="GROQ_API_KEY")  # replace with your API key

# ---------------- Load external knowledge ----------------
knowledge = load_knowledge()

def ask_llm(question):
    """
    Ask the Groq LLM (currently supported model) with cultural knowledge context.
    """
    # Search relevant cultural knowledge (top 5)
    context = search(question, k=5)

    prompt = f"""
You are an expert on Indian culture.

Use the cultural knowledge below if relevant.

Cultural Knowledge:
{context}

If the answer is not found in the knowledge,
use your general knowledge about Indian culture.

Answer clearly and explain in detail.

Question:
{question}
"""

    # Use a supported Llama model
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # updated supported model
        messages=[{"role": "user", "content": prompt}]
    )

    return chat.choices[0].message.content