# app.py
from flask import Flask, render_template, request
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pickle
import re
import math
from sympy import sympify
from transformers import pipeline
import requests

# Initialize Flask app
app = Flask(__name__)

# Global variable for SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="distilgpt2")

# Route to display the form and get user query
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        result = process_query(query)
        return render_template('index.html', result=result)
    return render_template('index.html', result=None)

# Function to process queries
def process_query(query):
    prompt = detect_words(query)

    if prompt == "calculate":
        return calculation(query)
    elif prompt == "define":
        return definition_s(query)
    else:
        index, chunks = load_data()
        query_embedding = get_query_embedding(query, model)
        top_indices = find_similar_chunks(query_embedding, index)
        top_chunks = get_top_chunks(top_indices, chunks)
        return answer_from_chunks(top_chunks, query)

# Function to detect type of query
def detect_words(query: str):
    query_lower = query.lower()
    if any(word in query_lower for word in ["calculate", "what is", "sum", "add", "subtract", "multiply", "divide"]):
        return "calculate"
    elif any(word in query_lower for word in ["define", "meaning of", "what does"]):
        return "define"
    else:
        return "search"

# Function for calculation (if query includes 'calculate')
def calculation(query: str):
    expression = re.findall(r"[\d\.\+\-\*\/\(\)\^ ]+", query)
    try:
        if expression:
            result = sympify(expression[0])
            return f"the result is {result}"
    except Exception as e:
        return f"errors: {str(e)}"
    return "could not calculate"

# Function for definition lookup (using Dictionary API)
def definition_s(query: str):
    term = query.strip().split()[-1]
    response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}")
    if response.status_code == 200:
        data = response.json()
        try:
            definition = data[0]['meanings'][0]['definitions'][0]['definition']
            return f"{term.capitalize()}: {definition}"
        except (IndexError, KeyError):
            return "Definition not found in the response."
    else:
        return f"No definition found for '{term}'."

# Load FAISS index and data
def load_data():
    index = faiss.read_index("faiss-index")
    with open("blocks", 'rb') as f:
        original_data = pickle.load(f)
    return index, original_data

# Get query embedding using SentenceTransformer
def get_query_embedding(query, model):
    return model.encode([query])

# Find similar chunks from the index
def find_similar_chunks(query_embedding, index, k=3):
    distances, indices = index.search(query_embedding, k)
    return indices[0]

# Get the top chunks from the index
def get_top_chunks(indices, chunks):
    return [chunks[i] for i in indices]

# Generate an answer from the top chunks
def answer_from_chunks(chunks, query):
    context = "\n\n".join(chunks)
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=300, do_sample=True, temperature=0.7)
    return result[0]["generated_text"].strip()

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
