# %%
import os
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import pickle

# %%
def documents(folder):  #exatracting text files from the folder
    docs=[]
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(os.path.join(folder,filename),'r',encoding='utf-8') as f:
                docs.append(f.read())
    return docs

# %%
def document_in_blocks(documents,chunk_sizes=200,chunk_overlap=20):#now dividing
    #the data extracted above into chuks and storing in a list
    split=RecursiveCharacterTextSplitter(chunk_size=chunk_sizes,chunk_overlap=chunk_overlap)
    all_blocks=[]
    for doc in documents:
        all_blocks.extend(split.split_text(doc))
    return all_blocks

def store_in_faiss_index(blocks,embedding_model_name="all-MiniLM-L6-v2"):
    model=SentenceTransformer(embedding_model_name)
    embed=model.encode(blocks)  #now convering chunks into embeddings(vectorization)
    
    index=faiss.IndexFlatL2(embed.shape[1])
    index.add(embed)
    faiss.write_index(index,"faiss-index")   #file storing embeddings(vectors)
    with open('blocks',"wb") as f:     #pickle storing the original file but pickle
        pickle.dump(blocks,f)               #stores in binary format therefore write binary
    print("vectors & original data saved")                               #is used
     
documents=documents("C:/Users/user/Desktop/c++ programs/RAG Powered Q&A ASSISTANT")
blocks=document_in_blocks(documents)
store_in_faiss_index(blocks)

# %%
def load_data():
    index=faiss.read_index("faiss-index")
    with open("blocks",'rb') as f:
        original_data=pickle.load(f)
    return index ,original_data

import re
import math
from sympy import sympify


def calculation(query:str):#calculation if calculate word found in query
    expression = re.findall(r"[\d\.\+\-\*\/\(\)\^ ]+", query) #regex for operators
    try:
        if expression:
            result=sympify(expression[0])
            return f"the result is {result}"
    except expression as e:
        return f"errors{str(e)}"
    return "could not calculate"

from transformers import pipeline
generator = pipeline("text-generation", model="distilgpt2")

import requests

def definition_s(query: str):
    # Extract the  term(input) 
    term = query.strip().split()[-1]

    # Make a request to the dictionary API
    response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}")

    if response.status_code == 200:  #validity check
        data = response.json()
        try:
            definition = data[0]['meanings'][0]['definitions'][0]['definition']
            return f"{term.capitalize()}: {definition}"
        except (IndexError, KeyError):
            return "Definition not found in the response."
    else:
        return f"No definition found for '{term}'."

# def definition_s(query:str):
#         prompted=f"define the term {query.strip()}"
#         result=generator(prompted,max_length=100,do_sample=True,temperature=0.3,truncation=True)
#         return result[0]["generated_text"].strip()
    
def get_query_embedding(query, model):
    return model.encode([query])

def find_similar_chunks(query_embedding, index, k=3):
    distances, indices = index.search(query_embedding, k)
    return indices[0]

def get_top_chunks(indices, chunks):
    return [chunks[i] for i in indices]

def answer_from_chunks(chunks, query):
    context = "\n\n".join(chunks)
    prompt = f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    #response
    result = generator(prompt, max_length=300, do_sample=True, temperature=0.7)

    #
    return result[0]["generated_text"].strip()

    
def detect_words(query: str):
    query_lower = query.lower()
    if any(word in query_lower for word in ["calculate", "what is", "sum", "add", "subtract", "multiply", "divide"]):
        return "calculate"
    elif any(word in query_lower for word in ["define", "meaning of", "what does"]):
        return "define"
    else:
        return "search"

    
def process_query(query):
    prompt = detect_words(query)

    if prompt == "calculate":
        return calculation(query)

    elif prompt == "define":
        return definition_s(query)

    else:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        index, chunks = load_data()
        query_embedding = get_query_embedding(query, model)
        top_indices = find_similar_chunks(query_embedding, index)
        top_chunks = get_top_chunks(top_indices, chunks)
        return answer_from_chunks(top_chunks, query)


# %%
while True:
    query=input("ENTER YOUR QUERY")
    if query.lower() in['exit','terminate','quit',"stop"]:
        print("Quiting")
        break
    answer=process_query(query)
    print("\n",answer[:1000])


