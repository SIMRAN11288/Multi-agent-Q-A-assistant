# Multi-agent-Q-A-assistant
A RAG-Powered multi-agent Q/A assistant built using  Hugging facing module and dictionary API
How the Code Works:
User Query Input:
The program first prompts the user to enter a query.

Mathematical Computation:
If the query includes mathematical keywords like "calculate", "sum", "add", etc., it uses the sympify function to evaluate the mathematical expression safely and accurately.

Word Definition:
If the query includes the word "define", the program fetches the definition from a free Dictionary API.
     Note: The dictionary has limited entries, so not all words may return a definition.

Document-Based Answering:
For all other queries, the code retrieves relevant information from a set of documents by comparing the query's embedding to document chunk embeddings using FAISS.

Major Components Used in the Project:
 FAISS (Facebook AI Similarity Search)
 FAISS is a library developed by Facebook AI Research for efficient similarity search and clustering of dense vectors.
 In this project, it is used to match the query embedding with document embeddings to find the most relevant content.

Dictionary API
 URL: https://api.dictionaryapi.dev/
 A free and open-source API that provides:   Word definitions       Phonetics
Requires no API key or authentication.

Integrated in the project to define terms extracted from the user's query.

sympify (from SymPy Library)
 Converts a string representation of a math expression into a symbolic expression.
 Safer than eval() and supports mathematical operations like:

Arithmetic
Square roots
Powers

