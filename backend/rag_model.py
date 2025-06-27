# backend/rag_model.py
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch
import json
import faiss
import numpy as np

# Load the custom knowledge base
with open('data/knowledge_base.json') as f:
    knowledge_data = json.load(f)

# Initialize the tokenizer and model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="custom")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Create embeddings for the custom knowledge base
embeddings = []
for doc in knowledge_data:
    inputs = tokenizer(doc['text'], return_tensors="pt")
    embeddings.append(model.retrieval_embeddings(**inputs).detach().numpy())

embeddings = np.vstack(embeddings)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
retriever.retrieval_index = index

def generate_response(query):
    input_ids = tokenizer(query, return_tensors="pt").input_ids
    question_hidden_states = model.question_encoder(input_ids)[0]
    docs_dict = retriever(input_ids, question_hidden_states.numpy(), return_tensors="pt")
    doc_ids = docs_dict["doc_ids"]
    context_input_ids = docs_dict["context_input_ids"]
    outputs = model.generate(input_ids=input_ids, context_input_ids=context_input_ids)
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return response