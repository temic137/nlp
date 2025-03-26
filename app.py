from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import logging

# Import your existing NLP utilities (copy from nlp_utils.py)
import cohere
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from huggingface_hub import InferenceClient
from rank_bm25 import BM25Okapi
import numpy as np
from groq import Groq

# Setup
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

cohere_token = os.getenv('COHERE_CLIENT')
huggingface_token = os.getenv('HUGGINGFACE_API_TOKEN')
groq_token = os.getenv('GROQ_API_KEY')

cohere_client = cohere.Client(api_key=cohere_token)
huggingface_client = InferenceClient(api_key=huggingface_token)
groq_client = Groq(api_key=groq_token)

# Copy your existing functions from nlp_utils.py
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
    return ' '.join(tokens)

def preprocess_data(pdf_data, folder_data, web_data):
    structured_data = []
    for item in pdf_data + folder_data:
        if isinstance(item, dict) and 'text' in item:
            sentences = sent_tokenize(item['text'])
            structured_data.extend([{'type': 'text', 'content': sent, 'source': 'pdf/folder'} for sent in sentences])
    if web_data:
        if isinstance(web_data, list) and web_data:
            web_data = web_data[0]
        if isinstance(web_data, dict):
            if 'title' in web_data:
                structured_data.append({'type': 'title', 'content': web_data['title'], 'source': 'web'})
            # Add more web data processing as needed
    return structured_data

def prepare_bm25_index(documents):
    tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
    return BM25Okapi(tokenized_documents)

def get_relevant_passages(question, documents, bm25, top_k=3):
    tokenized_question = word_tokenize(question.lower())
    scores = bm25.get_scores(tokenized_question)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [documents[i] for i in top_indices]

def generate_answer(question, documents, max_length=500, context_threshold=4000):
    total_context_length = sum(len(doc) for doc in documents)
    if total_context_length > context_threshold:
        bm25 = prepare_bm25_index(documents)
        relevant_info = get_relevant_passages(question, documents, bm25)
        context = " ".join(relevant_info)
    else:
        context = " ".join(documents)

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant..."},  # Your system prompt
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ],
            # model="llama3-8b-8192",
            model="llama-3.3-70b-specdec",
            temperature=0.0,
            max_tokens=max_length
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Error generating answer: {str(e)}")
        return "Error processing question"

@app.route('/generate_answer', methods=['POST'])
def api_generate_answer():
    data = request.json
    if not data or 'data' not in data or 'question' not in data:
        return jsonify({"error": "Missing data or question"}), 400

    try:
        chatbot_data = json.loads(data['data']) if isinstance(data['data'], str) else data['data']
        if isinstance(chatbot_data, list) and chatbot_data:
            chatbot_data = chatbot_data[-1]

        pdf_data = chatbot_data.get('pdf_data', [])
        folder_data = chatbot_data.get('folder_data', [])
        web_data = chatbot_data.get('web_data', {})
        
        structured_data = preprocess_data(pdf_data, folder_data, web_data)
        text_data = [item['content'] for item in structured_data if 'content' in item]
        
        answer = generate_answer(data['question'], text_data)
        return jsonify({"answer": answer})
    except Exception as e:
        logging.error(f"Error in NLP service: {str(e)}")
        return jsonify({"error": "Failed to generate answer"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)