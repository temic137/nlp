from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import logging
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import cohere
from huggingface_hub import InferenceClient
from rank_bm25 import BM25Okapi
from groq import Groq

# Initialize Flask app
app = Flask(__name__)

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Logs to console, which Azure captures
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables (optional for local dev, Azure uses App Settings)
load_dotenv()

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load API keys from environment variables (set in Azure App Settings)
cohere_token = os.getenv('COHERE_CLIENT')
huggingface_token = os.getenv('HUGGINGFACE_API_TOKEN')
groq_token = os.getenv('GROQ_API_KEY')

# Initialize clients with error handling
try:
    cohere_client = cohere.Client(api_key=cohere_token)
    huggingface_client = InferenceClient(api_key=huggingface_token)
    groq_client = Groq(api_key=groq_token)
    logger.info("Successfully initialized NLP clients")
except Exception as e:
    logger.error(f"Failed to initialize NLP clients: {str(e)}")
    raise  # Fail fast in production if clients can't initialize

# NLP utility functions
def preprocess_text(text):
    """Preprocess text by tokenizing, lemmatizing, and removing stopwords."""
    try:
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error in preprocess_text: {str(e)}")
        return text  # Return original text as fallback

def preprocess_data(pdf_data, folder_data, web_data):
    """Convert raw data into structured format for processing."""
    structured_data = []
    try:
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
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
    return structured_data

def prepare_bm25_index(documents):
    """Prepare BM25 index for document ranking."""
    try:
        tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
        return BM25Okapi(tokenized_documents)
    except Exception as e:
        logger.error(f"Error in prepare_bm25_index: {str(e)}")
        raise

def get_relevant_passages(question, documents, bm25, top_k=3):
    """Retrieve top-k relevant passages using BM25."""
    try:
        tokenized_question = word_tokenize(question.lower())
        scores = bm25.get_scores(tokenized_question)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [documents[i] for i in top_indices]
    except Exception as e:
        logger.error(f"Error in get_relevant_passages: {str(e)}")
        return documents[:top_k]  # Fallback to first top_k documents

def generate_answer(question, documents, max_length=500, context_threshold=4000):
    """Generate an answer using Groq based on context and question."""
    try:
        total_context_length = sum(len(doc) for doc in documents)
        if total_context_length > context_threshold:
            bm25 = prepare_bm25_index(documents)
            relevant_info = get_relevant_passages(question, documents, bm25)
            context = " ".join(relevant_info)
        else:
            context = " ".join(documents)

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant providing accurate and concise answers."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
            ],
            model="llama-3.3-70b-specdec",
            temperature=0.0,
            max_tokens=max_length
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return "Error processing question"

# API endpoint
@app.route('/generate_answer', methods=['POST'])
def api_generate_answer():
    """Handle POST requests to generate answers."""
    data = request.get_json()
    if not data or 'data' not in data or 'question' not in data:
        logger.warning("Invalid request: Missing data or question")
        return jsonify({"error": "Missing data or question"}), 400

    try:
        # Parse chatbot data
        chatbot_data = json.loads(data['data']) if isinstance(data['data'], str) else data['data']
        if isinstance(chatbot_data, list) and chatbot_data:
            chatbot_data = chatbot_data[-1]

        pdf_data = chatbot_data.get('pdf_data', [])
        folder_data = chatbot_data.get('folder_data', [])
        web_data = chatbot_data.get('web_data', {})
        
        # Process data and generate answer
        structured_data = preprocess_data(pdf_data, folder_data, web_data)
        text_data = [item['content'] for item in structured_data if 'content' in item]
        
        answer = generate_answer(data['question'], text_data)
        logger.info(f"Generated answer for question: {data['question']}")
        return jsonify({"answer": answer})
    except Exception as e:
        logger.error(f"Error in NLP service: {str(e)}")
        return jsonify({"error": "Failed to generate answer"}), 500

# Debug route for troubleshooting (optional)
@app.route('/debug', methods=['GET'])
def debug():
    """Return list of registered routes for debugging."""
    return jsonify({"routes": [str(rule) for rule in app.url_map.iter_rules()]})

# WSGI entry point for Azure
application = app

# Only run development server if not in production (e.g., local testing)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
