from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
STORAGE_DIR = "storage"
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index")
DOCS_JSON_PATH = os.path.join(STORAGE_DIR, "docs_metadata.json")

# Initialize models and storage
os.makedirs(STORAGE_DIR, exist_ok=True)
embedding_model = None
vectorstore = None
qa_chain = None

def init_models():
    global embedding_model, vectorstore, qa_chain
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(FAISS_INDEX_PATH):
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model)
        llm = ChatOllama(model="mistral")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
        )

@app.before_request
def before_request():
    init_models()

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the current status of the system"""
    try:
        has_documents = os.path.exists(FAISS_INDEX_PATH)
        docs_metadata = []
        if os.path.exists(DOCS_JSON_PATH):
            with open(DOCS_JSON_PATH, 'r', encoding='utf-8') as f:
                docs_metadata = json.load(f)
        
        return jsonify({
            'status': 'ready' if has_documents else 'no_documents',
            'has_documents': has_documents,
            'document_count': len(docs_metadata),
            'docs_metadata': docs_metadata
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    """Ask a question to the system"""
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400

        if not qa_chain:
            return jsonify({'error': 'System not initialized or no documents loaded'}), 400

        answer = qa_chain.run(data['question'])
        return jsonify({
            'question': data['question'],
            'answer': answer
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all loaded documents"""
    try:
        if os.path.exists(DOCS_JSON_PATH):
            with open(DOCS_JSON_PATH, 'r', encoding='utf-8') as f:
                docs_metadata = json.load(f)
            return jsonify(docs_metadata)
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001) 