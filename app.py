from flask import Flask, request, jsonify, render_template
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import PyPDF2
import os
import json
from datetime import datetime
import gc
import atexit

# === Init Flask ===
app = Flask(__name__)
os.makedirs("pdf", exist_ok=True)
os.makedirs("storage", exist_ok=True)

# === Configuration ===
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
STORAGE_DIR = "storage"
FAISS_INDEX_PATH = os.path.join(STORAGE_DIR, "faiss_index")
DOCS_JSON_PATH = os.path.join(STORAGE_DIR, "docs_metadata.json")

# === Lazy Loading of Models ===
llm = None
embedding_model = None
vectorstore = None
qa_chain = None
conversation_chain = None
_is_first_request = True

def init_llm():
    global llm
    if llm is None:
        llm = ChatOllama(model="qwen2.5")
    return llm

def init_embedding():
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    return embedding_model

def init_conversation_chain():
    global conversation_chain
    if conversation_chain is None:
        conversation_template = """Tu es un assistant intelligent et amical qui peut répondre à la fois aux questions générales et aux questions spécifiques sur des documents.

Contexte du système:
- Documents chargés: {has_documents}
- Nombre de documents: {doc_count}

Question de l'utilisateur: {question}

Instructions:
1. Si la question est générale (salutations, questions sur tes capacités, etc.), réponds de manière naturelle et conviviale.
2. Si la question porte sur les documents et qu'ils sont chargés, utilise leur contenu pour répondre.
3. Si la question porte sur les documents mais qu'aucun n'est chargé, suggère gentiment d'en charger.
4. Adapte ton ton pour être toujours serviable et professionnel.

Réponse:"""

        conversation_prompt = PromptTemplate(
            template=conversation_template,
            input_variables=["question", "has_documents", "doc_count"]
        )
        conversation_chain = LLMChain(llm=init_llm(), prompt=conversation_prompt)
    return conversation_chain

# === PDF Processing Functions ===
def load_and_split(pdf_path):
    if not is_valid_pdf(pdf_path):
        raise ValueError("Invalid or corrupted PDF file")
    
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    
    # Libérer la mémoire
    del docs
    gc.collect()
    
    return chunks

# === FAISS Functions ===
def build_faiss_index(chunks, save=True):
    global vectorstore
    vectorstore = FAISS.from_documents(chunks, init_embedding())
    if save:
        vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore

def load_or_create_faiss_index():
    global vectorstore, qa_chain
    try:
        if os.path.exists(FAISS_INDEX_PATH):
            app.logger.info("Loading existing FAISS index...")
            vectorstore = FAISS.load_local(
                FAISS_INDEX_PATH,
                init_embedding(),
                allow_dangerous_deserialization=True
            )
            # Initialize QA chain with loaded vectorstore
            if vectorstore is not None:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                qa_chain = RetrievalQA.from_chain_type(llm=init_llm(), retriever=retriever)
                app.logger.info("FAISS index and QA chain loaded successfully")
            return vectorstore
    except Exception as e:
        app.logger.error(f"Error loading FAISS index: {str(e)}")
        vectorstore = None
        qa_chain = None
    return None

# === Save Functions ===
def save_system_state():
    """Save system state before shutdown"""
    if vectorstore is not None:
        try:
            app.logger.info("Saving FAISS index...")
            vectorstore.save_local(FAISS_INDEX_PATH)
            app.logger.info("FAISS index saved successfully")
        except Exception as e:
            app.logger.error(f"Error saving FAISS index: {str(e)}")

# Register save function to run at shutdown
atexit.register(save_system_state)

# === System Initialization ===
@app.before_request
def initialize_system():
    """Initialize system state on first request"""
    global _is_first_request
    if _is_first_request:
        app.logger.info("Initializing system on first request...")
        load_or_create_faiss_index()
        _is_first_request = False

# === Routes ===
@app.route('/')
def index():
    docs_metadata = get_docs_metadata()
    return render_template('index.html', 
                         has_data=bool(vectorstore),
                         docs_metadata=docs_metadata)

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global vectorstore, qa_chain
    
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({"error": "File size exceeds maximum limit (10MB)"}), 400

    try:
        file_path = os.path.join("pdf", file.filename)
        file.save(file_path)
        
        if not is_valid_pdf(file_path):
            os.remove(file_path)
            return jsonify({"error": "Invalid or corrupted PDF file"}), 400

        # Process document in chunks to manage memory
        new_docs = load_and_split(file_path)
        
        # Update or create FAISS index
        if vectorstore is None:
            vectorstore = build_faiss_index(new_docs)
        else:
            vectorstore.add_documents(new_docs)
            vectorstore.save_local(FAISS_INDEX_PATH)
        
        # Save metadata
        save_docs_metadata(file.filename, len(new_docs))
        
        # Initialize QA chain only when needed
        qa_chain = None
        
        # Clean up
        del new_docs
        gc.collect()
        
        return jsonify({
            "message": f"Uploaded and indexed {file.filename}",
            "docs_metadata": get_docs_metadata()
        })
    except Exception as e:
        app.logger.error(f"Error processing PDF: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_chain
    
    data = request.get_json()
    query = data.get("question", "").strip()
    if not query:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Get current system state
        docs_metadata = get_docs_metadata()
        has_documents = bool(vectorstore)
        doc_count = len(docs_metadata)

        # For general questions, use conversation chain
        if is_general_query(query):
            response = init_conversation_chain().run(
                question=query,
                has_documents="oui" if has_documents else "non",
                doc_count=doc_count
            )
            return jsonify({"answer": response})

        # For document questions, initialize QA chain if needed
        if has_documents:
            if qa_chain is None:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
                qa_chain = RetrievalQA.from_chain_type(llm=init_llm(), retriever=retriever)
            
            try:
                qa_response = qa_chain.run(query)
                return jsonify({"answer": qa_response})
            except Exception as e:
                app.logger.error(f"QA chain error: {str(e)}")
                # Fallback to conversation chain
                response = init_conversation_chain().run(
                    question=query,
                    has_documents="oui",
                    doc_count=doc_count
                )
                return jsonify({"answer": response})
        else:
            response = init_conversation_chain().run(
                question=query,
                has_documents="non",
                doc_count=0
            )
            return jsonify({"answer": response})

    except Exception as e:
        app.logger.error(f"Error processing question: {str(e)}")
        return jsonify({"error": str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_valid_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if len(pdf_reader.pages) > 0:
                return True
            return False
    except Exception as e:
        app.logger.error(f"PDF validation error: {str(e)}")
        return False

# === Save and Load Functions ===
def save_docs_metadata(filename, chunk_count):
    """Save document metadata to JSON"""
    if os.path.exists(DOCS_JSON_PATH):
        with open(DOCS_JSON_PATH, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = []
    
    metadata.append({
        'filename': filename,
        'chunks': chunk_count,
        'date_added': datetime.now().isoformat(),
    })
    
    with open(DOCS_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def get_docs_metadata():
    """Get document metadata from JSON"""
    if os.path.exists(DOCS_JSON_PATH):
        with open(DOCS_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def is_general_query(query):
    """
    Détecte si une question est générale ou spécifique aux documents.
    """
    general_keywords = {
        'bonjour', 'hello', 'hi', 'salut', 'merci', 'thanks', 'comment', 'who are you',
        'que fais-tu', 'what do you do', 'help', 'aide', 'comment vas-tu', 'how are you'
    }
    
    query_lower = query.lower()
    
    for keyword in general_keywords:
        if keyword in query_lower:
            return True
            
    if len(query_lower.split()) <= 3:
        return True
        
    return False

@app.route("/status", methods=["GET"])
def get_status():
    docs_metadata = get_docs_metadata()
    return jsonify({
        "has_documents": bool(vectorstore),
        "document_count": len(docs_metadata),
        "docs_metadata": docs_metadata
    })

# === Run app ===
if __name__ == "__main__":
    app.run(debug=True)
