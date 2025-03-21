import os
import sys
import argparse
import time
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import fitz
from datetime import datetime

def extract_text_and_links(pdf_path: str):
    """Extract text and hyperlinks from a PDF file."""
    documents = []
    
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text()
        links = []
        for link in page.get_links():
            if "uri" in link:
                rect = fitz.Rect(link["from"])
                words = page.get_text("words", clip=rect)
                link_text = " ".join([w[4] for w in words]) if words else "link"
                
                links.append({
                    "text": link_text,
                    "url": link["uri"],
                    "rect": [link["from"].x0, link["from"].y0, link["from"].x1, link["from"].y1]
                })
        
        if links:
            link_section = "\n\nHyperlinks on this page:\n"
            for link in links:
                link_section += f"- {link['text']}: {link['url']}\n"
            text += link_section
        
        documents.append(Document(
            page_content=text,
            metadata={
                "page": page_num,
                "source": pdf_path,
                "links": links
            }
        ))
    
    return documents

def init_rag_pipeline(
    pdf_path: str,
    model_name: str, 
    chunk_size: int,
    chunk_overlap: int, 
    retriever_k:int,
    debug: bool,
):

    if debug:
        print(f"Loading PDF from {pdf_path}...")
    start_time = time.time()
    documents = extract_text_and_links(pdf_path)
    if debug:
        print(f"PDF loaded in {time.time() - start_time:.2f} seconds")
        print("Splitting text into chunks...")
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    if debug:
        print(f"Created {len(chunks)} chunks of text in {time.time() - start_time:.2f} seconds")
    
    enhanced_chunks = []
    for _, chunk in enumerate(chunks):
        if 'page' in chunk.metadata:
            page = chunk.metadata['page']
            related_chunks = [c for c in chunks if 'page' in c.metadata and 
                              abs(c.metadata['page'] - page) <= 1 and c != chunk]
            
            if related_chunks:
                related_content = "\n\n".join([c.page_content[:150] + "..." for c in related_chunks[:2]])
                chunk.page_content = f"{chunk.page_content}\n\nRelated content: {related_content}"
        
        enhanced_chunks.append(chunk)
    if debug:
        print("Creating embeddings and vector store...")
    start_time = time.time()
    
    num_workers = min(multiprocessing.cpu_count(), 6)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
    )
    
    def batch_embed(docs_batch):
        return embeddings.embed_documents([d.page_content for d in docs_batch])
    
    batch_size = min(32, len(enhanced_chunks))
    batches = [enhanced_chunks[i:i + batch_size] for i in range(0, len(enhanced_chunks), batch_size)]
    
    all_embeddings = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for embs in executor.map(batch_embed, batches):
            all_embeddings.extend(embs)
    
    vectorstore = FAISS.from_documents(
        enhanced_chunks,
        embeddings
    )
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": retriever_k,
            "fetch_k": retriever_k * 2,
            "lambda_mult": 0.8
        }
    )
    
    if debug:
        print(f"Vector store created in {time.time() - start_time:.2f} seconds")
        print(f"Initializing Ollama with model {model_name}...")
    llm = OllamaLLM(
        model=model_name,
        temperature=0.6, 
        num_ctx=8192,
        repeat_penalty=1.2,
        keep_alive=-1
    )
    
    template = """
    You are an expert assistant tasked with answering questions about a document. Use the following context to provide a comprehensive answer. 
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    - Answer thoroughly using all relevant information from the context
    - Include specific details from the document where appropriate
    - Preserve any URLs or hyperlinks that appear in the context by including them in your answer
    - If the context contains links, format them properly as [text](url) in your response
    - If information is not in the context, say "I don't have information about that"
    - Never mention the document or context in your answer
    - Structure your answer with clear paragraphs
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    if debug:
        print("RAG pipeline initialized successfully!")
    return qa_chain, retriever

def log(question, answer, model_name, processing_time):
    log_file = "/dls/science/users/mrg27357/llm_chat_log.json"
    timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "model": model_name,
        "processing_time": processing_time
    }
    
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = []
        else:
            log_data = []
        
        log_data.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
            
    except Exception as e:
        print(f"Error logging response: {str(e)}")


def create_app(qa_chain, model_name):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/status', methods=['GET'])
    def status():
        """Handle GET requests to /status endpoint."""
        return jsonify({
            "status": "online",
            "api_version": "1.0",
            "model": model_name
        })
    
    @app.route('/ask', methods=['POST'])
    def ask():
        """Handle POST requests to /ask endpoint."""
        data = request.json
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' in request body"}), 400
        
        question = data['question']
        
        try:
            start_time = time.time()
            result = qa_chain.invoke({"query": question})
            total_time = time.time() - start_time
            
            answer = result["result"]
            response = {
                "answer": answer, 
                "metadata": {
                    "model": model_name,
                    "processing_time": f"{total_time:.2f}",
                }
            }
            
            log(question, answer, model_name, total_time)
            return jsonify(response)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "Online", "model": model_name}), 200
    
    return app

def main():
    parser = argparse.ArgumentParser(description="Diamond RAG API")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--model", default="phi4:14b", help="Ollama model name (default: phi4:14b)")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Text chunk size (default: 3000)")
    parser.add_argument("--chunk-overlap", type=int, default=600, help="Text chunk overlap (default: 600)")
    parser.add_argument("--retriever-k", type=int, default=8, help="Number of chunks to retrieve (default: 8)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the API on (default: 5000)")
    args = parser.parse_args()
    debug = args.debug
    
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file not found at {args.pdf}")
        sys.exit(1)
    
    qa_chain, _ = init_rag_pipeline(
        args.pdf, 
        args.model,
        args.chunk_size,
        args.chunk_overlap,
        args.retriever_k,
        args.debug
    )
    
    app = create_app(qa_chain, args.model)
    
    if debug:
        print(f"Starting Diamond RAG API on http://0.0.0.0:{args.port}")
    app.run(host='0.0.0.0', port=args.port, threaded=True)

if __name__ == "__main__":
    main()
