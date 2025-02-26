import os
import sys
import argparse
from typing import List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import PyPDF2
from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import FAISS 
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[str] = []


app = FastAPI(title="PDF RAG Chatbot")

pdf_path = None
retrieval_qa = None


def init_rag_pipeline(pdf_path: str, model_name: str = "llama3:8b"):
    """Initialize the RAG pipeline with the given PDF and model."""
    
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks of text")
    
    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    print(f"Initializing Ollama with model {model_name}...")
    llm = Ollama(model=model_name)
    
    template = """
    You are a helpful assistant that answers questions based on the provided context from a PDF guide about the user office.
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    - Answer the question based only on the provided context
    - If the answer is not in the context, say "I don't have information about that in the guide"
    - Keep your answers concise and to the point
    - Do not make up or hallucinate information
    
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
    
    print("RAG pipeline initialized successfully!")
    return qa_chain


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Endpoint to query the RAG system with a question about the PDF."""
    
    if retrieval_qa is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized. Start the server with a valid PDF path.")
    
    try:
        result = retrieval_qa.invoke({"query": request.query})
        source_docs = []
        for doc in result.get("source_documents", []):
            page_info = f" (Page {doc.metadata.get('page', 'unknown')})" if doc.metadata.get('page') else ""
            source_docs.append(f"Document excerpt{page_info}: {doc.page_content[:150]}...")
        
        return QueryResponse(
            answer=result["result"],
            source_documents=source_docs
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="PDF RAG Chatbot API Server")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--model", default="llama3:8b", help="Ollama model name (default: llama3:8b)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file not found at {args.pdf}")
        sys.exit(1)
    
    global retrieval_qa
    retrieval_qa = init_rag_pipeline(args.pdf, args.model)
    
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
