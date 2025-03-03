import os
import sys
import argparse
import time
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS 
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def init_rag_pipeline(pdf_path: str, model_name: str = "llama3:8b", 
                      chunk_size: int = 8000, chunk_overlap: int = 1600, 
                      retriever_k: int = 10):
    """Initialize the RAG pipeline with the given PDF and model."""
    
    print(f"Loading PDF from {pdf_path}...")
    start_time = time.time()
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"PDF loaded in {time.time() - start_time:.2f} seconds")
    
    print("Splitting text into chunks...")
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks of text in {time.time() - start_time:.2f} seconds")
    
    enhanced_chunks = []
    for i, chunk in enumerate(chunks):
        if 'page' in chunk.metadata:
            page = chunk.metadata['page']
            related_chunks = [c for c in chunks if 'page' in c.metadata and 
                              abs(c.metadata['page'] - page) <= 1 and c != chunk]
            
            if related_chunks:
                related_content = "\n\n".join([c.page_content[:150] + "..." for c in related_chunks[:2]])
                chunk.page_content = f"{chunk.page_content}\n\nRelated content: {related_content}"
        
        enhanced_chunks.append(chunk)
    
    print("Creating embeddings and vector store...")
    start_time = time.time()
    
    num_workers = min(multiprocessing.cpu_count(), 4)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
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
    
    print(f"Vector store created in {time.time() - start_time:.2f} seconds")
    
    print(f"Initializing Ollama with model {model_name}...")
    llm = Ollama(
        model=model_name,
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.5, 
        num_ctx=8192,
        repeat_penalty=1.1
    )
    
    template = """
    You are an expert assistant tasked with answering questions about a document. Use the following context to provide a comprehensive answer. 
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    - Answer thoroughly using all relevant information from the context
    - Include specific details from the document where appropriate
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
    
    print("RAG pipeline initialized successfully!")
    return qa_chain, retriever


def main():
    parser = argparse.ArgumentParser(description="Enhanced PDF RAG Chatbot")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--model", default="llama3:8b", help="Ollama model name (default: llama3:8b)")
    parser.add_argument("--chunk-size", type=int, default=3000, help="Text chunk size (default: 3000)")
    parser.add_argument("--chunk-overlap", type=int, default=600, help="Text chunk overlap (default: 600)")
    parser.add_argument("--retriever-k", type=int, default=8, help="Number of chunks to retrieve (default: 8)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file not found at {args.pdf}")
        sys.exit(1)
    
    qa_chain, retriever = init_rag_pipeline(
        args.pdf, 
        args.model,
        args.chunk_size,
        args.chunk_overlap,
        args.retriever_k
    )
    
    print("\n" + "="*50)
    print("Enhanced PDF RAG Chatbot initialized. Ask questions about your document.")
    print("Type 'exit', 'quit', or press Ctrl+C to end the session.")
    print("="*50 + "\n")
    
    try:
        while True:
            query = input("\nAsk a question: ")
            
            if query.lower() in ["exit", "quit", "q"]:
                print("\nExiting chatbot. Goodbye!")
                break
            
            if not query.strip():
                continue
            
            print("\nSearching and generating answer...")
            
            # Start total time measurement
            total_start_time = time.time()
            
            # Measure retrieval time
            retrieval_start = time.time()
            retrieved_docs = retriever.get_relevant_documents(query)
            retrieval_time = time.time() - retrieval_start
            print(f"\nRetrieval time: {retrieval_time:.2f} seconds")
            
            # Measure LLM inference time
            print("\nResponse: ")
            inference_start = time.time()
            result = qa_chain({"query": query})
            inference_time = time.time() - inference_start
            
            # Calculate total time
            total_time = time.time() - total_start_time
            
            # Calculate overhead
            overhead = total_time - retrieval_time - inference_time
            
            print(f"\nTiming Summary:")
            print(f"  - Total time: {total_time:.2f} seconds")
            print(f"  - Retrieval time: {retrieval_time:.2f} seconds")
            print(f"  - LLM inference time: {inference_time:.2f} seconds")
            print(f"  - Overhead: {overhead:.2f} seconds")
            
            # print("\nSOURCE DOCUMENTS:")
            # for i, doc in enumerate(result.get("source_documents", []), 1):
            #     page_info = f" (Page {doc.metadata.get('page', 'unknown')})" if doc.metadata.get('page') else ""
            #     print(f"{i}. Document excerpt{page_info}:")
            #     print(f"   {doc.page_content[:150]}..." if len(doc.page_content) > 150 else f"   {doc.page_content}")
            #     print()
            
    except KeyboardInterrupt:
        print("\n\nExiting chatbot. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    main()

