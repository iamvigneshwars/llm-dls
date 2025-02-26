import os
import sys
import argparse
from typing import List, Dict, Any
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS 
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def init_rag_pipeline(pdf_path: str, model_name: str = "llama3:8b"):
    """Initialize the RAG pipeline with the given PDF and model."""
    
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=1000,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks of text")
    
    print("Creating embeddings and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    print(f"Initializing Ollama with model {model_name}...")
    llm = Ollama(model=model_name)
    
    template = """
    You are a helpful assistant that answers questions based on the provided context from a PDF guide about the user office.
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    - Answer the question based only on the provided context
    - If the answer is not in the context, say "I don't have information about that"
    - Give detailed answers for the question based on the provided context from the PDF guide
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


def main():
    parser = argparse.ArgumentParser(description="PDF RAG Chatbot (Terminal Version)")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--model", default="llama3:8b", help="Ollama model name (default: llama3:8b)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf):
        print(f"Error: PDF file not found at {args.pdf}")
        sys.exit(1)
    
    qa_chain = init_rag_pipeline(args.pdf, args.model)
    
    print("\n" + "="*50)
    print("PDF RAG Chatbot initialized. Ask questions about your document.")
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
            result = qa_chain.invoke({"query": query})
            
            print("\n" + "-"*50)
            print("ANSWER:")
            print(result["result"])
            print("-"*50)
            
            print("\nSOURCE DOCUMENTS:")
            for i, doc in enumerate(result.get("source_documents", []), 1):
                page_info = f" (Page {doc.metadata.get('page', 'unknown')})" if doc.metadata.get('page') else ""
                print(f"{i}. Document excerpt{page_info}:")
                print(f"   {doc.page_content[:150]}..." if len(doc.page_content) > 150 else f"   {doc.page_content}")
                print()
            
    except KeyboardInterrupt:
        print("\n\nExiting chatbot. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    main()
