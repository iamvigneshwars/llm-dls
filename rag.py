from cmd import PROMPT
from pydoc import doc
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores.chroma import Chroma 
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import re
import time

PROMPT_TEMPLATE = """
    You are a helpful assistant that answers questions based on the provided context from a PDF guide about the user office.
    
    Context:
    {context}
    
    Question: {question}
    
    Instructions:
    - Answer the question based only on the provided context
    - If the answer is not in the context, say "I don't have information about that"
    - Provide detailed answers for the question based on the provided context from the PDF guide
    = Do not mention about your context
    - Do not make up or hallucinate information
    
    Answer:
"""

def load_documents():
    document_loader = PyPDFDirectoryLoader("data")
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory="chroma", embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

documents = load_documents()
chunks = split_documents(documents)
add_to_chroma(chunks)

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory="chroma", embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama3.1:70b")

    response_text = model.invoke(prompt)
    response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    sources = [doc.metadata.get("id", None) for doc, _score in results]

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    return response_text


def main():
    try:
        while True:
            query = input("\nAsk a question: ")
            
            if query.lower() in ["exit", "quit", "q"]:
                print("\nExiting chatbot. Goodbye!")
                break
            
            if not query.strip():
                continue
            
            print("\nSearching and generating answer...")
            start_time = time.time()
            query_rag(query)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\n########### Time taken to generate response: {elapsed_time:.2f} seconds ###########")

    except KeyboardInterrupt:
        print("\n\nExiting chatbot. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()
