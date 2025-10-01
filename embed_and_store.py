from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

def build_vector_store(chunks, model_name="mxbai-embed-large", persist_dir="chroma_db"):
    """
    Build a Chroma vector store from text chunks using Ollama embeddings via LangChain.
    """
    # Extract texts and metadata
    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Initialize Ollama embeddings
    embeddings = OllamaEmbeddings(model=model_name)

    # Build vector store in Chroma
    db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=persist_dir
    )
    db.persist()
    return db