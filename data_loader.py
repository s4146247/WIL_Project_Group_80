import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

def load_and_chunk(pdf_paths, chunk_size=800, chunk_overlap=100):
    """
    Load multiple PDFs and split text into chunks.
    Returns a list of dicts with 'content' and 'metadata'.
    """
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for path in pdf_paths:
        # Load PDF pages (PyPDFLoader returns Document objects per page)
        loader = PyPDFLoader(path)
        pages = loader.load()  # list of Document objects (one per page)

        for page_num, doc in enumerate(pages, start=1):
            text = doc.page_content or ""
            if not text.strip():
                continue
            # split the single page text into overlapping chunks
            for i, chunk in enumerate(splitter.split_text(text)):
                chunks.append({
                    "content": chunk,
                    "metadata": {
                        "source": os.path.basename(path),
                        "source_path": os.path.abspath(path),
                        "page": page_num,
                        "chunk_id": i
                    }
                })
    return chunks
