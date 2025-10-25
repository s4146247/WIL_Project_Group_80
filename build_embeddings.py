import glob
from data_loader import load_and_chunk
from embed_and_store import build_vector_store

def main():
    # find all PDFs in the data folder
    pdf_files = sorted(glob.glob("data/*.pdf"))
    if not pdf_files:
        print("No PDF files found in data/. Put your .pdf files into the data folder.")
        return

    chunks = load_and_chunk(pdf_files)
    print(f"Loaded and chunked {len(chunks)} text segments from PDFs")

    db = build_vector_store(chunks)
    print("Vector store created and persisted!")

if __name__ == "__main__":
    main()
