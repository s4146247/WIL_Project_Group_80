from data_loader import load_and_chunk
from embed_and_store import build_vector_store

def main():
    csv_files = ["data/disease_precaution.csv", "data/DiseaseAndSymptoms.csv"]  
    chunks = load_and_chunk(csv_files)
    print(f"✅ Loaded and chunked {len(chunks)} text segments")

    db = build_vector_store(chunks)
    print("✅ Vector store created and persisted!")

if __name__ == "__main__":
    main()