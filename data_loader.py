import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk(csv_paths, chunk_size=800, chunk_overlap=100):
    """
    Load multiple CSVs, concatenate them, and split text into chunks.
    Returns a list of dicts with 'content' and 'metadata'.
    Works for CSVs with arbitrary columns.
    """
    dfs = [pd.read_csv(path) for path in csv_paths]
    df = pd.concat(dfs, ignore_index=True)

    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    for idx, row in df.iterrows():
        # Combine all columns into one text
        text = "\n".join([f"{col}: {row[col]}" for col in df.columns if col in row and pd.notna(row[col])])
        for i, chunk in enumerate(splitter.split_text(text)):
            chunks.append({
                "content": chunk,
                "metadata": {"row_index": idx, "chunk_id": i}
            })
    return chunks