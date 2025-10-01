def split_text(text, chunk_size=800, chunk_overlap=100):
    """
    Split text into chunks with overlap.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap  # overlap
        if start < 0:
            start = 0
    return chunks