import ollama

def query_rag(collection, query, k=3):
    """
    Perform retrieval of top-k passages and generate an answer with Llama3.1.
    Returns the answer text and the list of retrieved passages.
    """
    # 1) Embed the query text
    query_resp = ollama.embed(model="nomic-embed-text", input=query)
    query_vector = query_resp["embeddings"]
    
    # 2) Retrieve top-k documents from Chroma
    result = collection.query(query_embeddings=[query_vector], n_results=k)
    passages = result["documents"][0]  # list of retrieved chunks
    
    # 3) Build prompt with context and question
    context = "\n\n---\n\n".join(passages)
    prompt = (
        f"Answer the question using only the following context:\n{context}\n\n"
        f"Question: {query}"
    )
    
    # 4) Generate answer with Llama3.1 via Ollama
    response = ollama.generate(model="llama3.1", prompt=prompt)
    answer_text = response.get("response", "").strip()
    
    return answer_text, passages