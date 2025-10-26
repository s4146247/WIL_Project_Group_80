Patient-facing Healthcare RAG Chatbot (Streamlit)
- Simple, friendly UI for patients
- Preserves session chat history and uses chat history in answers
- Retrieval from ChromaDB using Ollama embeddings (via LangChain)
- Uses Ollama LLM for generation (locally via Ollama)
- Sources shown as expandable items for transparency
- "Advanced" admin controls live in the sidebar (collapsed by default)

Usage:
  1. Ensure your environment has ollama, langchain, chromadb, streamlit, pandas, pypdf.
  2. Build embeddings: python build_embeddings.py (creates ./chroma_db)
  3. Run: python -m streamlit run app.py

Generate Evaluation Report:
  1. Run: python eval_framework.py --pdf_dir data