# import streamlit as st
# from data_loader import load_and_chunk
# from embed_and_store import build_vector_store
# from rag_retriever import query_rag

# # Initialize or load vector store (should be done once)
# csv_files = ["data/disease_precaution.csv",
#         "data/DiseaseAndSymptoms.csv"]  # Example paths
# chunks = load_and_chunk(csv_files)
# print("**")
# collection = build_vector_store(chunks)
# print("***")

# # Streamlit chat interface
# st.set_page_config(page_title="Disease QA Chatbot")
# st.title("Disease Q&A Assistant")

# if "history" not in st.session_state:
#     st.session_state.history = []

# # Display past messages
# for msg in st.session_state.history:
#     if msg["role"] == "user":
#         st.chat_message("user").write(msg["content"])
#     else:
#         st.chat_message("assistant").write(msg["content"])
#         # Show which passages were used (highlighted)
#         for passage in msg.get("sources", []):
#             st.markdown(f"<div style='background-color:#fff3b0; padding:4px'>{passage}</div>", unsafe_allow_html=True)

# # User input
# if prompt := st.chat_input("Ask a question about diseases..."):
#     st.session_state.history.append({"role": "user", "content": prompt})
    
#     # Get answer and source passages
#     answer, sources = query_rag(collection, prompt)
#     st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})

# import streamlit as st
# from langchain.vectorstores import Chroma
# from langchain.embeddings import OllamaEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.llms import Ollama

# persist_dir = "chroma_db"

# st.title("Healthcare RAG Chatbot")

# query = st.text_input("Ask about any disease:")

# if query:
#     # Load persisted vector store
#     embeddings = OllamaEmbeddings(model="mxbai-embed-large")
#     db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
#     retriever = db.as_retriever(search_kwargs={"k": 3})

#     # Setup LLM
#     llm = Ollama(model="llama3.1")  # for generating answers

#     qa = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         return_source_documents=True
#     )

#     result = qa(query)
#     st.write("### Answer")
#     st.write(result["result"])

#     st.write("### Source Documents")
#     for doc in result["source_documents"]:
#         st.write(doc.metadata, doc.page_content[:500])

# app.py
"""
Streamlit app for Healthcare RAG Chatbot

Features:
- Session chat history (persisted in st.session_state)
- Retrieval using Chroma + Ollama embeddings (via LangChain)
- Conversation-aware answers (uses recent chat history)
- Fancy UI: sidebar controls, chat message bubbles, source expanders
"""

# import streamlit as st
# from langchain.embeddings import OllamaEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.llms import Ollama
# from typing import List
# import time

# # ---------- Config ----------
# PERSIST_DIR = "chroma_db"         # where the Chroma DB was persisted
# EMBED_MODEL = "mxbai-embed-large" # or "nomic-embed-text"
# LLM_MODEL = "llama3.1"            # your locally available Llama model name
# K_RETRIEVE = 4                    # number of retrieved docs used as context
# MAX_HISTORY_MESSAGES = 6          # how many previous messages to include in prompt
# # ----------------------------

# def safe_rerun():
#     """
#     Try to call st.experimental_rerun() if available.
#     If not available in this Streamlit build, make a tiny session_state change
#     and stop execution to force Streamlit to re-run the script on next interaction.
#     This mimics a rerun in a robust, version-tolerant way.
#     """
#     try:
#         # preferred method (may not exist on some Streamlit builds)
#         st.experimental_rerun()
#     except Exception:
#         # fallback: make a small state change and stop execution.
#         # The page will reflect updated st.session_state on the next run.
#         st.session_state["_rerun_token"] = st.session_state.get("_rerun_token", 0) + 1
#         # Delay a tiny bit to avoid race conditions in some environments
#         time.sleep(0.01)
#         st.stop()

# st.set_page_config(page_title="Healthcare RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

# # Sidebar UI
# with st.sidebar:
#     st.title("Settings")
#     st.markdown("Configure models and session controls.")
#     embed_model_input = st.text_input("Embedding model", value=EMBED_MODEL)
#     llm_model_input = st.text_input("LLM model", value=LLM_MODEL)
#     k_input = st.number_input("Retriever k", min_value=1, max_value=10, value=K_RETRIEVE)
#     clear_btn = st.button("Clear chat")

# # Clear chat button handling
# if clear_btn:
#     st.session_state.clear()
#     safe_rerun()

# # Initialize session history
# if "history" not in st.session_state:
#     # history is a list of dict {role: "user"/"assistant", "content": str, "sources": [ {meta, text} ] }
#     st.session_state.history = []

# # Try loading the vectorstore (fail gracefully)
# try:
#     embeddings = OllamaEmbeddings(model=embed_model_input)
#     db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
#     retriever = db.as_retriever(search_kwargs={"k": k_input})
#     retriever_search = lambda q, k=k_input: db.similarity_search(q, k=k)
# except Exception as e:
#     st.sidebar.error("Could not open Chroma DB. Run build_embeddings.py first.")
#     st.sidebar.exception(e)
#     st.stop()

# # Create the LLM
# try:
#     llm = Ollama(model=llm_model_input)
# except Exception as e:
#     st.sidebar.error("Could not initialize Ollama LLM. Check Ollama installation and models.")
#     st.sidebar.exception(e)
#     st.stop()

# # Page layout
# col1, col2 = st.columns([3, 1])

# with col1:
#     st.title("Healthcare RAG Chatbot")
#     st.markdown("Ask about diseases, symptoms, prevention. Answers use your CSV data and show the supporting passages.")

#     # Display chat history
#     def render_chat():
#         for i, msg in enumerate(st.session_state.history):
#             role = msg["role"]
#             content = msg["content"]
#             if role == "user":
#                 with st.chat_message("user"):
#                     st.write(content)
#             else:
#                 # assistant message
#                 with st.chat_message("assistant"):
#                     st.write(content)
#                     # show source passages as expanders
#                     sources = msg.get("sources", [])
#                     if sources:
#                         for s in sources:
#                             meta = s.get("metadata", {})
#                             text = s.get("text", "")
#                             with st.expander(f"Source — {meta.get('row_index','?')} / chunk {meta.get('chunk_id','?')}", expanded=False):
#                                 st.write(text)

#     # Render current chat
#     render_chat()

#     # Input box
#     prompt = st.chat_input("Type your question here...")

# with col2:
#     st.markdown("### Session controls")
#     st.write("Messages in this session:")
#     st.write(len(st.session_state.history))
#     st.markdown("---")
#     st.markdown("### Prompt & System")
#     system_prompt = st.text_area("System prompt (guides assistant)", value="You are a helpful, concise healthcare assistant. Use only the provided context passages and the chat history to answer the user. Do not hallucinate facts.")
#     st.markdown("---")
#     st.markdown("### Quick actions")
#     if st.button("Regenerate last answer") and st.session_state.history:
#         # remove last assistant message so we can regenerate
#         # If last message is user, do nothing
#         if st.session_state.history[-1]["role"] == "assistant":
#             st.session_state.history.pop(-1)
#             safe_rerun()

# # Handle user input
# if prompt:
#     # add user message to history
#     st.session_state.history.append({"role": "user", "content": prompt})

#     # build a compact chat history string of the last N messages for context
#     def build_chat_history_text(history: List[dict], max_messages: int = MAX_HISTORY_MESSAGES) -> str:
#         # take last max_messages messages (user+assistant pairs)
#         tail = history[-max_messages:]
#         lines = []
#         for m in tail:
#             prefix = "User" if m["role"] == "user" else "Assistant"
#             # keep message short in prompt
#             lines.append(f"{prefix}: {m['content']}")
#         return "\n".join(lines)

#     chat_history_text = build_chat_history_text(st.session_state.history, MAX_HISTORY_MESSAGES)

#     # Retrieve relevant documents
#     with st.spinner("Retrieving relevant passages..."):
#         docs = retriever_search(prompt, k=k_input)

#     # Build context from retrieved docs
#     context_blocks = []
#     sources_for_ui = []
#     for d in docs:
#         # Document objects from Chroma (LangChain Document)
#         txt = d.page_content if hasattr(d, "page_content") else getattr(d, "content", str(d))
#         meta = d.metadata if hasattr(d, "metadata") else {}
#         context_blocks.append(f"Source (meta: {meta}):\n{txt}")
#         sources_for_ui.append({"metadata": meta, "text": txt})

#     context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context available."

#     # Construct final prompt (system + chat history + context + question)
#     final_prompt = (
#         f"{system_prompt}\n\n"
#         f"Conversation history:\n{chat_history_text}\n\n"
#         f"Context passages (use only these to answer):\n{context}\n\n"
#         f"Question: {prompt}\n\n"
#         f"Answer concisely and list which source chunks you used (by metadata)."
#     )

#     # Generate answer
#     with st.spinner("Generating answer..."):
#         try:
#             answer = llm(final_prompt)  # LangChain Ollama llm call returns string
#         except Exception as e:
#             st.error("LLM generation failed. See sidebar for error.")
#             st.sidebar.exception(e)
#             answer = "Sorry — generation failed. Check Ollama model availability."

#     # Append assistant message (with sources) to history
#     st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources_for_ui})


