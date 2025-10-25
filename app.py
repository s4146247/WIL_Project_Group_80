import streamlit as st
from typing import List
import logging
import traceback

# LangChain / Ollama imports
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import ollama

# ---------------- CONFIG ----------------
PERSIST_DIR = "chroma_db"
DEFAULT_EMBED_MODEL = "mxbai-embed-large"
DEFAULT_LLM_MODEL = "llama3.1"
DEFAULT_K = 2
MAX_HISTORY_MESSAGES = 6  # how many previous messages to include
MAX_PROMPT_CHARS = 1200
# ----------------------------------------

st.set_page_config(page_title="Health Assistant", layout="wide", initial_sidebar_state="expanded")

# ---- helper functions ----

def load_vectorstore(persist_dir: str, embed_model: str, collection_name: str = "pdf_collection"):
    """
    Try to load a persisted Chroma vectorstore using OllamaEmbeddings via LangChain.
    Returns db or raises an exception with a helpful message.
    """
    try:
        embeddings = OllamaEmbeddings(model=embed_model)
        # NOTE: use the same collection_name used when creating the DB
        db = Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name=collection_name)
        return db
    except TypeError:
        # Some LangChain versions expect `embedding` instead of `embedding_function`
        try:
            db = Chroma(persist_directory=persist_dir, embedding=embeddings, collection_name=collection_name)
            return db
        except Exception as e:
            raise RuntimeError(f"Could not open Chroma DB (second attempt failed): {e}")
    except Exception as e:
        raise RuntimeError(f"Could not open Chroma DB or initialize embeddings: {e}")



def build_chat_history_text(history: List[dict], max_messages: int = MAX_HISTORY_MESSAGES) -> str:
    """
    Convert list of messages in st.session_state.history into a compact text block
    (keeps only the last max_messages entries).
    """
    tail = history[-max_messages:]
    lines = []
    for m in tail:
        prefix = "Patient" if m["role"] == "user" else "Assistant"
        # keep message short in prompt
        lines.append(f"{prefix}: {m['content']}")
    return "\n".join(lines)


def safe_llm_call(llm, prompt: str, fallback_model_name: str = None, retry_truncate_chars: int = 1000) -> str:
    """
    Robust LLM caller with strict prompt-size and fallback token cap:
    - Ensures prompt length <= MAX_PROMPT_CHARS
    - Tries LangChain LLM first
    - On failure falls back to direct ollama.generate(..., stream=False, num_predict=...)
    - If model-runner crash occurs, retries once with truncated prompt.
    """
    # Enforce an absolute max prompt size (prevent OOM due to huge prompts)
    try:
        if len(prompt) > MAX_PROMPT_CHARS:
            logging.info(f"Prompt length {len(prompt)} > MAX_PROMPT_CHARS ({MAX_PROMPT_CHARS}) — truncating before LLM call.")
            prompt = prompt[:MAX_PROMPT_CHARS]
    except Exception:
        # safe-guard; continue if len() fails for any reason
        pass

    # Try LangChain LLM wrapper first
    try:
        if hasattr(llm, "predict"):
            return llm.predict(prompt)
        result = llm(prompt)
        if isinstance(result, str):
            return result
        if hasattr(result, "generations"):
            gens = result.generations
            if isinstance(gens, list) and len(gens) > 0 and len(gens[0]) > 0:
                maybe_text = getattr(gens[0][0], "text", None) or str(gens[0][0])
                return maybe_text
        return str(result)
    except Exception as e_lang:
        logging.exception("LangChain LLM call failed; attempting direct ollama fallback")

        # Prepare model name for fallback
        model_name = fallback_model_name or getattr(llm, "model", None) or getattr(llm, "model_name", None)
        if not model_name:
            logging.error("No model name available for ollama fallback.")
            raise RuntimeError(f"LLM generation failed (langchain attempt): {e_lang}\nNo fallback model name available.")

        # Direct ollama generate (non-streaming) with explicit token cap
        try:
            logging.info(f"Falling back to ollama.generate(model={model_name}, stream=False)")
            gen = ollama.generate(model=model_name, prompt=prompt, stream=False)
            if isinstance(gen, dict):
                # commonly 'response' or 'text'
                txt = gen.get("response") or gen.get("text") or str(gen)
                return str(txt).strip()
            return str(gen).strip()
        except Exception as e_ollama:
            logging.exception("Direct ollama.generate fallback failed")

            # If runner crash/internal error, retry once with a shorter prompt
            err_text = str(e_ollama).lower()
            if ("model runner has unexpectedly stopped" in err_text
                    or "status code 500" in err_text
                    or "runner crashed" in err_text
                    or "internal error" in err_text):
                logging.info("Detected model-runner crash in fallback; retrying once with truncated prompt via ollama.generate")
                try:
                    short_prompt = prompt[:retry_truncate_chars]
                    gen2 = ollama.generate(model=model_name, prompt=short_prompt, stream=False)
                    if isinstance(gen2, dict):
                        txt2 = gen2.get("response") or gen2.get("text") or str(gen2)
                        return str(txt2).strip()
                    return str(gen2).strip()
                except Exception as e_retry:
                    logging.exception("Retry after truncation with direct ollama.generate also failed")
                    # raise a combined informative error
                    raise RuntimeError(
                        f"LLM generation failed (langchain attempt): {e_lang}\n"
                        f"ollama fallback failed: {e_ollama}\n"
                        f"retry after truncation failed: {e_retry}\n{traceback.format_exc()}"
                    )
            # otherwise re-raise combined error
            raise RuntimeError(f"LLM generation failed (langchain attempt): {e_lang}\nollama fallback failed: {e_ollama}\n{traceback.format_exc()}")

    
def truncate_context_by_chars(context: str, max_chars: int = 1200) -> str:
    """
    Keep as many full context blocks (separated by the delimiter) as will fit
    into max_chars. If none fit, return the first max_chars characters.
    """
    delimiter = "\n\n---\n\n"
    if len(context) <= max_chars:
        return context
    parts = context.split(delimiter)
    out_parts = []
    cur_len = 0
    for p in parts:
        add_len = len(p) + len(delimiter)
        if cur_len + add_len > max_chars:
            break
        out_parts.append(p)
        cur_len += add_len
    if out_parts:
        return delimiter.join(out_parts)
    # nothing fits as a full block, fall back to hard truncation
    return context[:max_chars]



# ---- Sidebar ----

with st.sidebar:
    st.markdown("### Settings (Advanced)")
    advanced_mode = st.checkbox("Show advanced options", value=False)
    if advanced_mode:
        embed_model_input = st.text_input("Embedding model (Ollama)", value=DEFAULT_EMBED_MODEL)
        llm_model_input = st.text_input("LLM model (Ollama)", value=DEFAULT_LLM_MODEL)
        k_input = st.number_input("Number of retrieved passages (k)", min_value=1, max_value=10, value=DEFAULT_K)
        st.markdown("---")
        st.markdown("Vector store")
        st.write(f"Persistent folder: `{PERSIST_DIR}`")
        if st.button("Reload vector store"):
            # attempt to reload vector store, we will handle errors later in main flow
            st.session_state["_reload_vectorstore"] = st.session_state.get("_reload_vectorstore", 0) + 1
    else:
        # keep defaults hidden
        embed_model_input = DEFAULT_EMBED_MODEL
        llm_model_input = DEFAULT_LLM_MODEL
        k_input = DEFAULT_K

    st.markdown("---")
    st.markdown("About this assistant")
    st.write(
        "This assistant is for **informational** purposes only. It is not a substitute for professional medical advice, "
        "diagnosis, or treatment. If you have an emergency, call your local emergency number or go to the nearest hospital."
    )

# ---- Main UI (Patient-facing) ----

# Top header
st.markdown("<h1 style='font-size:32px; margin:0'>Health Assistant</h1>", unsafe_allow_html=True)
st.markdown("A simple assistant to answer questions based on the medical data provided. "
            "Short, friendly answers. If in doubt, always consult a healthcare professional.")

# Visible notice + emergency instructions
st.info(
    "If you are experiencing a medical emergency (severe chest pain, trouble breathing, sudden weakness, severe bleeding), "
    "call your local emergency number **right now** or go to the nearest emergency department."
)

# Provide example queries to help non-technical users
with st.expander("Example questions (click to view)"):
    st.write(
        "- What are the typical symptoms of influenza (flu)?\n"
        "- How can I reduce the risk of catching a cold?\n"
        "- What should I do if I have a rash after taking a medicine?\n"
        "- What are common precautions for [disease name]?"
    )

# Create two-column layout: left is chat, right is quick tips / actions
col_chat, col_side = st.columns([3, 1])

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"/"assistant", "content": str, "sources": [ ... ]}

if "vectorstore_loaded" not in st.session_state:
    st.session_state.vectorstore_loaded = False
    st.session_state.db = None

# Try to load vector store (non-blocking UI)
try:
    # If user requested reload via sidebar, or not loaded yet, reload now
    if (not st.session_state.vectorstore_loaded) or st.session_state.get("_reload_vectorstore", 0) > 0:
        try:
            db = load_vectorstore(PERSIST_DIR, embed_model_input, collection_name="pdf_collection")
            st.session_state.db = db
            st.session_state.vectorstore_loaded = True
            st.session_state["_reload_vectorstore"] = 0
        except Exception as e:
            # keep vectorstore_loaded False and store the error in session for later display
            st.session_state.vectorstore_loaded = False
            st.session_state._load_error = str(e)
except Exception as e:
    st.session_state.vectorstore_loaded = False
    st.session_state._load_error = str(e)

# Right column: simple tips for patients
with col_side:
    st.markdown("### Quick tips")
    st.write("• If symptoms are mild, rest, stay hydrated, and monitor symptoms.")
    st.write("• For allergic reactions, stop the suspected medicine and seek help if breathing or swelling occurs.")
    st.write("• Preventive measures (handwashing, vaccination) are often effective.")
    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.history = []

    st.markdown("### Privacy")
    st.write("This chat stores conversation only in your browser session. No personal data is sent anywhere else by this app itself.")

# Chat container where we render messages (so we can re-render after update)
chat_container = col_chat.container()

def render_chat_into(container):
    """Render messages into the given container (used to re-render after appending messages)."""
    with container:
        # Render each message
        for msg in st.session_state.history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                with st.chat_message("user"):
                    st.markdown(f"**You:** {content}")
            else:
                # assistant message
                with st.chat_message("assistant"):
                    st.markdown(f"**Assistant:** {content}")
                    sources = msg.get("sources", [])
                    if sources:
                        for s in sources:
                            meta = s.get("metadata", {})
                            text = s.get("text", "")
                            with st.expander(f"Source — chunk {meta.get('chunk_id','?')}", expanded=False):
                                st.write(text)

# initial render
render_chat_into(chat_container)

# Input UI (large friendly box)
with col_chat:
    if st.session_state.get("_clear_user_input"):
        # remove the existing widget-backed state so new widget starts empty
        st.session_state.pop("user_input", None)
        st.session_state["_clear_user_input"] = False

    st.markdown("**Ask a question about a disease, symptom, or prevention.**")
    user_input = st.text_area("", placeholder="e.g., What are the symptoms of measles?", height=120, key="user_input")
    submit = st.button("Ask", key="ask_button")

# Main logic: handle submission
if submit and (user_input and user_input.strip()):
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": user_input.strip()})

    # If vector store not loaded, tell patient gently and stop
    if not st.session_state.vectorstore_loaded:
        err_msg = (
            "Sorry — the knowledge base is not available right now. "
            "Please try again later or contact the system administrator."
        )
        # Also show admin-friendly error in sidebar if advanced mode
        if advanced_mode:
            st.sidebar.error("Vector store load error:")
            st.sidebar.write(st.session_state.get("_load_error", "Unknown error loading vector store."))
        st.session_state.history.append({"role": "assistant", "content": err_msg, "sources": []})
        # re-render chat so the user sees the response immediately
        render_chat_into(chat_container)
    else:
        # Proceed with retrieval + generation
        try:
            db = st.session_state.db

            # 1) Retrieval (explicit debug)
            try:
                docs = db.similarity_search(user_input, k=k_input)
            except Exception as e_retr:
                # fallback to retriever API
                try:
                    retriever = db.as_retriever(search_kwargs={"k": k_input})
                    docs = retriever.get_relevant_documents(user_input)
                except Exception as e2:
                    # Surface retrieval error for debugging
                    if advanced_mode:
                        st.sidebar.error("Retrieval error (details):")
                        st.sidebar.write(traceback.format_exc())
                    raise RuntimeError(f"Retrieval failed: {e_retr} | fallback failed: {e2}")

            # If no docs found, inform the user rather than continuing to LLM
            if not docs:
                no_docs_msg = "I couldn't find any relevant passages in the knowledge base for that question."
                st.session_state.history.append({"role": "assistant", "content": no_docs_msg, "sources": []})
                render_chat_into(chat_container)
                st.session_state["_clear_user_input"] = True
            else:
                # Build context text and collect sources (for UI)
                context_blocks = []
                sources_for_ui = []
                for d in docs:
                    txt = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
                    meta = getattr(d, "metadata", {}) or {}
                    context_blocks.append(f"Source meta: {meta}\n{txt}")
                    sources_for_ui.append({"metadata": meta, "text": txt})

                raw_context = "\n\n---\n\n".join(context_blocks) if context_blocks else "No context passages available."
                context = truncate_context_by_chars(raw_context, max_chars=2000)
                # optionally add a tiny note included only for the LLM that we truncated context
                if len(raw_context) > len(context):
                    context += "\n\n[Context truncated to fit model limits]"

                # Prepare system prompt...
                system_prompt = (
                    "You are a concise, compassionate health assistant helping patients. "
                    "Answer in simple, non-technical language. Use only the information provided in the context passages. "
                    "Do not make a definitive medical diagnosis. If the user seems in immediate danger or mentions emergency symptoms, advise them to seek emergency care. "
                    "At the end, include a short line that tells the user to consult a healthcare professional for personalized advice."
                )

                chat_history_text = build_chat_history_text(st.session_state.history, MAX_HISTORY_MESSAGES)

                final_prompt = (
                    f"{system_prompt}\n\n"
                    f"Context passages (use ONLY these passages to answer). If the context doesn't contain the information, say you don't know and recommend seeing a professional:\n{context}\n\n"
                    f"Patient question: {user_input.strip()}\n\n"
                    f"Provide a short, clear answer in plain language."
                )

                # Initialize LLM and call it (separate try/except so we can see LLM errors)
                try:
                    llm = Ollama(model=llm_model_input)
                    with st.spinner("Thinking..."):
                        answer_text = safe_llm_call(llm, final_prompt, fallback_model_name=llm_model_input)
                except Exception as e_llm:
                    if advanced_mode:
                        st.sidebar.error("LLM generation error (details):")
                        st.sidebar.write(traceback.format_exc())
                    raise RuntimeError(f"LLM generation failed: {e_llm}")

                # Append assistant message with sources
                st.session_state.history.append({"role": "assistant", "content": answer_text, "sources": sources_for_ui})
                render_chat_into(chat_container)

            st.session_state["_clear_user_input"] = True

        except Exception as e:
            # On error, append a friendly message and provide admin info in sidebar (if advanced)
            user_friendly = (
                "Sorry — I couldn't process that right now. Please try again in a few moments."
            )
            st.session_state.history.append({"role": "assistant", "content": user_friendly, "sources": []})
            if advanced_mode:
                st.sidebar.error("Generation error (details):")
                st.sidebar.write(traceback.format_exc())
            render_chat_into(chat_container)

    # clear input box after processing
    st.session_state["_clear_user_input"] = True

# At the bottom: small footer and option to download conversation (simple copy)
st.markdown("---")
st.markdown("**Disclaimer:** This tool is informational only and not a replacement for professional medical advice.")
st.caption("Built for demonstration; verify clinical use and data sources before deploying in production.")
