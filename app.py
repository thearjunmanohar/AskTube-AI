"""
app.py  —  AskTube AI
---------------------
User pastes a YouTube URL or uploads a .txt file
→ RAG chain builds automatically → chat unlocks.

Run:
    streamlit run app.py
"""

import tempfile
import streamlit as st

st.set_page_config(page_title="AskTube AI", page_icon="🎬",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
body,[data-testid="stAppViewContainer"]{background:#0f0f11;color:#e8e8f0;font-family:'Inter',sans-serif}
[data-testid="stSidebar"]{background:#16161e;border-right:1px solid #2a2a3a}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#a78bfa}
.main-title{font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#a78bfa,#38bdf8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.2rem}
.sub-title{color:#6b7280;font-size:.95rem;margin-bottom:1.5rem}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;
  font-size:.8rem;font-weight:600}
.pill-ready{background:#14532d;color:#86efac}
.pill-busy{background:#713f12;color:#fde68a}
.pill-idle{background:#1e293b;color:#94a3b8}
.pill-error{background:#7f1d1d;color:#fca5a5}
.chat-wrap{display:flex;gap:12px;margin-bottom:16px;align-items:flex-start}
.chat-wrap.user{flex-direction:row-reverse}
.avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:1.1rem;flex-shrink:0}
.avatar-user{background:#4c1d95}.avatar-bot{background:#0c4a6e}
.bubble{max-width:78%;padding:10px 15px;border-radius:14px;font-size:.93rem;line-height:1.65}
.bubble-user{background:#1e1b4b;border:1px solid #312e81;border-top-right-radius:4px}
.bubble-bot{background:#0c1220;border:1px solid #1e3a5f;border-top-left-radius:4px}
.stTextInput>div>div>input{background:#1a1a2e!important;border:1px solid #2d2d50!important;
  border-radius:10px!important;color:#e8e8f0!important;padding:10px 14px!important}
.stTextInput>div>div>input:focus{border-color:#7c3aed!important;
  box-shadow:0 0 0 2px rgba(124,58,237,.25)!important}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#2563eb);color:#fff;
  border:none;border-radius:10px;font-weight:600;padding:8px 20px;transition:opacity .2s}
.stButton>button:hover{opacity:.88}
.tx-box{background:#11111a;border:1px solid #2a2a3a;border-radius:10px;
  padding:14px 18px;font-size:.83rem;color:#94a3b8;max-height:220px;
  overflow-y:auto;white-space:pre-wrap;line-height:1.7}
hr{border-color:#1f1f30}
[data-testid="stExpander"]{background:#13131d;border:1px solid #2a2a3a;border-radius:10px}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    qa_chain=None, transcript_text="", transcript_path="",
    messages=[], chain_ready=False,
    status="idle", status_msg="", last_source="", _last_upload="",
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _auto_build(path, embed_model, llm_model, chroma_dir,
                chunk_size, chunk_overlap, k):
    """Embed + index transcript, update session state."""
    from rag_engine import build_qa_chain
    st.session_state.status = "busy"
    st.session_state.chain_ready = False
    chain = build_qa_chain(
        transcript_path=path, embed_model=embed_model,
        llm_model=llm_model, chroma_dir=chroma_dir,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k,
    )
    st.session_state.qa_chain = chain
    st.session_state.chain_ready = True
    st.session_state.status = "ready"
    st.session_state.messages = []


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎬 AskTube AI")
    st.markdown("*Chat based on  any YouTube video or transcript*")
    st.divider()

    # # ── Model config ──────────────────────────────────────────────────────────
    # with st.expander("⚙️ Model settings", expanded=False):
    #     embed_model   = st.text_input("Embedding model", value="nomic-embed-text:latest", key="s_embed")
    #     llm_model     = st.text_input("LLM model", value="gemma3:1b", key="s_llm")
    #     chroma_dir    = st.text_input("ChromaDB dir", value="./chroma_db", key="s_chroma")
    #     chunk_size    = st.slider("Chunk size", 100, 2000, 500, 50, key="s_cs")
    #     chunk_overlap = st.slider("Chunk overlap", 0, 200, 40, 10, key="s_co")
    #     k_docs        = st.slider("Top-k chunks", 1, 10, 4, key="s_k")

    # # read back (use defaults if expander was never opened)
    # embed_model   = st.session_state.get("s_embed",   "nomic-embed-text:latest")
    # llm_model     = st.session_state.get("s_llm",     "gemma3:1b")
    # chroma_dir    = st.session_state.get("s_chroma",  "./chroma_db")
    # chunk_size    = st.session_state.get("s_cs",      500)
    # chunk_overlap = st.session_state.get("s_co",      40)
    # k_docs        = st.session_state.get("s_k",       4)

    # st.divider()

    # ── Option 1: YouTube URL ─────────────────────────────────────────────────
    st.markdown("### 🔗 Load from YouTube")
    yt_url    = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...",
                              label_visibility="collapsed", key="yt_url_input")
    # lang_pref = st.text_input("Language codes", value="en-US,en",
    #                           help="Comma-separated priority, e.g. en-US,en")
    lang_pref = "en-US,en"
    load_yt   = st.button("▶ Fetch & Load", use_container_width=True,
                          disabled=not bool(yt_url.strip()))

    if load_yt and yt_url.strip():
        with st.spinner("Fetching transcript…"):
            try:
                from transcript_fetcher import extract_video_id, fetch_transcript
                vid   = extract_video_id(yt_url.strip())
                langs = [l.strip() for l in lang_pref.split(",") if l.strip()]
                text  = fetch_transcript(vid, langs)
                tmp   = tempfile.NamedTemporaryFile(delete=False, suffix=".txt",
                                                    mode="w", encoding="utf-8")
                tmp.write(text); tmp.close()
                st.session_state.transcript_text = text
                st.session_state.transcript_path = tmp.name
                st.session_state.last_source = yt_url.strip()
                st.toast("Transcript fetched ✅")
            except Exception as e:
                st.session_state.status = "error"
                st.session_state.status_msg = str(e)
                st.error(f"❌ {e}")

        if st.session_state.transcript_path and st.session_state.status != "error":
            with st.spinner("Building RAG chain…"):
                try:
                    _auto_build(st.session_state.transcript_path,
                                'nomic-embed-text:latest', 'gemma3:1b', './chroma_db',
                                500, 40, 4)
                    st.toast("Chain ready 🚀")
                    st.rerun()
                except Exception as e:
                    st.session_state.status = "error"
                    st.session_state.status_msg = str(e)
                    st.error(f"❌ Chain build failed: {e}")

    st.divider()

    # ── Option 2: Upload .txt ─────────────────────────────────────────────────
    st.markdown("### 📄 Or upload a .txt file")
    uploaded = st.file_uploader("Upload transcript", type=["txt"],
                                label_visibility="collapsed")

    if uploaded:
        file_key = uploaded.name + str(uploaded.size)
        if st.session_state._last_upload != file_key:
            st.session_state._last_upload = file_key
            text = uploaded.read().decode("utf-8")
            tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".txt",
                                               mode="w", encoding="utf-8")
            tmp.write(text); tmp.close()
            st.session_state.transcript_text = text
            st.session_state.transcript_path = tmp.name
            st.session_state.last_source = uploaded.name

            with st.spinner("Building RAG chain…"):
                try:
                    _auto_build(st.session_state.transcript_path,
                                'nomic-embed-text:latest', 'gemma3:1b', './chroma_db',
                                500, 40, 4)
                    st.toast("Chain ready 🚀")
                    st.rerun()
                except Exception as e:
                    st.session_state.status = "error"
                    st.session_state.status_msg = str(e)
                    st.error(f"❌ Chain build failed: {e}")

    st.divider()

    # ── Transcript preview ────────────────────────────────────────────────────
    if st.session_state.transcript_text:
        with st.expander("📜 Transcript preview"):
            preview = st.session_state.transcript_text[:1500]
            if len(st.session_state.transcript_text) > 1500:
                preview += "\n\n… [truncated]"
            st.markdown(f'<div class="tx-box">{preview}</div>', unsafe_allow_html=True)

    # ── Status ────────────────────────────────────────────────────────────────
    s = st.session_state.status
    if s == "ready":
        st.markdown('<span class="pill pill-ready">● Chain ready</span>', unsafe_allow_html=True)
        st.caption(f"Source: {st.session_state.last_source}")
    elif s == "busy":
        st.markdown('<span class="pill pill-busy">⏳ Building…</span>', unsafe_allow_html=True)
    elif s == "error":
        st.markdown('<span class="pill pill-error">✗ Error</span>', unsafe_allow_html=True)
        st.caption(st.session_state.status_msg)
    else:
        st.markdown('<span class="pill pill-idle">○ Waiting for input</span>', unsafe_allow_html=True)

    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🎬 AskTube AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Paste a YouTube link or upload a transcript — '
    'the RAG chain builds automatically, then chat begins</div>',
    unsafe_allow_html=True,
)
st.divider()

# ── Onboarding (shown until chain is ready) ───────────────────────────────────
if not st.session_state.chain_ready:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**How it works**

1. 🔗 Paste a YouTube URL **or** upload a `.txt` transcript in the sidebar
2. ⚡ The RAG chain builds automatically (no extra button)
3. 💬 Ask anything — answers are grounded in the video content
        """)
    with c2:
        st.markdown("""
**Requirements**

- [Ollama](https://ollama.com) running locally
- Pull the models once:
```
ollama pull nomic-embed-text:latest
ollama pull gemma3:1b
```
        """)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
<div class="chat-wrap user">
  <div class="avatar avatar-user">👤</div>
  <div class="bubble bubble-user">{msg["content"]}</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="chat-wrap">
  <div class="avatar avatar-bot">🤖</div>
  <div class="bubble bubble-bot">{msg["content"]}</div>
</div>""", unsafe_allow_html=True)

# ── Input bar ─────────────────────────────────────────────────────────────────
st.divider()

if st.session_state.chain_ready:
    q_col, s_col = st.columns([9, 1])
    with q_col:
        user_input = st.text_input(
            "q", label_visibility="collapsed",
            placeholder="Ask anything about the content…",
            key=None,
        )
    with s_col:
        send = st.button("Send")

    if send and user_input.strip():
        q = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Thinking…"):
            try:
                from rag_engine import ask
                answer = ask(st.session_state.qa_chain, q)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
            
        st.session_state.messages.append({"role": "bot", "content": answer})
        st.rerun()
            
else:
    st.caption("⬅️ Load a video or transcript from the sidebar to start chatting.")
