"""
asktube_app.py  —  AskTube AI  (single file)
--------------------------------------------
Run:
    streamlit run asktube_app.py
"""

import tempfile
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AskTube AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: #080810 !important;
    color: #dde2f0;
    font-family: 'DM Sans', sans-serif;
}

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0f0f1a; }
::-webkit-scrollbar-thumb { background: #2d2d50; border-radius: 4px; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0c0c18 !important;
    border-right: 1px solid #1c1c30;
}
[data-testid="stSidebar"] section { padding-top: 1.2rem !important; }

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── main content padding ── */
[data-testid="stAppViewContainer"] > .main .block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 900px;
}

/* ─────────────────────────────────────────
   HERO HEADER
───────────────────────────────────────── */
.hero {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 6px;
}
.hero-icon {
    font-size: 2.6rem;
    line-height: 1;
    filter: drop-shadow(0 0 14px rgba(139,92,246,.7));
}
.hero-text h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(120deg, #c4b5fd 0%, #818cf8 40%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.hero-text p {
    font-size: .85rem;
    color: #555878;
    margin-top: 3px;
    font-weight: 300;
    letter-spacing: .03em;
}

/* ─────────────────────────────────────────
   SIDEBAR LABELS
───────────────────────────────────────── */
.sb-label {
    font-family: 'Space Mono', monospace;
    font-size: .68rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: #5b5f80;
    margin-bottom: 6px;
    margin-top: 18px;
}

/* ─────────────────────────────────────────
   STATUS PILL
───────────────────────────────────────── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: .75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    letter-spacing: .04em;
}
.pill-idle  { background: rgba(255,255,255,.04); color: #555878; border: 1px solid #1e1e35; }
.pill-busy  { background: rgba(251,191,36,.08);  color: #fbbf24; border: 1px solid rgba(251,191,36,.2); }
.pill-ready { background: rgba(52,211,153,.08);  color: #34d399; border: 1px solid rgba(52,211,153,.2); }
.pill-error { background: rgba(248,113,113,.08); color: #f87171; border: 1px solid rgba(248,113,113,.2); }
.dot { width: 7px; height: 7px; border-radius: 50%; }
.dot-idle  { background: #555878; }
.dot-busy  { background: #fbbf24; box-shadow: 0 0 6px #fbbf24; animation: pulse 1.2s infinite; }
.dot-ready { background: #34d399; box-shadow: 0 0 6px #34d399; }
.dot-error { background: #f87171; box-shadow: 0 0 6px #f87171; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

/* ─────────────────────────────────────────
   DIVIDER
───────────────────────────────────────── */
.hline {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e1e35 30%, #1e1e35 70%, transparent);
    margin: 18px 0;
}

/* ─────────────────────────────────────────
   TRANSCRIPT BOX
───────────────────────────────────────── */
.tx-box {
    background: #0a0a15;
    border: 1px solid #1c1c2e;
    border-radius: 10px;
    padding: 14px 16px;
    font-size: .78rem;
    color: #4a4e6a;
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.75;
    font-family: 'Space Mono', monospace;
}

/* ─────────────────────────────────────────
   CHAT MESSAGES
───────────────────────────────────────── */
.chat-outer { display: flex; gap: 11px; margin-bottom: 18px; align-items: flex-start; }
.chat-outer.user { flex-direction: row-reverse; }

.avatar {
    width: 34px; height: 34px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.av-user { background: linear-gradient(135deg,#4c1d95,#312e81); border: 1px solid #5b21b6; }
.av-bot  { background: linear-gradient(135deg,#0c2240,#0c4a6e); border: 1px solid #1e4d7a; }

.bubble {
    max-width: 76%;
    padding: 11px 16px;
    border-radius: 14px;
    font-size: .9rem;
    line-height: 1.7;
}
.bubble-user {
    background: #120e2a;
    border: 1px solid #2e2460;
    border-top-right-radius: 4px;
    color: #c4b5fd;
}
.bubble-bot {
    background: #080f1e;
    border: 1px solid #152038;
    border-top-left-radius: 4px;
    color: #bfcfea;
}
.bubble-name {
    font-size: .68rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: .06em;
    margin-bottom: 5px;
    opacity: .45;
}

/* ─────────────────────────────────────────
   ONBOARDING CARD
───────────────────────────────────────── */
.onboard {
    background: linear-gradient(135deg, #0d0d1f 0%, #0a0f20 100%);
    border: 1px solid #1a1a30;
    border-radius: 16px;
    padding: 28px 30px;
    margin-bottom: 24px;
}
.onboard h3 {
    font-family: 'Space Mono', monospace;
    font-size: .8rem;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: #818cf8;
    margin-bottom: 14px;
}
.step {
    display: flex;
    gap: 12px;
    margin-bottom: 12px;
    align-items: flex-start;
}
.step-num {
    width: 24px; height: 24px;
    border-radius: 6px;
    background: rgba(129,140,248,.12);
    border: 1px solid rgba(129,140,248,.25);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Space Mono', monospace;
    font-size: .72rem;
    color: #818cf8;
    flex-shrink: 0;
    margin-top: 2px;
}
.step-body { font-size: .87rem; color: #5b6080; line-height: 1.6; }
.step-body strong { color: #9ca3c8; font-weight: 500; }

/* ─────────────────────────────────────────
   STREAMLIT WIDGET OVERRIDES
───────────────────────────────────────── */
/* text inputs */
.stTextInput > div > div > input,
.stTextArea textarea {
    background: #0d0d1e !important;
    border: 1px solid #1e1e35 !important;
    border-radius: 10px !important;
    color: #dde2f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .88rem !important;
    padding: 10px 14px !important;
    transition: border-color .2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: #4c1d95 !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,.15) !important;
}

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, #5b21b6, #1d4ed8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: .87rem !important;
    padding: 9px 18px !important;
    transition: opacity .18s, transform .12s !important;
    letter-spacing: .02em !important;
}
.stButton > button:hover { opacity: .85 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled { opacity: .3 !important; transform: none !important; }

/* file uploader */
[data-testid="stFileUploader"] {
    background: #0a0a18 !important;
    border: 1px dashed #1e1e38 !important;
    border-radius: 10px !important;
    padding: 10px !important;
}

/* sliders */
[data-testid="stSlider"] > div > div > div {
    background: #1e1e35 !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
}

/* expander */
[data-testid="stExpander"] {
    background: #0a0a18 !important;
    border: 1px solid #1a1a2e !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: .75rem !important;
    letter-spacing: .06em !important;
    color: #555878 !important;
}

/* select box */
[data-testid="stSelectbox"] > div > div {
    background: #0d0d1e !important;
    border: 1px solid #1e1e35 !important;
    border-radius: 10px !important;
    color: #dde2f0 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    qa_chain=None,
    transcript_text="",
    transcript_path="",
    messages=[],
    chain_ready=False,
    status="idle",          # idle | busy | ready | error
    status_msg="",
    last_source="",
    _last_upload="",
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  CORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are AskTube AI. Answer strictly from the transcript excerpts below.
If the answer is not in the excerpts, reply: Sorry , the transcript doesn't have the knowledge for me to answer this question.

Transcript excerpts:
{context}

Question: {question}
Answer:""",
)


def extract_video_id(url: str) -> str:
    if "v=" in url:
        vid = url.split("v=")[1].split("&")[0]
        return vid
    return url.strip()


def fetch_transcript(video_id: str, langs: list[str]) -> str:
    api = YouTubeTranscriptApi()
    tl  = api.list(video_id)
    t   = tl.find_transcript(langs).fetch()
    return " ".join(seg.text for seg in t)


def build_chain(path: str, embed_model: str, llm_model: str,
                chroma_dir: str, chunk_size: int, chunk_overlap: int, k: int):
    loader    = TextLoader(path, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model=embed_model)
    vector_db  = Chroma.from_documents(
        documents=chunks, embedding=embeddings,
        persist_directory=chroma_dir,
    )

    llm = Ollama(model=llm_model, num_ctx=2048, temperature=0.2)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        ),
        chain_type="stuff",
        chain_type_kwargs={"prompt": _RAG_PROMPT},
    )


def run_build(path: str, **kwargs):
    """Wrap build_chain and update session state."""
    st.session_state.status      = "busy"
    st.session_state.chain_ready = False
    try:
        chain = build_chain(path, **kwargs)
        st.session_state.qa_chain    = chain
        st.session_state.chain_ready = True
        st.session_state.status      = "ready"
        st.session_state.messages    = []
    except Exception as e:
        st.session_state.status     = "error"
        st.session_state.status_msg = str(e)
        raise


def ask(chain, question: str) -> str:
    result = chain.invoke(question)
    answer = result.get("result", "").strip()
    if not answer or answer.lower().rstrip(".") == "i don't know":
        return "Sorry, this video doesn't seem to cover that topic."
    return answer


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:

    # logo
    st.markdown("""
    <div style="padding:4px 0 20px">
      <div style="font-family:'Space Mono',monospace;font-size:1.1rem;
                  font-weight:700;color:#818cf8;letter-spacing:.04em;">
        🎬 AskTube<span style="color:#38bdf8">AI</span>
      </div>
      <div style="font-size:.72rem;color:#2d3050;margin-top:3px;
                  font-family:'Space Mono',monospace;letter-spacing:.08em;">
        RAG · CHROMADB · OLLAMA
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model settings ────────────────────────────────────────────────────────
    with st.expander("⚙  MODEL SETTINGS", expanded=False):
        embed_model   = st.text_input("Embedding model", "nomic-embed-text:latest", key="s_embed")
        llm_model     = st.text_input("LLM model",       "llama3.2:3b",               key="s_llm")
        chroma_dir    = st.text_input("ChromaDB dir",    "./chroma_db",             key="s_chroma")
        chunk_size    = st.slider("Chunk size (chars)",  100, 1000, 400, 50,        key="s_cs")
        chunk_overlap = st.slider("Chunk overlap",        0,   200,  40, 10,        key="s_co")
        k_docs        = st.slider("Top-k chunks",         1,     6,   2,            key="s_k")

    em = st.session_state.get("s_embed",  "nomic-embed-text:latest")
    lm = st.session_state.get("s_llm",   "llama3.2:3b")
    cd = st.session_state.get("s_chroma","./chroma_db")
    cs = st.session_state.get("s_cs",    1000)
    co = st.session_state.get("s_co",    200)
    kk = st.session_state.get("s_k",     4)

    chain_kwargs = dict(embed_model=em, llm_model=lm, chroma_dir=cd,
                        chunk_size=cs, chunk_overlap=co, k=kk)

    # ── YouTube URL ───────────────────────────────────────────────────────────
    st.markdown('<div class="sb-label">YouTube URL</div>', unsafe_allow_html=True)
    yt_url  = st.text_input("url", placeholder="https://youtube.com/watch?v=...",
                            label_visibility="collapsed", key="yt_url")
    # lang    = st.text_input("Language codes", "en-US,en",
    #                         help="Comma-separated priority, e.g. en-US,en")
    lang = "en-US,en"
    yt_btn  = st.button("▶  Fetch & Load", use_container_width=True,
                        disabled=not bool(yt_url.strip()))

    if yt_btn and yt_url.strip():
        with st.spinner("Fetching transcript…"):
            try:
                vid   = extract_video_id(yt_url.strip())
                langs = [l.strip() for l in lang.split(",") if l.strip()]
                text  = fetch_transcript(vid, langs)
                tmp   = tempfile.NamedTemporaryFile(delete=False, suffix=".txt",
                                                    mode="w", encoding="utf-8")
                tmp.write(text); tmp.close()
                st.session_state.transcript_text = text
                st.session_state.transcript_path = tmp.name
                st.session_state.last_source     = yt_url.strip()
            except Exception as e:
                st.session_state.status     = "error"
                st.session_state.status_msg = str(e)
                st.error(f"Transcript error: {e}")

        if st.session_state.transcript_path and st.session_state.status != "error":
            with st.spinner("Embedding & indexing…"):
                try:
                    run_build(st.session_state.transcript_path, **chain_kwargs)
                    st.toast("Chain ready 🚀")
                    st.rerun()
                except Exception as e:
                    st.error(f"Chain error: {e}")

    st.markdown('<div class="hline"></div>', unsafe_allow_html=True)

    # ── Upload .txt ───────────────────────────────────────────────────────────
    st.markdown('<div class="sb-label">Or upload a .txt transcript</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("txt", type=["txt"], label_visibility="collapsed")

    if uploaded:
        fkey = uploaded.name + str(uploaded.size)
        if st.session_state._last_upload != fkey:
            st.session_state._last_upload = fkey
            text = uploaded.read().decode("utf-8")
            tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".txt",
                                               mode="w", encoding="utf-8")
            tmp.write(text); tmp.close()
            st.session_state.transcript_text = text
            st.session_state.transcript_path = tmp.name
            st.session_state.last_source     = uploaded.name
            with st.spinner("Embedding & indexing…"):
                try:
                    run_build(st.session_state.transcript_path, **chain_kwargs)
                    st.toast("Chain ready 🚀")
                    st.rerun()
                except Exception as e:
                    st.error(f"Chain error: {e}")

    # ── Transcript preview ────────────────────────────────────────────────────
    if st.session_state.transcript_text:
        with st.expander("📜  TRANSCRIPT PREVIEW"):
            preview = st.session_state.transcript_text[:1200]
            if len(st.session_state.transcript_text) > 1200:
                preview += "\n\n… [truncated]"
            st.markdown(f'<div class="tx-box">{preview}</div>', unsafe_allow_html=True)

    st.markdown('<div class="hline"></div>', unsafe_allow_html=True)

    # ── Status ────────────────────────────────────────────────────────────────
    s = st.session_state.status
    labels = {
        "idle":  ("idle",  "WAITING FOR INPUT"),
        "busy":  ("busy",  "BUILDING CHAIN…"),
        "ready": ("ready", "CHAIN READY"),
        "error": ("error", "ERROR"),
    }
    cls, txt = labels.get(s, ("idle", "IDLE"))
    st.markdown(
        f'<div class="pill pill-{cls}">'
        f'<span class="dot dot-{cls}"></span>{txt}</div>',
        unsafe_allow_html=True,
    )
    if s == "ready" and st.session_state.last_source:
        st.markdown(
            f'<div style="font-size:.7rem;color:#2d3258;margin-top:6px;'
            f'font-family:Space Mono,monospace;word-break:break-all;">'
            f'{st.session_state.last_source}</div>',
            unsafe_allow_html=True,
        )
    if s == "error":
        st.caption(st.session_state.status_msg)

    # ── Clear chat ────────────────────────────────────────────────────────────
    if st.session_state.messages:
        st.markdown('<div style="margin-top:14px"></div>', unsafe_allow_html=True)
        if st.button("🗑  Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

# hero
st.markdown("""
<div class="hero">
  <div class="hero-icon">🎬</div>
  <div class="hero-text">
    <h1>AskTube AI</h1>
    <p>Paste a YouTube link or upload a transcript — then ask anything about the video</p>
  </div>
</div>
<div class="hline" style="margin-bottom:24px"></div>
""", unsafe_allow_html=True)

# ── Onboarding ────────────────────────────────────────────────────────────────
if not st.session_state.chain_ready:
    st.markdown("""
    <div class="onboard">
      <h3>Get started</h3>
      <div class="step">
        <div class="step-num">1</div>
        <div class="step-body">
          <strong>Paste a YouTube URL</strong> or <strong>upload a .txt transcript</strong> in the sidebar
        </div>
      </div>
      <div class="step">
        <div class="step-num">2</div>
        <div class="step-body">
          The RAG chain <strong>builds automatically</strong> — no extra button needed
        </div>
      </div>
      <div class="step">
        <div class="step-num">3</div>
        <div class="step-body">
          Ask anything below — answers are <strong>grounded in the video content</strong>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#08080f;border:1px solid #141425;border-radius:12px;
                padding:18px 20px;font-size:.82rem;color:#2d3258;
                font-family:'Space Mono',monospace;line-height:1.9;">
    <span style="color:#4a4e6a">## Requirements</span><br>
    <span style="color:#3d4060"># Ollama running locally</span><br>
    ollama pull <span style="color:#818cf8">nomic-embed-text:latest</span><br>
    ollama pull <span style="color:#38bdf8">gemma3:1b</span>
    </div>
    """, unsafe_allow_html=True)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-outer user">
          <div class="avatar av-user">👤</div>
          <div class="bubble bubble-user">
            <div class="bubble-name">YOU</div>
            {msg["content"]}
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-outer">
          <div class="avatar av-bot">🤖</div>
          <div class="bubble bubble-bot">
            <div class="bubble-name">ASKTUBE AI</div>
            {msg["content"]}
          </div>
        </div>""", unsafe_allow_html=True)

# ── Input bar ─────────────────────────────────────────────────────────────────
if st.session_state.chain_ready:
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    q_col, s_col = st.columns([9, 1])
    with q_col:
        user_input = st.text_input(
            "q", label_visibility="collapsed",
            placeholder="Ask anything about the video…"
        )
    with s_col:
        send = st.button("Send", use_container_width=True)

    if send and user_input.strip():
        q = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": q})
        with st.spinner("Thinking…"):
            try:
                answer = ask(st.session_state.qa_chain, q)
            except Exception as e:
                answer = f"⚠️ Error: {e}"
        st.session_state.messages.append({"role": "bot", "content": answer})
        st.rerun()
else:
    st.markdown(
        '<div style="font-size:.83rem;color:#252840;font-family:Space Mono,monospace;'
        'margin-top:20px;">← load a video from the sidebar to unlock chat</div>',
        unsafe_allow_html=True,
    )
