# 🎬 AskTube AI — YouTube Video Question Answering System

AskTube AI is an AI-powered Retrieval-Augmented Generation (RAG) application that enables users to interact with YouTube video content through natural language conversations.

The application automatically extracts video transcripts, converts them into embeddings, stores them in a vector database, and allows users to ask context-aware questions about the video content using Large Language Models (LLMs).

---

# 🚀 Features

- 🔗 YouTube URL transcript extraction
- 📄 Upload custom transcript `.txt` files
- 🧠 Retrieval-Augmented Generation (RAG)
- 📚 ChromaDB vector database integration
- 🤖 Local LLM inference using Ollama
- 💬 Interactive AI chat interface
- ⚡ Semantic similarity search
- ✂️ Recursive text chunking
- 🎨 Custom futuristic Streamlit UI
- 📜 Transcript preview panel
- ⚙️ Configurable embedding and LLM models

---

# 🛠️ Tech Stack

## Frontend
- Streamlit

## Backend / AI
- Python
- LangChain
- Ollama

## Vector Database
- ChromaDB

## NLP / Retrieval
- RecursiveCharacterTextSplitter
- Semantic Embeddings
- RetrievalQA

## APIs
- YouTube Transcript API

---

# 🧠 System Workflow

```text
User Input
   ↓
YouTube Transcript Extraction
   ↓
Text Chunking
   ↓
Embedding Generation
   ↓
ChromaDB Vector Storage
   ↓
Semantic Retrieval
   ↓
LLM Response Generation
```

---

# 📂 Project Structure

```bash
AskTube-AI/
│
├── asktube_app.py        # Main Streamlit application
├── requirements.txt      # Dependencies
├── chroma_db/            # Persistent vector database
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/asktube-ai.git
cd asktube-ai
```

---

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🦙 Setup Ollama

Install Ollama from:

https://ollama.com/

Pull required models:

```bash
ollama pull nomic-embed-text:latest
ollama pull llama3.2:3b
```

---

# ▶️ Run Application

```bash
streamlit run asktube_app.py
```

---

# 🎯 Usage

## Option 1 — YouTube URL

1. Paste a YouTube video URL
2. Click **Fetch & Load**
3. Wait for vector indexing
4. Start asking questions

---

## Option 2 — Upload Transcript

1. Upload a `.txt` transcript file
2. Wait for indexing
3. Ask context-aware questions

---

# 🧩 Core AI Concepts Used

- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Semantic Search
- Embedding Models
- Prompt Engineering
- LLM Grounding
- Context Retrieval Pipelines

---

# ⚡ Current Limitations

- Requires Ollama running locally
- Streamlit Cloud deployment does not support local Ollama inference
- English transcript support prioritized
- Long transcripts may increase embedding time

---

# 🔮 Future Improvements

- Multi-video knowledge base
- Conversation memory
- PDF support
- Whisper audio transcription
- Source citations
- Cloud LLM integration (Groq/OpenAI/Gemini)
- Authentication system
- Streaming AI responses

---

# 📸 UI Highlights

- Futuristic dark-themed interface
- Fully customized Streamlit styling
- Interactive transcript preview
- Dynamic chain status indicators
- Real-time chat interface

---

## 👨‍💻 Author 

Developed by Arjun Manohar ,
Linkedin Profile : www.linkedin.com/in/thearjunmanohar
