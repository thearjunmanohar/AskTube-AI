# 🎬 AskTube AI — YouTube Video Question Answering System

AskTube AI is an AI-powered Retrieval-Augmented Generation (RAG) application that enables users to interact with YouTube video content through natural language conversations.

The application automatically extracts video transcripts, converts them into embeddings, stores them in a vector database, and allows users to ask context-aware questions about the video content using Large Language Models (LLMs).

---

## 🚀 Features

- 🔗 Accepts YouTube video URLs
- 📝 Fetches and processes video transcripts automatically
- 🧠 Uses Retrieval-Augmented Generation (RAG)
- 📚 ChromaDB-based vector storage
- 🤖 Local LLM inference using Ollama
- 💬 Interactive Streamlit chat interface
- ✂️ Intelligent text chunking for retrieval optimization
- 📂 Supports transcript file uploads
- ⚡ Fast semantic search and contextual response generation

---

## 🛠️ Tech Stack

### Frontend
- Streamlit

### Backend / AI
- Python
- LangChain
- Ollama
- ChromaDB

### NLP & Retrieval
- Recursive Character Text Splitter
- Vector Embeddings
- Semantic Search
- RetrievalQA Chain

### APIs
- YouTube Transcript API

---

## 📌 System Workflow

1. User submits a YouTube URL or transcript file
2. Transcript is extracted using YouTube Transcript API
3. Text is split into manageable chunks
4. Chunks are converted into embeddings
5. Embeddings are stored in ChromaDB
6. User asks questions in natural language
7. Relevant chunks are retrieved semantically
8. LLM generates context-aware responses

---

## 🧠 Core Concepts Used

1. Retrieval-Augmented Generation (RAG)
2. Semantic Search
3. Vector Databases
4. Embedding Models
5. Large Language Models (LLMs)
6. Context Retrieval
7. Prompt Augmentation

---

## 📈 Future Improvements

1. Multi-video knowledge base
2. Conversation memory
3. PDF and document support
4. Whisper-based audio transcription
5. Source citation in responses
   
---

## 🎯 Use Cases

1. Educational video assistants
2. AI-powered lecture summarization
3. Research content exploration
4. Technical tutorial Q&A
5. Knowledge extraction from long videos

---

## 📂 Project Structure

```bash
AskTube LLM/
│
├── app.py                     # Streamlit frontend
├── rag_engine.py              # RAG pipeline and RetrievalQA logic
├── transcript_fetcher.py      # Transcript extraction utilities
├── requirements.txt           # Project dependencies
├── chroma_db/                 # Persistent vector database

```

---

## 👨‍💻 Author 

Developed by Arjun Manohar
Linkedin Profile : www.linkedin.com/in/thearjunmanohar
