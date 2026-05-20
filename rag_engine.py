"""
rag_engine.py
-------------
Builds and exposes the LangChain RAG chain (ChromaDB + Ollama).
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama 
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

_DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"
_DEFAULT_LLM_MODEL = "gemma3:1b"
_DEFAULT_CHROMA_DIR = "./chroma_db"


def build_qa_chain(
    transcript_path: str,
    embed_model: str = _DEFAULT_EMBED_MODEL,
    llm_model: str = _DEFAULT_LLM_MODEL,
    chroma_dir: str = _DEFAULT_CHROMA_DIR,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    k: int = 4,
) -> RetrievalQA:
    """
    Load a transcript, embed it into ChromaDB, and return a RetrievalQA chain.

    Parameters
    ----------
    transcript_path : str
        Path to the .txt transcript file.
    embed_model : str
        Ollama embedding model name.
    llm_model : str
        Ollama LLM model name.
    chroma_dir : str
        Directory to persist the Chroma vector store.
    chunk_size : int
        Character chunk size for the text splitter.
    chunk_overlap : int
        Overlap between consecutive chunks.
    k : int
        Number of chunks to retrieve per query.

    Returns
    -------
    RetrievalQA
        A ready-to-invoke LangChain RAG chain.
    """
    # 1. Load transcript
    loader = TextLoader(transcript_path, encoding="utf-8")
    documents = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    # 3. Embed & store in ChromaDB
    embeddings = OllamaEmbeddings(model=embed_model)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
    )

    # 4. LLM
    llm = Ollama(model=llm_model)

    # 5. RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        ),
        chain_type="stuff",
    )
    return qa_chain


def ask(chain: RetrievalQA, question: str) -> str:
    """
    Query the RAG chain and return the answer string.

    Returns a fallback message if the model says it doesn't know.
    """
    result = chain.invoke(question)
    answer = result.get("result", "").strip()
    if answer.lower() in ("i don't know.", "i don't know", ""):
        return "Sorry, this video doesn't provide knowledge based on your query."
    return answer
