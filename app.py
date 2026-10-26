import os
import re
import streamlit as st
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE CONFIG ---
st.set_page_config(page_title="Local RAG Assistant", page_icon="🤖", layout="wide")

st.title("📚 Local Hybrid RAG Assistant")
st.markdown("Powered by **Ollama (Llama 3.2)**, **ChromaDB**, **BM25**, and **RRF Hybrid Search** with full provenance tracking.")

# --- INITIALIZE PIPELINE COMPONENTS (Cached for Performance) ---
@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

@st.cache_resource
def init_llm():
    return OllamaLLM(model="llama3.2")

@st.cache_resource
def init_splitter():
    return RecursiveChunking() # or your standard text splitter configuration

embeddings = init_embeddings()
llm = init_llm()

# --- SIDEBAR: PDF INGESTION ---
st.sidebar.header("📁 Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload your PDF document", type=["pdf"])

# Helper function for RRF Hybrid Search (simplified for app integration)
def run_hybrid_search(query, bm25_ret, vec_ret, docs_list, k=3):
    # Retrieve from both
    bm25_docs = bm25_ret.invoke(query)
    vec_docs = vec_ret.invoke(query)
    
    # RRF fusion logic
    fusion_scores = {}
    doc_map = {doc.page_content: doc for doc in docs_list}
    
    for rank, doc in enumerate(bm25_docs):
        fusion_scores[doc.page_content] = fusion_scores.get(doc.page_content, 0) + 1 / (60 + rank + 1)
    for rank, doc in enumerate(vec_docs):
        fusion_scores[doc.page_content] = fusion_scores.get(doc.page_content, 0) + 1 / (60 + rank + 1)
        
    sorted_docs = sorted(fusion_scores.keys(), key=lambda x: fusion_scores[x], reverse=True)
    return [doc_map[content] for content in sorted_docs[:k] if content in doc_map]

# Initialize Session State for Retrievers and Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retrievers_ready" not in st.session_state:
    st.session_state.retrievers_ready = False

# Process Uploaded PDF
if uploaded_file is not None:
    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if not st.session_state.retrievers_ready:
        with st.spinner("Processing and normalizing PDF chunks..."):
            loader = PyPDFLoader(temp_pdf_path)
            raw_pages = loader.load()
            
            clean_pages = []
            for page in raw_pages:
                normalized_text = re.sub(r'\s+', ' ', page.page_content).strip()
                clean_pages.append(Document(page_content=normalized_text, metadata=page.metadata.copy()))
                
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(clean_pages)
            
            for i, doc in enumerate(docs):
                doc.metadata["chunk_id"] = f"Chunk_{i+1}"
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"] + 1
                    
            st.session_state.docs = docs
            st.session_state.bm25 = BM25Retriever.from_documents(docs)
            st.session_state.bm25.k = 4
            
            vectorstore = Chroma.from_documents(docs, embeddings)
            st.session_state.vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            st.session_state.retrievers_ready = True
            st.success(f"Successfully indexed {len(docs)} pristine chunks from your PDF!")

# --- MAIN CHAT INTERFACE ---
if st.session_state.retrievers_ready:
    # Display chat history (multi-turn support)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input for new questions
    if user_query := st.chat_input("Ask a question about your document..."):
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Retrieve relevant chunks using hybrid search
        with st.spinner("Searching document & generating response with citations..."):
            source_docs = run_hybrid_search(
                user_query, 
                st.session_state.bm25, 
                st.session_state.vector_retriever, 
                st.session_state.docs, 
                k=3
            )
            
            context_text = "\n\n".join([
                f"[{doc.metadata.get('chunk_id')}] (Page {doc.metadata.get('page')}):\n{doc.page_content}" 
                for doc in source_docs
            ])
            
            # Incorporate chat history for multi-turn conversational context if desired, or keep direct prompt
            generation_prompt = f"""
You are a precise technical assistant. Answer the user question using ONLY the provided context blocks. 
Every key claim or fact you state MUST be explicitly cited with its corresponding source bracket (e.g., [Chunk_19, Page 10]). 
If the answer cannot be found in the context, state "I cannot find this information in the document."

Context:
{context_text}

User Question: {user_query}
Answer:
"""
            answer = llm.invoke(generation_prompt)

        # Append assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("View Source Provenance"):
                for doc in source_docs:
                    st.write(f"**[{doc.metadata.get('chunk_id')}] Page {doc.metadata.get('page')}**: {doc.page_content[:150]}...")
else:
    st.info("👈 Please upload a PDF file in the sidebar to begin your chat session.")