import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from pypdf import PdfReader

load_dotenv()

st.set_page_config(page_title="NVIDIA RAG System", layout="wide")

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    st.error("⚠️ Invalid NVIDIA API Key. Check your .env file.")
    st.stop()

st.title("🚀 NVIDIA NIM - Document Q&A System")
st.markdown("Using ChromaDB for vector storage")

@st.cache_resource
def get_llm():
    return ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0.7)

@st.cache_resource
def get_embeddings():
    return NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")

def extract_text_from_pdf(file_path):
    """Extract text from PDF file with better error handling"""
    try:
        reader = PdfReader(file_path)
        text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    page_text = page_text.encode('utf-8', errors='ignore').decode('utf-8')
                    text += page_text + "\n"
            except Exception as e:
                st.warning(f"Could not extract text from page {page_num + 1}: {str(e)}")
                continue
        
        if text.strip():
            return text
        else:
            st.error("No text could be extracted from this PDF. It might be scanned or image-based.")
            return None
            
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing with ChromaDB..."):
                all_texts = []
                for f in uploaded_files:
                    suffix = f".{f.name.split('.')[-1]}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.getvalue())
                        path = tmp.name
                    
                    if f.name.endswith('.pdf'):
                        text = extract_text_from_pdf(path)
                        if text:
                            all_texts.append({"content": text, "source": f.name})
                    else:
                        with open(path, 'r', encoding='utf-8') as txt_file:
                            text = txt_file.read()
                            all_texts.append({"content": text, "source": f.name})
                    
                    os.unlink(path)
                
                if all_texts:
                    class SimpleDocument:
                        def __init__(self, page_content, metadata):
                            self.page_content = page_content
                            self.metadata = metadata
                    
                    documents = [SimpleDocument(t["content"], {"source": t["source"]}) for t in all_texts]
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(documents)
                    
                    vectorstore = Chroma.from_documents(
                        chunks, 
                        get_embeddings(), 
                        persist_directory="./chroma_db"
                    )
                    st.session_state.vectorstore = vectorstore
                    st.success(f"✅ Processed {len(uploaded_files)} files into {len(chunks)} chunks")
                    st.info("💡 ChromaDB is now storing your document vectors")
                else:
                    st.error("No text could be extracted")
        else:
            st.warning("Select files first")

st.header("💬 Ask Questions")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        if "vectorstore" in st.session_state:
            with st.spinner("Searching with ChromaDB..."):
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n\n".join([d.page_content for d in docs])
                result = get_llm().invoke(
                    f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer based only on context:"
                )
                # ✅ Extract clean text from ChatNVIDIA response
                response = result.content if hasattr(result, 'content') else str(result)

                with st.expander("📚 Sources"):
                    for i, d in enumerate(docs):
                        source = d.metadata.get("source", "Unknown")
                        st.write(f"**{i+1} ({source}):** {d.page_content[:300]}...")
        else:
            response = "Please upload and process documents first using the sidebar."

        # ✅ response is now always a plain string
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})