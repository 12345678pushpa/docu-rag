# import streamlit as st
# import os
# import tempfile
# from langchain_community.document_loaders import TextLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.llms import Ollama

# st.set_page_config(page_title="Docu - Working RAG App", layout="wide")

# st.title("📄 Docu - Document Q&A System")
# st.markdown("Upload documents and ask questions")

# # Initialize Ollama
# @st.cache_resource
# def get_llm():
#     return Ollama(model="llama2", temperature=0.7)

# @st.cache_resource
# def get_embeddings():
#     return OllamaEmbeddings(model="llama2")

# # Sidebar for upload
# with st.sidebar:
#     st.header("📁 Upload")
#     files = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)
    
#     if st.button("Process"):
#         if files:
#             with st.spinner("Processing..."):
#                 all_docs = []
#                 for f in files:
#                     suffix = f".{f.name.split('.')[-1]}"
#                     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                         tmp.write(f.getvalue())
#                         path = tmp.name
                    
#                     if f.name.endswith('.pdf'):
#                         loader = PyPDFLoader(path)
#                     else:
#                         loader = TextLoader(path, encoding='utf-8')
                    
#                     all_docs.extend(loader.load())
#                     os.unlink(path)
                
#                 splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#                 chunks = splitter.split_documents(all_docs)
#                 vectorstore = Chroma.from_documents(chunks, get_embeddings(), persist_directory="./db")
#                 st.session_state.vectorstore = vectorstore
#                 st.success(f"✅ Done: {len(files)} files, {len(chunks)} chunks")
#         else:
#             st.warning("Select files first")

# # Chat area
# st.header("💬 Ask")

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"])

# if prompt := st.chat_input("Ask about your documents..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     with st.chat_message("assistant"):
#     if "vectorstore" in st.session_state:
#         docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
#         context = "\n\n".join([d.page_content for d in docs])
    
#         result = get_llm().invoke(
#         f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer based only on context:"
#         )
    
#     # ✅ Handles both string and object responses
#     response = result.content if hasattr(result, 'content') else str(result)
    
#     with st.expander("Sources"):
#         for i, d in enumerate(docs):
#             st.write(f"**{i+1}:** {d.page_content[:300]}...")
#   else:
#     response = "⚠️ Upload and process documents first"

# st.markdown(response)
# st.session_state.messages.append({"role": "assistant", "content": response})



import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

st.set_page_config(page_title="Docu - Working RAG App", layout="wide")

st.title("📄 Docu - Document Q&A System")
st.markdown("Upload documents and ask questions")

@st.cache_resource
def get_llm():
    return Ollama(model="gemma3:4b", temperature=0.7)

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

# Sidebar for upload
with st.sidebar:
    st.header("📁 Upload")
    files = st.file_uploader("Choose PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True)

    if st.button("Process"):
        if files:
            with st.spinner("Processing..."):
                all_docs = []
                for f in files:
                    suffix = f".{f.name.split('.')[-1]}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(f.getvalue())
                        path = tmp.name

                    if f.name.endswith('.pdf'):
                        loader = PyPDFLoader(path)
                    else:
                        loader = TextLoader(path, encoding='utf-8')

                    all_docs.extend(loader.load())
                    os.unlink(path)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(all_docs)
                vectorstore = Chroma.from_documents(chunks, get_embeddings(), persist_directory="./db")
                st.session_state.vectorstore = vectorstore
                st.success(f"✅ Done: {len(files)} files, {len(chunks)} chunks")
        else:
            st.warning("Select files first")

# Chat area
st.header("💬 Ask")

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
            docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
            context = "\n\n".join([d.page_content for d in docs])

            result = get_llm().invoke(
                f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer based only on context:"
            )

            # Extract only the text content
            response = result.content if hasattr(result, 'content') else str(result)

            with st.expander("📚 Sources"):
                for i, d in enumerate(docs):
                    st.write(f"**{i+1}:** {d.page_content[:300]}...")
        else:
            response = "⚠️ Upload and process documents first"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})