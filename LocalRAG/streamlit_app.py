import streamlit as st
from rag.loader import load_data, prepare_chunk
from rag.retrieval import run_llm
from rag.vstores import create_vstores
from langchain_qdrant import Qdrant
import logging
from streamlit.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    layout="wide",
    page_title="LocalRAG",
    page_icon="📝",
    initial_sidebar_state="expanded",
    menu_items={"About": "# This is an *extremely* cool Local RAG app!"},
)
st.header("📝 LocalRAG")
# st.title("📝 LocalRAG")

with st.sidebar:
    # anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    "[View the source code](https://github.com/HemachandranD/awesome_llm_apps/blob/main/LocalRAG/app.py)"

uploaded_file = st.file_uploader("Upload a File", type=("txt", "md", "pdf", "docx"))

question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if not uploaded_file:
    st.stop()

if uploaded_file:
    st.info("File uploaded Sucessfully!")
    file_type = check_valid_file(uploaded_file)
    data = load_data(
        file_type=file_type, "/teamspace/studios/this_studio/awesome_llm_apps/LocalRAG/langgraph.txt"
    )
    documents = prepare_chunk(data)
    vs_conn = create_vstores(documents)
