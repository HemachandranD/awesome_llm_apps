import os
import time

import streamlit as st
from streamlit.logger import get_logger

from localrag import loader, retrieval, vstores

logger = get_logger(__name__)

st.set_page_config(
    page_title="LocalRAG",
    page_icon="📝",
    initial_sidebar_state="expanded",
    menu_items={"About": "# This is an *extremely* cool Local RAG app!"},
)
st.header("📝 LocalRAG")

with st.sidebar:
    # anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    "[View the source code](https://github.com/HemachandranD/awesome_llm_apps/blob/main/LocalRAG/app.py)"

uploaded_file = st.file_uploader("Upload a File", type=("txt", "md", "pdf", "docx"))

# st.session_state['messages']=[]
def setup():
    if not uploaded_file:
        st.stop()

    if uploaded_file:
        st.toast("File uploaded Sucessfully!", icon="✅")
        time.sleep(0.5)

        with open(uploaded_file.name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        documents = loader(file=uploaded_file, file_loc=uploaded_file.name)
        os.remove(uploaded_file.name)
        if documents is None:
            st.info("No data found, Please be responsible and upload a file with data")
            st.stop()

        elif documents is not None:
            with st.spinner("Indexing document... This may take a while⏳"):
                vs_conn = vstores(documents=documents)
                st.toast("Indexing Completed", icon="🚀")
                time.sleep(0.5)

    return vs_conn

def welcome_chat():
    st.session_state["messages"] = [
        {"role": "assistant", "content": "What do you want to know from this file?"}
    ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


def rag_chat():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = retrieval(
        model_name="phi3", user_question=prompt, vstore_connection=None
    )
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)


if __name__ == "__main__":
    if "messages" not in st.session_state:
        vs_conn = setup()
        welcome_chat()
    if prompt := st.chat_input():
        rag_chat()
