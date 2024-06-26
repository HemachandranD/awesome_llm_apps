import os
import time

import streamlit as st
from rag.utils import sidebar, _get_session
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

sidebar()

uploaded_file = st.file_uploader("Upload a File", type=("txt", "md", "pdf", "docx"))


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


def welcome_message():
    # Initialize chat history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey, How can I help you with this File?"}
    ]
    with st.chat_message("assistant"):
        for msg in st.session_state.messages:
            st.markdown(msg["content"])


def rag_chat(prompt, model, session_id):
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Generating response.."):
        st.info(session_id)
        response = retrieval(
            model_name=model, user_question=prompt, vstore_connection=None, session_id=session_id
        )
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    session_id = _get_session()
    if "messages" not in st.session_state:
        vs_conn = setup()
        welcome_message()
    if prompt := st.chat_input():
        rag_chat(prompt, model="llama3", session_id=session_id)
