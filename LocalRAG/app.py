import streamlit as st
from localrag import loader, vstores, retrieval
import logging
import time
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

def setup():
    if not uploaded_file:
        st.stop()

    if uploaded_file:
        st.info("File uploaded Sucessfully!")
        st.toast(f"File uploaded Sucessfully!", icon="✅")
        time.sleep(0.5)
        documents=loader(file=uploaded_file)

        if documents == None:
            st.info("No data found, Please be responsible and upload a file with data")
            st.stop()

        elif documents != None:
            with st.spinner("Indexing document... This may take a while⏳"):
                vs_conn=vstores(documents=documents)
                st.toast(f"Indexing Completed", icon="🚀")
                time.sleep(0.5)
        
    return vs_conn

def rag_chat():
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        # client = OpenAI(api_key=openai_api_key)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        response = retrieval(model_name="llama3", user_question=prompt , vstore_connection=vs_conn)
        msg = response.content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)


def main():
    vs_conn = setup()
    answer = retrieval(model_name="llama3", user_question=question , vstore_connection=vs_conn)


if __name__ == "__main__":
    main()