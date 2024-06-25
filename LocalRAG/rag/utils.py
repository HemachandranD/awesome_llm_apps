import streamlit as st
import subprocess

def check_valid_file(file) -> str:
    """Reads an uploaded file and returns a File object"""
    if file.name.lower().endswith(".docx"):
        return "DOCX"
    elif file.name.lower().endswith(".pdf"):
        return "PDF"
    elif file.name.lower().endswith(".txt"):
        return "Text"
    elif file.name.lower().endswith(".md"):
        return "Markdown"
    else:
        raise NotImplementedError(f"File type {file.name.split('.')[-1]} not supported")


def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Follow the [prerequisites]()\n"
            "2. Run the LocalRAG app locally using streamlit\n"
            "2. Upload a pdf, docx, txt or md file📄\n"
            "3. Ask a question about the document💬\n"
        )

        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            "📝LocalRAG is a cool self-hosting application enables you to inquire "
            "about your documents and receive answers while ensuring the privacy "
            "and security of your data."
        )

        st.markdown("Made by [Hemz](https://hemz.medium.com/)")
        st.markdown("---")
