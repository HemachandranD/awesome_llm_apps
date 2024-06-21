from typing import Any, Dict, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# Load Data
def load_data(file_type: str, file_loc: str):
    """Load data into a list of Documents
    Args:
        file_type: the type of file to load
    Returns:    list of Documents
    """
    if file_type == "PDF":
        loader = PyPDFLoader(file_loc)
        data = loader.load()

    elif file_type == "Text":
        loader = TextLoader(file_loc)
        data = loader.load()

    return data


# Prepare Chunks
def prepare_chunk(data: list):
    """Prepare Chunks
    Args:
        data: list of Documents
    Returns:    list of Chunks
    """
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(data)

    print(f"Splitted {len(data)} documents into {len(documents)} chunks")
    return documents
