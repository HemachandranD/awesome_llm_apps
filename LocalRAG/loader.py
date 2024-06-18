from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# Load Data
def load_data(file_type: str) -> list:
    """Load data into a list of Documents
    Args:
        file_type: the type of file to load
    Returns:    list of Documents
    """
    if file_type == "PDF":
        loader = PyPDFLoader("")
        data = loader.load()

    elif file_type == "Text":
        loader = TextLoader("")
        data = loader.load()

    return data


# Prepare Chunks
def prepare_chunk(data: list) -> list:
    """Prepare Chunks
    Args:
        data: list of Documents
    Returns:    list of Chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=50, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(data)

    print(f"Splitted {len(data)} documents into {len(documents)} chunks")
    return documents
