from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import Qdrant

# Create Vectorstores
def create_vstores(documents: list):
    print("****Local Embedding is in progress, Please wait... ***")
    embeddings = HuggingFaceEmbeddings(
    )

    print("****Loading to Vectorstore, Please wait... ***")
    print(f"Adding {len(documents)} to Qdrant Local")
    qdrant = Qdrant.from_documents(
        documents,
        embeddings,
        url="http://localhost:6333",
        collection_name="my_documents",
        force_recreate=True,
    )

    print("****Loading to Vectorstore, Done! ***")

    return qdrant, embeddings
