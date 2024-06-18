from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant


# Create Vectorstores
def create_vstores(documents: list):
    print("****Local Embedding is in progress, Please wait... ***")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("****Loading to Vectorstore, Please wait... ***")
    print(f"Adding {len(documents)} to Qdrant Local")
    qdrant = Qdrant.from_documents(
        documents,
        embeddings,
        path="/tmp/local_qdrant",
        collection_name="my_documents",
    )
    print("****Loading to Vectorstore, Done! ***")

    return qdrant
