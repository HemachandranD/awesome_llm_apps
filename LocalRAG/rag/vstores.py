from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
import logging

# Creating an object
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Create Vectorstores
def create_vstores(documents: list):
    try:
        logger.info("****Local Embedding is in progress, Please wait...****")
        embeddings = HuggingFaceEmbeddings(
        )

        logger.info("****Loading to Vectorstore, Please wait...****")
        logger.info(f"Adding {len(documents)} to Qdrant Local")
        qdrant = Qdrant.from_documents(
            documents,
            embeddings,
            url="http://localhost:6333",
            collection_name="my_documents",
            force_recreate=True,
        )
        logger.info("****Loading to Vectorstore, Done!****")

        return qdrant
    
    except Exception as e:
        logger.error(f"An error occurred in create_vstores: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")
