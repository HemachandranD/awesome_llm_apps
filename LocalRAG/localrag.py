from rag.loader import load_data, prepare_chunk
from rag.retrieval import run_llm
from rag.vstores import create_vstores
from rag.utils import check_valid_file
import logging

# Creating an object
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

    
def loader(file=uploaded_file):
    try:
        file_type = check_valid_file(file)
        data = load_data(
            file_type=file_type, file_loc= file
        )
        documents, chunks = prepare_chunk(data)
        if chunks == 0:
            logger.error(f"No data found, Please upload a file with data")
            documents = None

        return documents
    
    except Exception as e:
        logger.error(f"An error occurred in loader: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")


def vstores(documents=documents):
    try:
        vs_conn = create_vstores(documents)

        return vs_conn
    
    except Exception as e:
        logger.error(f"An error occurred in vstores: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")


def retrieval(model_name: str, user_question: str, vstore_connection=None):
    response = run_llm(model_name=model_name, user_question=user_question, vstore_connection=vstore_connection)

    return response.content