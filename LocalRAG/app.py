from rag.loader import load_data, prepare_chunk
from rag.retrieval import run_llm
from rag.vstores import create_vstores
from langchain_qdrant import Qdrant
import logging

# Creating an object
logger = logging.getLogger()

question = "It is used for"

if __name__ == "__main__":
    data = load_data(
        "Text", "/teamspace/studios/this_studio/awesome_llm_apps/LocalRAG/langgraph.txt"
    )
    documents = prepare_chunk(data)
    vs_conn = create_vstores(documents)
    response = run_llm(model_name="llama3", user_question=question, vstore_connection=None)
    type(response)
    print(response.content)
    
