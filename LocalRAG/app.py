from rag.loader import load_data, prepare_chunk
from rag.retrieval import run_llm
from rag.vstores import create_vstores
from langchain_qdrant import Qdrant

query = "What is the name of the restaurant?"

if __name__ == "__main__":
    data = load_data(
        "Text", "/teamspace/studios/this_studio/awesome_llm_apps/LocalRAG/yelp.txt"
    )
    documents = prepare_chunk(data)
    qdrant, embeddings = create_vstores(documents)
    response = run_llm(qdrant, query)
    print(response)
    # print(qdrant.similarity_search("what is the name of the restaurant?"))
