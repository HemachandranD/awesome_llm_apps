from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_qdrant import Qdrant


def run_llm(qdrant, query):
    llm = ChatOllama(model="llama3")

    retreiver = qdrant.as_retriever()
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    return chain.invoke()
