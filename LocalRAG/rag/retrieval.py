from langchain import hub
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_llm(qdrant, query):
    llm = ChatOllama(model="llama3")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    custom_rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're an assistant。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = ({"context": qdrant.as_retriever() | format_docs, "input": RunnablePassthrough()}
                    | retrieval_qa_chat_prompt
                    | llm)

    chat_chain = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=lambda session_id: RedisChatMessageHistory(
        session_id, url="redis://localhost:6379"),
        input_message_key="input",
        history_message_key="chat_history",
    )
    

    # return chat_chain
    # return chat_chain.invoke("what is the name of the restaurant?", config = {"configurable": {"session_id": "cr7"}})
    return chat_chain.invoke({"input": query}, config = {"configurable": {"session_id": "cr7"}})
    # return chain.invoke("what is the name of the restaurant?")
