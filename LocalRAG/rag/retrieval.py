from langchain import hub
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def run_llm(qdrant, query):
    llm = ChatOllama(model="llama3")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    custom_rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're an assistant。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = ({"context": qdrant.as_retriever(), "input": RunnablePassthrough()}
                    | custom_rag_prompt
                    | llm)

    chat_chain = RunnableWithMessageHistory(
        runnable=chain,
        get_session_history=lambda session_id: RedisChatMessageHistory(
            session_id=session_id, url="redis://localhost:6379"
        ),
        input_message_key="question",
        history_message_key="chat_history",
    )
    config = {"configurable": {"session_id": "cr7"}}

    return chat_chain.invoke({"question": "what is the name of the restaurant?"}, config=config)
    # return chain.invoke("what is the name of the restaurant?")
