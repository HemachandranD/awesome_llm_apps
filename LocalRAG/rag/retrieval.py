import logging
from langchain import hub
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def run_llm(qdrant, query):
    logging.debug(f"Initial query: {query} (type: {type(query)})")
    
    llm = ChatOllama(model="llama3")
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    custom_rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer any use questions based solely on the context below:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    def custom_get_relevant_documents(query, retriever):
        assert isinstance(query, str)
        return retriever.get_relevant_documents(query)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = qdrant.as_retriever()
    
    custom_retriever = lambda query: custom_get_relevant_documents(query, retriever)

    context = itemgetter("input") | RunnableLambda(custom_retriever) | format_docs

    chain = RunnablePassthrough().assign(context=context) | custom_rag_prompt | llm

    chat_chain = RunnableWithMessageHistory(
        runnable= chain,
        get_session_history=lambda session_id: RedisChatMessageHistory(
            session_id, url="redis://localhost:6379"),
        input_message_key="input",
        history_message_key="chat_history",
    )

    logging.log("Invoking chat chain...")
    response = chat_chain.invoke(input={"input": query}, config={"configurable": {"session_id": "cr7"}})
    logging.log(f"Response: {response}")
    return response