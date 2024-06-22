import logging
from langchain import hub
from operator import itemgetter
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory, BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from langchain_qdrant import Qdrant
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def run_llm(embeddings, user_question):
    
    llm = ChatOllama(model="llama3")

    custom_rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer any use questions based solely on the context below:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def custom_get_relevant_documents(query, retriever):
        assert isinstance(query, str)
        return retriever.invoke(query)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qdrant = Qdrant.from_existing_collection(
    embedding=embeddings,
    collection_name="my_documents",
    url="http://localhost:6333")

    custom_retriever = lambda query: custom_get_relevant_documents(query, retriever= qdrant.as_retriever())

    context = itemgetter("question") | RunnableLambda(custom_retriever) | format_docs

    chain = RunnablePassthrough.assign(context=context) | custom_rag_prompt | llm

    chat_chain = RunnableWithMessageHistory(
        chain,
        RedisChatMessageHistory,
        input_messages_key="question",
        history_messages_key="chat_history")

    return chat_chain.invoke({"question": user_question}, config={"configurable": {"session_id": "cr7"}})