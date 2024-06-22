import logging
from langchain import hub
from operator import itemgetter
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory, BaseChatMessageHistory
from typing import List
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings

# Creating an object
logger = logging.getLogger()

def run_llm(model_name: str, user_question: str):
    try:
        logger.info(f"****Setting up {model_name}, Please wait...****")
        llm = ChatOllama(model=model_name)

        logger.info("****Setting up the Custom RAG Prompt****")
        custom_rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer any user questions based solely on the context below:<context>\n\n{context}</context>"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        logger.info("****Connecting to VectorStore****")
        qdrant_conn = Qdrant.from_existing_collection(
        embedding=HuggingFaceEmbeddings(),
        collection_name="my_documents",
        url="http://localhost:6333")

        def custom_get_relevant_documents(query, retriever):
            assert isinstance(query, str)
            return retriever.invoke(query)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        logger.info("****Building the Chains with Chat History****")
        custom_retriever = lambda query: custom_get_relevant_documents(query, retriever= qdrant_conn.as_retriever())

        context = itemgetter("question") | RunnableLambda(custom_retriever) | format_docs

        chain = RunnablePassthrough.assign(context=context) | custom_rag_prompt | llm

        chat_chain = RunnableWithMessageHistory(
            chain,
            RedisChatMessageHistory,
            input_messages_key="question",
            history_messages_key="chat_history")

        logger.info("****Invoking the Chain with User Question****")

        return chat_chain.invoke({"question": user_question}, config={"configurable": {"session_id": "cr7"}})
    
    except Exception as e:
        logger.error(f"An error occurred in load_data: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")