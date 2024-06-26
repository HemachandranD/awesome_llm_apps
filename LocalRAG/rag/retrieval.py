import logging

import streamlit as st
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant

# Creating an object
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def run_llm(model_name: str, user_question: str, session_id: str, vstore_connection):
    try:
        logger.info(f"****Setting up {model_name}, Please wait...****")
        llm = ChatOllama(model=model_name)

        logger.info("****Setting up RAG Prompt****")
        question_answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Answer any user questions based solely on the context below:<context>\n\n{context}</context>
                    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        rephrase_prompt = PromptTemplate.from_template(
            """Given the following conversation and a follow up question,
        rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\n
        Follow Up Input: {question}\nStandalone Question:"""
        )

        logger.info("****Connecting to VectorStore****")
        vstore_connection = Qdrant.from_existing_collection(
            embedding=HuggingFaceEmbeddings(),
            collection_name="my_documents",
            url="http://localhost:6333",
        )
        retriever = vstore_connection.as_retriever()

        def get_sources(docs):
            print(docs)
            return [", ".join(doc.metadata['source'] for doc in docs)]

        def format_docs(docs):
            print(docs)
            get_sources(docs)
            return "\n\n".join(doc.page_content for doc in docs)

        logger.info("****Building the Chains with Chat History****")
        # source_documents = (rephrase_prompt | llm | StrOutputParser() | retriever)
        retrieved_docs = (rephrase_prompt | llm | StrOutputParser()) | retriever | format_docs
        chain = (
            RunnablePassthrough.assign(context=retrieved_docs) | question_answer_prompt | llm
        )
        # context = itemgetter("question") | retriever | format_docs
        # chain = RunnablePassthrough.assign(context=context) | question_answer_prompt | llm

        # logger.info("****Getting Sources****")
        # sources = RunnablePassthrough.assign(sources=source_documents) | get_sources
        # sources.invoke({"question": user_question, "chat_history": ""})

        chat_chain = RunnableWithMessageHistory(
            chain,
            RedisChatMessageHistory,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        st.info(session_id)
        logger.info("****Invoking the Chain with User Question****")
        return chat_chain.invoke(
            {"question": user_question}, config={"configurable": {"session_id": session_id}}
        )

    except Exception as e:
        logger.error(f"An error occurred in run_llm: {str(e)}")
        raise SystemExit(f"Exiting due to the error: {str(e)}")
