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
from langchain_core.output_parsers import StrOutputParser

# Creating an object
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def run_llm(model_name: str, user_question: str, vstore_connection):
    try:
        logger.info(f"****Setting up {model_name}, Please wait...****")
        llm = ChatOllama(model=model_name)

        logger.info("****Setting up RAG Prompt****")
        question_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer any user questions based solely on the context below:<context>\n\n{context}</context>"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        rephrase_prompt = PromptTemplate.from_template("""Given the following conversation and a follow up question, 
        rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\n
        Follow Up Input: {question}\nStandalone Question:""")

        logger.info("****Connecting to VectorStore****")
        vstore_connection = Qdrant.from_existing_collection(
        embedding=HuggingFaceEmbeddings(),
        collection_name="my_documents",
        url="http://localhost:6333")
        retriever= vstore_connection.as_retriever()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        logger.info("****Building the Chains with Chat History****")
        context = (rephrase_prompt | llm | StrOutputParser()) | retriever | format_docs
        chain = RunnablePassthrough.assign(context=context) | question_answer_prompt | llm
        # context = itemgetter("question") | retriever | format_docs
        # chain = RunnablePassthrough.assign(context=context) | question_answer_prompt | llm

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