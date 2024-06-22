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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory, BaseChatMessageHistory
from langchain_core.pydantic_v1 import BaseModel, Field
from operator import itemgetter
from typing import List
from langchain_core.messages import BaseMessage, AIMessage

user_question = "What is the name of the restaurant?"

llm = ChatOllama(model="llama3")
loader = loader = TextLoader(os.getcwd()+"/awesome_llm_apps/LocalRAG/yelp_review.txt")
data = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings()

custom_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer any use questions based solely on the context below:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def custom_get_relevant_documents(query, retriever):
    assert isinstance(query, str)
    return retriever.get_relevant_documents(query)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

# qdrant = Qdrant.from_existing_collection(
# embedding=HuggingFaceEmbeddings(),
# collection_name="my_documents",
# url="http://localhost:6333")
# db = Chroma.from_documents(docs, embeddings, persist_directory= "./importlogging")
qdrant = Qdrant.from_documents(
    docs,
    embeddings,
    url="http://localhost:6333",
    collection_name="my_documents",
    force_recreate=True,
)
custom_retriever = lambda query: custom_get_relevant_documents(query, retriever= qdrant.as_retriever())

context = itemgetter("question") | RunnableLambda(custom_retriever) | format_docs

chain = RunnablePassthrough.assign(context=context) | custom_rag_prompt | llm

# chat_chain = RunnableWithMessageHistory(
#     runnable= chain,
#     get_session_history=lambda session_id : RedisChatMessageHistory(
#     session_id, url="redis://localhost:6379"
#     ),
#     input_message_key="question",
#     history_message_key="chat_history",
# )
chain_with_history = RunnableWithMessageHistory(
    chain,
    RedisChatMessageHistory,
    input_messages_key="question",
    history_messages_key="chat_history",
)
print("****Generating from llama3****")
print(chain_with_history.invoke({"question": user_question}, config={"configurable": {"session_id": "cr7"}}))