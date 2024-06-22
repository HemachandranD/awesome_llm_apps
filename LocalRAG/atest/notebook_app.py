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

loader = loader = TextLoader(os.getcwd()+"/awesome_llm_apps/LocalRAG/yelp_review.txt")
data = loader.load()
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(data)
embeddings = HuggingFaceEmbeddings()
db = Chroma.from_documents(docs, embeddings, persist_directory= "./genai_guide")

custom_rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer any use questions based solely on the context below:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

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

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def custom_retriever(query, retriever):
    assert isinstance(query, str)
    return retriever.get_relevant_documents(query)

retriever = db.as_retriever()
# Create a lambda to pass db to the custom_retriever
custom_retriever_with_db = lambda query: custom_retriever(query, retriever)
context = itemgetter("input") | RunnableLambda(custom_retriever_with_db) | format_docs
first_step = RunnablePassthrough.assign(context=context)
chain = first_step | custom_rag_prompt | ChatOllama(model="llama3")
chain_with_history = RunnableWithMessageHistory(
    chain,
    # Uses the get_by_session_id function defined in the example
    # above.
    get_by_session_id,
    input_messages_key="input",
    history_messages_key="chat_history",
)
print("****Generating from llama3****")
print(chain_with_history.invoke(  # noqa: T201
    {"input": user_question},
    config={"configurable": {"session_id": "foo"}}
))