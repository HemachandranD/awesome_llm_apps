# LocalRAG

# INTRODUCTION

LocalRAG (Retrieval-Augmented Generation) is an advanced Local running LLM application designed to leverage the power of Language Models (LLMs) for enhanced information retrieval and generation. This project is part of a broader learning initiative in the fields of LLMOps and GenAI. Built using LangChain, LocalRAG aims to provide efficient, secure and accurate responses by combining retrieval mechanisms with generative AI.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/HemachandranD/awesome_llm_apps.git
   cd localrag

## Guide

This Guide will help you to deploy a Production level RAG application into your workstation/local machine or Server.
[Medium: LOCALRAG](https://medium.com/gopenai/deploy-a-production-grade-rag-chatbot-on-your-local-machine-or-server-localrag-9f6fdede6f54)

## Architecture

![alt text](docs/localrag.png)

## Folder Tree

Data catalog and data engineering work should be split into layers (and folder structure). Avoid putting all data engineering in an overly complex and dense script or function. Following the convention below will ensure data transformations are traceable, easily understood, and maintainable.

```text
...
├── rag
│   ├── loader       <-- Corpus Preparation and chunking
│   ├── vstores      <-- Vectorize the Corpus uisng vector stores
│   ├── retrieval    <-- Implement the Retrieval & Genearation Component 
│   ├── utils        <-- Utility Functions
├── localrag.py      <-- main file to orchestrate the RAG APplication.
|── app.py           <-- Streamlit Application file
...
```

## [Document Loaders](rag/loader.py)

Document loaders is used to load data from a source as Document's. A Document is a piece of text and associated metadata.


Here, the load_data definition takes the file as an input along with the file type and returns the data in list of Documents.

Next step is to split the entire data into chunks as the LLM’s are known for token limit they are proven to be performing better on the right number of Tokens. I have set the chunk_size=600 with an overlap of 50 tokens in order not to lose the context of the data.

## [Vector Stores](rag/vstores.py)

Document loaders is used to load data from a source as Document's. A Document is a piece of text and associated metadata.


Here, the load_data definition takes the file as an input along with the file type and returns the data in list of Documents.

Next step is to split the entire data into chunks as the LLM’s are known for token limit they are proven to be performing better on the right number of Tokens. I have set the chunk_size=600 with an overlap of 50 tokens in order not to lose the context of the data.