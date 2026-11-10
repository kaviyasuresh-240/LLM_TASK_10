# Week 10 – LangChain Basics

## Project Overview

This project demonstrates the integration of LangChain with the local Gemma Large Language Model (LLM) through Ollama. A healthcare chatbot was developed using prompt templates and chains to generate structured responses for healthcare-related queries.

## Objectives

- Understand the basics of LangChain.
- Create reusable prompt templates.
- Build and execute LangChain chains.
- Integrate Ollama and the Gemma model.
- Implement summarization and question-answering chains.

## Technologies Used

- Python
- LangChain
- Ollama
- Gemma
- Pandas
- Jupyter Notebook

## Project Structure

```text
LangChain_Basics/

├── dataset/
│   ├── prompts.csv
│   ├── user_queries.csv
│   ├── chatbot_responses.csv
│   └── chain_results.csv
│
├── notebook/
│   └── Task10_LangChain_Basics.ipynb
│
├── langchain_chatbot.py
├── README.md
└── requirements.txt
```

## Features

- Healthcare chatbot using LangChain and Gemma.
- Five prompt templates for healthcare queries.
- Dynamic user input handling.
- Summarization chain.
- Question Answering (QA) chain.
- Automatic storage of chatbot responses.
- Automatic generation of chain results.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Project

Execute the chatbot:

```bash
python langchain_chatbot.py
```

## Generated Output Files

The following files are generated automatically during execution:

- chatbot_responses.csv
- chain_results.csv

## Future Enhancements

- Integrate Retrieval-Augmented Generation (RAG).
- Add conversation memory.
- Connect to vector databases such as ChromaDB.
- Develop a Streamlit-based user interface.
- Deploy the chatbot as a web application.

## Conclusion

This project successfully demonstrates the use of LangChain with Ollama and Gemma for building a healthcare chatbot. Prompt templates and chains improve the modularity, reusability, and scalability of LLM applications.