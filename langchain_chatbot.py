import os
import pandas as pd

from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

# =====================================================
# Load Local Gemma Model
# =====================================================

llm = OllamaLLM(model="gemma")

# =====================================================
# Chatbot Chain
# =====================================================

template = """
You are a professional healthcare assistant.

Provide:
1. Explanation
2. Symptoms (if applicable)
3. Prevention Tips
4. Example

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

chain = prompt | llm

print("=" * 60)
print("HEALTHCARE CHATBOT USING LANGCHAIN + GEMMA")
print("Type 'exit' to stop the chatbot.")
print("=" * 60)

# =====================================================
# Continuous Chat Loop
# =====================================================

while True:

    question = input("\nEnter your healthcare question: ")

    if question.lower() == "exit":
        break

    response = chain.invoke(
        {
            "question": question
        }
    )

    print("\nResponse:\n")
    print(response)

    # Save chatbot responses
    df = pd.DataFrame([
        {
            "Question": question,
            "Response": response
        }
    ])

    if os.path.exists("dataset/chatbot_responses.csv"):
        df.to_csv(
            "dataset/chatbot_responses.csv",
            mode="a",
            header=False,
            index=False
        )
    else:
        df.to_csv(
            "dataset/chatbot_responses.csv",
            index=False
        )

    print("\nResponse saved successfully.")

print("\nChatbot testing completed.")

# =====================================================
# Summarization Chain
# =====================================================

summary_template = """
Summarize the following healthcare information:

{text}
"""

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=summary_template
)

summary_chain = summary_prompt | llm

summary_text = """
Artificial Intelligence helps doctors analyze medical images
faster and more accurately, supporting diagnosis and treatment.
"""

summary_output = summary_chain.invoke(
    {
        "text": summary_text
    }
)

print("\nSummary:\n")
print(summary_output)

# =====================================================
# Question Answering Chain
# =====================================================

qa_template = """
Answer the healthcare question.

Question:
{question}

Answer:
"""

qa_prompt = PromptTemplate(
    input_variables=["question"],
    template=qa_template
)

qa_chain = qa_prompt | llm

qa_output = qa_chain.invoke(
    {
        "question": "How does AI help doctors?"
    }
)

print("\nQA Output:\n")
print(qa_output)

# =====================================================
# Save Chain Results
# =====================================================

chain_results = pd.DataFrame([
    {
        "Chain_Type": "Summarization",
        "Input": summary_text,
        "Output": summary_output
    },
    {
        "Chain_Type": "Question Answering",
        "Input": "How does AI help doctors?",
        "Output": qa_output
    }
])

chain_results.to_csv(
    "dataset/chain_results.csv",
    index=False
)

print("\nChain results saved successfully.")