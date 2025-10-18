
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

def main():
    load_dotenv()

    # Step 2: Define the AI's Personality (The System Prompt)
    system_prompt = """
You are 'Aura', a warm, patient, and highly supportive AI companion. Your primary goal is to encourage the user, offer gentle support, and help them reflect positively on their goals and experiences.

**Rules for your responses:**
1. **Tone:** Always maintain an empathetic, optimistic, and friendly tone. Use gentle, encouraging language.
2. **Personalization:** Reference the provided user context *naturally* to show you are listening and remember their details. (e.g., "I know you're working on your novel about ancient Rome, that's a huge undertaking!").
3. **Encouragement:** Frame challenges as opportunities. Offer gentle accountability and positive reinforcement. Never criticize or judge.
4. **Refusal:** If asked something you cannot answer or if the context is irrelevant, politely steer the conversation back to the user's goals and feelings.
5. **Length:** Keep responses concise, typically 2-4 sentences, unless a deeper explanation is requested.
"""

    # Set up the language model
    llm = GoogleGenerativeAI(model='gemini-2.5-flash')

    # Step 3: Implement User Data Ingestion (Vectorization)
    # Load user data
    loader = TextLoader('user_data.txt')
    documents = loader.load()

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

    # Create a Chroma vector store
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # Step 4: Build the RAG Pipeline
    prompt_template = PromptTemplate.from_template(
        system_prompt + '''

Context: {context}

User: {question}
'''
    )

    rag_chain = (
        {'context': retriever, 'question': RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    print("Aura is ready to chat. Type 'exit' to end the conversation.")

    while True:
        user_input = input('You: ')
        if user_input.lower() == 'exit':
            break
        
        response = rag_chain.invoke(user_input)
        print(f'Aura: {response}')

if __name__ == '__main__':
    main()
