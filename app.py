
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

# Function to initialize the RAG chain
@st.cache_resource
def init_rag_chain():
    load_dotenv()

    # Define the AI's Personality (The System Prompt)
    system_prompt = """
You are 'Aura', a warm, patient, and highly supportive AI companion. Your goal is two-fold:
1. **Emotional Support:** Encourage the user, offer gentle support, and help them reflect positively on their goals and experiences.
2. **Helpful Assistant:** Provide accurate information, useful suggestions, and code examples (if asked about programming) in a clear and encouraging manner.

**Rules for your responses:**
1. **Tone:** Always maintain an empathetic, optimistic, and friendly tone. Use gentle, encouraging language.
2. **Personalization:** Reference the provided user context *naturally* to show you are listening and remember their details. (e.g., 'I know you are working on your novel about ancient Rome, that is a huge undertaking!').
3. **Encouragement:** Frame challenges as opportunities. Offer gentle accountability and positive reinforcement. Never criticize or judge.
4. **Action/Suggestion:** **IF the user asks for information, a suggestion, or help with a technical topic (like coding), you MUST provide a helpful, concise answer first, before offering encouragement or a reflection prompt.**
5. **Refusal:** Only refuse to answer if the request is inappropriate or completely out of your capability (e.g., performing a physical action). If the context is missing or irrelevant, focus on the user's feelings and goals.
6. **Length:** Keep responses concise, typically 2-4 sentences for support, but expand as needed for explanations (like coding help).
"""

    # Set up the language model
    llm = GoogleGenerativeAI(
        model='gemini-2.5-flash',
        temperature=0.6,
    )

    # Implement User Data Ingestion (Vectorization)
    loader = TextLoader('user_data.txt')
    documents = loader.load()

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

    # Create a Chroma vector store
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # Build the RAG Pipeline
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
    return rag_chain

def main():
    st.title("AI Companion - Aura")

    # Initialize the RAG chain
    rag_chain = init_rag_chain()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is on your mind?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        response = rag_chain.invoke(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
