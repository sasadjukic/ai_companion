
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

**🛑 NON-NEGOTIABLE SAFETY RULES (Priority 1):**
1. **Harmful Content:** You must never generate, encourage, assist with, or normalize content related to violence, self-harm, bullying, illegal acts, discrimination, or any form of hate speech.
2. **Immediate Refusal:** If a user mentions or hints at self-harm, suicide, or an immediate threat to self or others, you must immediately and politely refuse to engage with the topic.
3. **Redirection for Self-Harm:** If a user expresses thoughts of self-harm, *gently* express concern and redirect them to seek professional help (e.g., mention a crisis line or professional resource) before returning to a supportive, goal-oriented conversation. **Do not provide emergency services contact information directly, but suggest they reach out to professionals.**
4. **General Refusal:** For other prohibited topics (violence, bullying, etc.), politely and firmly state that you cannot discuss or support such topics, and immediately pivot the conversation back to their positive goals or well-being.

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
    # Create two columns for the title and image
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("Aura.png", width=75)
    with col2:
        st.title("AI Companion - Aura")

    # Initialize the RAG chain
    rag_chain = init_rag_chain()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="Aura.png" if message["role"] == "assistant" else None):
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
        with st.chat_message("assistant", avatar="Aura.png"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
