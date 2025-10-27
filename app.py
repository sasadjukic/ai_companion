
import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
import bcrypt
from database import (
    init_db, save_conversation, get_conversations,
    get_conversation, create_user, get_user_by_email
)

# Function to initialize the RAG chain
@st.cache_resource
def init_rag_chain(user_interests):
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
    # loader = TextLoader('user_data.txt') # No longer needed
    # documents = loader.load() # No longer needed
    documents = [Document(page_content=user_interests, metadata={"source": "user_profile"})]

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')

    # Create a Chroma vector store
    vectorstore = Chroma.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # Build the RAG Pipeline
    prompt_template = PromptTemplate.from_template(
        system_prompt + '''

History: {history}

Context: {context}

User: {question}
'''
    )

    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return rag_chain

def format_chat_history(chat_history):
    """Formats chat history into a string."""
    buffer = ""
    for message in chat_history:
        if message["role"] == "user":
            buffer += "Human: " + message["content"] + "\n"
        elif message["role"] == "assistant":
            buffer += "AI: " + message["content"] + "\n"
    return buffer

def login_signup_page():
    st.header("Welcome to Aura")
    
    choice = st.radio("Choose an option", ('Sign In', 'Sign Up'))

    if choice == 'Sign In':
        st.subheader("Sign In")
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In")
            if submitted:
                user = get_user_by_email(email)
                if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
                    st.session_state.logged_in = True
                    st.session_state.user_id = user[0]
                    st.session_state.user_name = user[1]
                    st.session_state.user_interests = user[4] # Store interests in session state
                    st.rerun()
                else:
                    st.error("Invalid email or password")

    elif choice == 'Sign Up':
        st.subheader("Sign Up")
        with st.form("signup_form"):
            name = st.text_input("First Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            interests = st.text_area("Your interests and goals")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if name and email and password and interests:
                    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                    user_id = create_user(name, email, hashed_password, interests)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.session_state.user_name = name
                        st.session_state.user_interests = interests # Store interests in session state
                        st.rerun()
                    else:
                        st.error("Email already exists.")
                else:
                    st.error("Please fill out all fields.")

def main():
    init_db()

    # Create two columns for the title and image
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("Aura.png", width=75)
    with col2:
        st.title("AI Companion - Aura")

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_signup_page()
    else:
        st.sidebar.title("History")
        if st.sidebar.button("New Chat"):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

        conversations = get_conversations(st.session_state.user_id)
        for conv in conversations:
            if st.sidebar.button(f"Chat from {conv[1]}"):
                st.session_state.messages = get_conversation(conv[0])
                st.session_state.conversation_id = conv[0]
                st.rerun()

        # Add a sign out button to the sidebar
        if st.sidebar.button("Sign Out"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.user_name = None
            st.session_state.user_interests = None
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

        # Initialize the RAG chain with user interests
        rag_chain = init_rag_chain(st.session_state.user_interests)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.conversation_id = None

        # Greet the user by name
        if not st.session_state.messages:
            greeting = f"Hello {st.session_state.user_name}! It's great to see you again. What's on your mind today?"
            st.session_state.messages.append({"role": "assistant", "content": greeting})
            st.session_state.conversation_id = save_conversation(st.session_state.user_id, st.session_state.messages)

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
            chat_history_str = format_chat_history(st.session_state.messages)
            response = rag_chain.invoke({"question": prompt, "history": chat_history_str})
            
            # Display assistant response in chat message container
            with st.chat_message("assistant", avatar="Aura.png"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Save the updated conversation
            save_conversation(st.session_state.user_id, st.session_state.messages, st.session_state.conversation_id)

if __name__ == '__main__':
    main()
