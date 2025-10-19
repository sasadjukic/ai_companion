# AI Companion - "Aura"

## Overview

Aura is a warm, patient, and highly supportive AI companion built with Python. It leverages a Retrieval-Augmented Generation (RAG) pipeline to provide personalized and context-aware encouragement. By understanding the user's goals and challenges from a simple text file, Aura offers gentle support and helps users reflect positively on their experiences.

This project uses the following core technologies:
- **Programming Language:** Python
- **Core Framework:** LangChain
- **Language Model (LLM):** Google Gemini (but you can set your own LLM)
- **Vector Database:** ChromaDB
- **Embedding Model:** Hugging Face `all-mpnet-base-v2`

---

## Features

- **Context-Aware Conversations:** Aura remembers user-specific details from a data file to personalize interactions.
- **Supportive Personality:** A carefully crafted system prompt ensures Aura always maintains an empathetic, optimistic, and friendly tone.
- **Retrieval-Augmented Generation (RAG):** The system retrieves relevant user data to construct a rich prompt, enabling the LLM to generate highly relevant and personalized responses.
- **Simple Chat Interface:** Interact with Aura directly from your terminal.

---

## Getting Started

Follow these instructions to set up and run Aura on your local machine.

### Prerequisites

- Python 3.8+
- `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables:**
    - Create a file named `.env` in the project root.
    - Add your Google API key to this file:
      ```
      GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"
      ```

4.  **Add your personal context:**
    - Open the `user_data.txt` file.
    - Add sentences that describe your goals, interests, or any other context you want Aura to be aware of. Each piece of information should ideally be on a new line.

### Running the Application

Once the setup is complete, you can start the chat interface with the following command:

```bash
streamlit run app.py
```

You can now chat with Aura in your terminal. To end the session, type `exit`.

---

## How It Works

Aura is built on a RAG (Retrieval-Augmented Generation) architecture, which involves the following steps:

1.  **Data Ingestion:** The text from `user_data.txt` is loaded and split into chunks.
2.  **Embedding:** Each chunk of text is converted into a numerical vector using the `all-mpnet-base-v2` embedding model. These vectors capture the semantic meaning of the text.
3.  **Vector Storage:** The embeddings and their corresponding text chunks are stored in a ChromaDB vector database.
4.  **Retrieval:** When you send a message, your input is converted into a query vector. The system performs a similarity search in the vector database to find the most relevant text chunks from your user data.
5.  **Prompt Construction:** The retrieved text chunks are dynamically inserted into a prompt template, which also includes the system prompt (Aura's personality) and your message.
6.  **Response Generation:** This complete prompt is sent to the Gemini LLM, which generates a personalized, supportive, and context-aware response.

---

## Project Structure

```
.
├── .env                # Stores environment variables like API keys
├── app.py             # The main application script
├── README.md           # Project documentation
├── requirements.txt    # Python package dependencies
└── user_data.txt       # User-specific context for the AI
```
