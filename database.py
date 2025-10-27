
import sqlite3
import json
from datetime import datetime

def init_db():
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users
        (id INTEGER PRIMARY KEY,
         name TEXT NOT NULL,
         email TEXT NOT NULL UNIQUE,
         password TEXT NOT NULL,
         interests TEXT)
    ''')
    # Create conversations table
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations
        (id INTEGER PRIMARY KEY,
         user_id INTEGER,
         start_time TIMESTAMP,
         messages TEXT,
         FOREIGN KEY (user_id) REFERENCES users (id))
    ''')
    conn.commit()
    conn.close()

def create_user(name, email, password, interests):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (name, email, password, interests) VALUES (?, ?, ?, ?)",
                  (name, email, password, interests))
        user_id = c.lastrowid
        conn.commit()
        return user_id
    except sqlite3.IntegrityError:
        return None  # Email already exists
    finally:
        conn.close()

def get_user_by_email(email):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    return user

def save_conversation(user_id, messages, conversation_id=None):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    messages_json = json.dumps(messages)
    if conversation_id:
        c.execute("UPDATE conversations SET messages = ? WHERE id = ?", (messages_json, conversation_id))
    else:
        start_time = datetime.now()
        c.execute("INSERT INTO conversations (user_id, start_time, messages) VALUES (?, ?, ?)",
                  (user_id, start_time, messages_json))
        conversation_id = c.lastrowid
    conn.commit()
    conn.close()
    return conversation_id

def get_conversations(user_id):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("SELECT id, start_time FROM conversations WHERE user_id = ? ORDER BY start_time DESC", (user_id,))
    conversations = []
    for conv_id, start_time_str in c.fetchall():
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S.%f')
        formatted_time = start_time.strftime('%Y-%m-%d %H:%M')
        conversations.append((conv_id, formatted_time))
    conn.close()
    return conversations

def get_conversation(conversation_id):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("SELECT messages FROM conversations WHERE id = ?", (conversation_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return json.loads(result[0])
    return None
