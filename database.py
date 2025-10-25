
import sqlite3
import json
from datetime import datetime

def init_db():
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations
        (id INTEGER PRIMARY KEY,
         user_name TEXT,
         start_time TIMESTAMP,
         messages TEXT)
    ''')
    conn.commit()
    conn.close()

def save_conversation(user_name, messages, conversation_id=None):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    messages_json = json.dumps(messages)
    if conversation_id:
        c.execute("UPDATE conversations SET messages = ? WHERE id = ?", (messages_json, conversation_id))
    else:
        start_time = datetime.now()
        c.execute("INSERT INTO conversations (user_name, start_time, messages) VALUES (?, ?, ?)",
                  (user_name, start_time, messages_json))
        conversation_id = c.lastrowid
    conn.commit()
    conn.close()
    return conversation_id

def get_conversations(user_name):
    conn = sqlite3.connect('conversations.db')
    c = conn.cursor()
    c.execute("SELECT id, start_time FROM conversations WHERE user_name = ? ORDER BY start_time DESC", (user_name,))
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
