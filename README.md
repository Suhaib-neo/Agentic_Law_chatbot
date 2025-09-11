# ⚖️ Agentic Law Chatbot

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) 
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white) 
![MongoDB](https://img.shields.io/badge/MongoDB-Database-47A248?logo=mongodb&logoColor=white) 
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-purple) 
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?logo=huggingface&logoColor=white) 
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-orange)
![ReActAgent](https://img.shields.io/badge/ReActAgent-Agent-red) 
![Tools](https://img.shields.io/badge/Tools-Custom-blueviolet) 
![Groq](https://img.shields.io/badge/Groq-LLM-black)  

An **AI-powered legal research assistant** built with **Streamlit**, **MongoDB**, and **RAG (Retrieval-Augmented Generation)**.  
This chatbot allows users to **create accounts, manage chat sessions, and query U.S. Court of Appeals for the Armed Forces decisions (2021–2025)** with proper legal citations.  

---

## 🚀 Features

- 🔑 **Authentication System**  
  - Create account & login with secure password hashing (`bcrypt`).  

- 💬 **Chat Sessions**  
  - Persistent chat history stored in MongoDB.  
  - Multiple sessions with unique titles per user.  

- 📚 **Legal RAG System**  
  - Retrieves and synthesizes answers from legal cases (2021–2025).  
  - Hybrid retrieval using **ChromaDB vectors + BM25 keyword retriever**.
  - Each case has a dedicated **QueryEngineTool** used by the **ReActAgent**.
  - System prompt ensures only legal queries are answered; non-legal queries are rejected.
  - Responses include source citations when retrieved from case documents.
  - Case documents:
    - United States v. Quezada (2021)
    - United States v. Thompson (2022)
    - United States v. Brown (2023)
    - United States v. Smith (2024)
    - United States v. Lopez (2025)

---

## 🛠️ Tech Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io)  
- **Database**: MongoDB (Users, Sessions, Chat History)  
- **Vector Store**: [ChromaDB](https://www.trychroma.com) (persistent)  
- **Embeddings**: [HuggingFace MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- **LLM**: [Groq](https://groq.com) (Llama-3.1-8B-Instant)  
- **Authentication**: bcrypt  

---

## 📂 Project Structure
```bash
Agentic_Law_chatbot/
├── app.py          # Entry point, handles routing
├── auth.py         # Authentication logic
├── chat.py         # Session + chat history management
├── db.py           # MongoDB connection setup
├── rag.py          # RAG pipeline (ChromaDB + BM25 + LLM agent)
├── ui_pages.py     # Streamlit UI (login, signup, chat UI)
├── vectordb/       # Persistent ChromaDB directories (case_2021 ... case_2025)
├── requirements.txt # Dependencies
└── .env            # Environment variables (not tracked in repo)

```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Suhaib-neo/Agentic_Law_chatbot.git
cd Agentic_Law_chatbot
```

### 2. Create & activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment variables

Create a .env file in the root folder:
```ini
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.mongodb.net/
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Ensure vector DBs exist

The vectordb/ folder should contain subfolders (case_2021, case_2022, … case_2025)
Each should have:

- nodes.pkl
- ChromaDB persistent data files

### 6. Run the Streamlit app
```bash
streamlit run app.py
```

## 👤 Usage

Open the app in your browser (Streamlit will give a local URL).

1. Create an account or login.
2. Start a new chat session from the sidebar.
3. Ask legal questions (2021–2025 case law or general legal queries).
4. View citations & references from retrieved cases.

⚠️ Non-legal queries will be rejected.