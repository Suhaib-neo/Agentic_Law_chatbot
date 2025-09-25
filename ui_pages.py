import os
import time
import streamlit as st
import re
import asyncio
import nest_asyncio

from auth import check_login
from chat import load_user_sessions, load_chat_history, save_message
from rag import setup_rag_system
from db import init_connection

nest_asyncio.apply()
from datasets import Dataset
from ragas import evaluate
from dotenv import load_dotenv
from ragas.metrics import faithfulness, LLMContextPrecisionWithoutReference
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import Embeddings

load_dotenv()  # load GROQ_API_KEY from .env

# ---- LLM Setup ----
groq_llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
llm_wrapper = LangchainLLMWrapper(langchain_llm=groq_llm)

# ---- Embeddings Setup ----
class CustomRagasEmbeddings(Embeddings):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts):
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text):
        return self.embeddings.embed_query(text)

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings_model = CustomRagasEmbeddings(embeddings=hf_embeddings)


def login_page():
    # Custom CSS for login page
    st.markdown("""
    <style>
    .login-container {
        max-width: 450px;
        margin: 2rem auto;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .login-title {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .login-caption {
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="login-container">
        <h1 class="login-title">Legal Case RAG</h1>
        <p class="login-caption">Ask me about legal cases (2021–2025). I'll retrieve documents and give citations.</p>
    </div>
    """, unsafe_allow_html=True)

    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### Login to Continue")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_button = st.form_submit_button("🚀 Login", use_container_width=True)

            if login_button:
                db_conn = init_connection()
                users_collection = db_conn["users"]

                if check_login(username, password, users_collection):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"Welcome back, {username}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        st.markdown("---")
        if st.button("Create an account", use_container_width=True):
            st.session_state.show_create_account = True
            st.rerun()


def create_account_page():
    # Custom CSS for create account page
    st.markdown("""
    <style>
    .signup-container {
        max-width: 450px;
        margin: 2rem auto;
        padding: 2rem;
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .signup-title {
        color: white;
        text-align: center;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .signup-caption {
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="signup-container">
        <h1 class="signup-title">Create Account</h1>
        <p class="signup-caption">Set up your account to start chatting with the legal RAG system.</p>
    </div>
    """, unsafe_allow_html=True)

    db_conn = init_connection()
    users_collection = db_conn["users"]

    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("create_account_form"):
            st.markdown("### Account Details")
            new_username = st.text_input("Choose a Username", placeholder="Enter desired username")
            new_password = st.text_input("Choose a Password", type="password", placeholder="Enter secure password")
            create_button = st.form_submit_button("Create Account", use_container_width=True)

            if create_button:
                import bcrypt
                if not new_username or not new_password:
                    st.error("Username and password cannot be empty.")
                elif users_collection.find_one({"username": new_username}):
                    st.error("Username already exists.")
                else:
                    salt = bcrypt.gensalt()
                    hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), salt)

                    users_collection.insert_one({
                        "username": new_username,
                        "password": hashed_password.decode('utf-8')
                    })
                    st.success("Account created successfully! Please log in.")
                    st.session_state.show_create_account = False
                    st.rerun()

        if st.button("Back to Login", use_container_width=True):
            st.session_state.show_create_account = False
            st.rerun()


def main_page():
    # Enhanced CSS for chat interface
    st.markdown("""
    <style>
    /* Main chat container */
    .chat-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* User message styling */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        position: relative;
    }
    
    .user-bubble::after {
        content: '';
        position: absolute;
        bottom: 0;
        right: -8px;
        width: 0;
        height: 0;
        border: 8px solid transparent;
        border-top-color: #764ba2;
        border-bottom: 0;
        margin-left: -8px;
        margin-bottom: -8px;
    }
    
    /* Assistant message styling */
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1rem;
    }
    
    /* Assistant message styling */
    .assistant-bubble {
        background: linear-gradient(135deg, #06beb6 0%, #48b1bf 100%) !important;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        box-shadow: 0 2px 10px rgba(72, 177, 191, 0.3);
        position: relative;
    }
    
    .assistant-bubble::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: -8px;
        width: 0;
        height: 0;
        border: 8px solid transparent;
        border-top-color: #48b1bf;
        border-bottom: 0;
        margin-right: -8px;
        margin-bottom: -8px;
    }
    
    /* Avatar styling */
    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        margin-left: 10px;
        margin-top: 5px;
    }
    
    .assistant-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #06beb6 0%, #48b1bf 100%) !important;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        margin-right: 10px;
        margin-top: 5px;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Session button styling */
    .session-btn {
        width: 100%;
        margin-bottom: 5px;
        padding: 8px;
        border-radius: 8px;
        border: none;
        background: #f0f2f6;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .session-btn:hover {
        background: #e0e0e0;
        transform: translateY(-2px);
    }
    
    /* Welcome message */
    .welcome-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        text-align: center;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Chat title */
    .chat-title {
        background: linear-gradient(135deg, #06beb6 0%, #48b1bf 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Source metadata styling */
    .source-metadata {
        background: rgba(72, 177, 191, 0.1);
        border-left: 4px solid #48b1bf;
        padding: 10px;
        margin-top: 10px;
        border-radius: 5px;
    }
    """, unsafe_allow_html=True)

    # Initialize chat memory for RAG agent
    if 'chat_memory' not in st.session_state:
        st.session_state['chat_memory'] = []

    if 'agent' not in st.session_state:
        with st.spinner("🔄 Loading legal case documents..."):
            agent, llm = setup_rag_system()
            st.session_state.agent = agent
            st.session_state.llm = llm

    # Enhanced sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>Chat Sessions</h3>
        </div>
        """, unsafe_allow_html=True)

        db_conn = init_connection()
        sessions_collection = db_conn["sessions"]
        messages_collection = db_conn["messages"]

        if 'chat_sessions' not in st.session_state or not st.session_state.chat_sessions:
            sessions, current, messages = load_user_sessions(
                st.session_state.username,
                sessions_collection,
                last_session_id=st.session_state.get("last_session_id")
            )
            st.session_state.chat_sessions = sessions
            st.session_state.current_chat_session = current
            if current:
                st.session_state.messages = load_chat_history(str(current["_id"]), messages_collection)
            else:
                st.session_state.messages = []

        # Create new session form
        with st.form("new_chat_form"):
            st.session_state.chat_memory = []
            st.markdown("### Start New Chat")
            session_name = st.text_input("Chat name:", placeholder="e.g., Contract Law Questions", key="new_chat_name")
            if st.form_submit_button("Create", use_container_width=True):
                session_name = session_name.strip()
                if not session_name:
                    st.error("Chat name cannot be empty.")
                else:
                    normalized = session_name.lower()
                    existing = sessions_collection.find_one({
                        "username": st.session_state.username,
                        "session_name_normalized": normalized
                    })
                    if existing:
                        st.error(f"You already have a chat named '{existing['session_name']}'. Please choose a different name.")
                    else:
                        new_session = {
                            "username": st.session_state.username,
                            "session_name": session_name,
                            "timestamp": time.time(),
                            "session_name_normalized": normalized
                        }
                        inserted_id = sessions_collection.insert_one(new_session).inserted_id
                        st.session_state.chat_sessions.append({
                            "_id": inserted_id,
                            "session_name": session_name,
                            "username": st.session_state.username,
                            "timestamp": new_session["timestamp"]
                        })
                        st.session_state.current_chat_session = st.session_state.chat_sessions[-1]
                        st.session_state.messages = []
                        st.session_state.last_session_id = str(inserted_id)
                        st.success(f"New chat '{session_name}' created!")
                        time.sleep(1)
                        st.rerun()
                        

        # Previous sessions
        st.markdown("---")
        st.markdown("### Previous Sessions")
        for session in st.session_state.chat_sessions:
            if st.button(f"{session['session_name']}", key=str(session["_id"]), use_container_width=True):
                st.session_state.current_chat_session = session
                st.session_state.messages = load_chat_history(str(session["_id"]), messages_collection)
                st.session_state.last_session_id = str(session["_id"])
                st.rerun()

        # Logout button
        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            keys_to_clear = ['logged_in', 'username', 'current_chat_session', 'chat_sessions', 'messages', 'agent', 'llm']
            for key in keys_to_clear:
                st.session_state.pop(key, None)
            st.rerun()
        
        st.markdown("---")
        # Instructions expander
        with st.expander("ℹ️ How to use this chatbot", expanded=False):
            st.markdown("""
            **How it works:**
            1. Type your **legal question** in the chat box below  
            2. I'll retrieve relevant cases from **2021–2025** and respond with context + citations  
            3. Use the sidebar to **start a new chat** or switch between past sessions  
            4. Chat titles are what you enter when creating a session  
            5. Use **Logout** in the sidebar to securely end your session
            
            **Tips for better results:**
            - Be specific about the legal area (e.g., "contract law", "criminal procedure")
            - Include relevant case names if you know them
            - Ask follow-up questions to dive deeper into specific aspects
            """)

    # Main content area
    if st.session_state.current_chat_session:
        # Welcome header
        st.markdown(f"""
        <div style="padding: 1rem; border: 0px solid #ddd; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
            <h2 style="margin-bottom: 0.5rem;">Legal Case RAG Chatbot</h2>
            <p style="margin: 0;">Welcome, <strong>{st.session_state.username}</strong>! Create a new chat session to get started.</p>
        </div>
        """, unsafe_allow_html=True)

        # Current chat title
        # Current chat title (minimalist text)
        st.markdown(
            f"<p style='font-size: 1.1rem;'>You are opened <strong>{st.session_state.current_chat_session['session_name']}</strong> chat!</p>",
            unsafe_allow_html=True
        )


        # Chat messages container
        chat_container = st.container()
        
        with chat_container:
            # Display previous messages with enhanced styling
            for i, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="user-message">
                        <div class="user-bubble">
                            {message["content"]}
                        </div>
                        <div class="user-avatar">
                            {st.session_state.username[0].upper()}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="assistant-message">
                        <div class="assistant-avatar">
                            🤖
                        </div>
                        <div class="assistant-bubble">
                            {message["content"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask a question about the cases...", key="chat_input"):
            # Display user message immediately
            st.markdown(f"""
            <div class="user-message">
                <div class="user-bubble">
                    {prompt}
                </div>
                <div class="user-avatar">
                    {st.session_state.username[0].upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.chat_memory.append({"role": "user", "content": prompt})

            # Get last 4 messages for context
            recent_messages = st.session_state.chat_memory[-4:]
            context_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent_messages])
            prompt_for_agent = st.session_state.agent.system_prompt + "\n" + context_str

            # Show thinking indicator
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("""
            <div class="assistant-message">
                <div class="assistant-avatar">
                    🤖
                </div>
                <div class="assistant-bubble">
                    🤔 Thinking and searching through legal documents...
                </div>
            </div>
            """, unsafe_allow_html=True)

            try:
                async def ask_agent(agent, prompt_for_agent):
                    return await agent.run(prompt_for_agent, max_iterations=20)

                response = asyncio.run(ask_agent(st.session_state.agent, prompt_for_agent))
                print(response)

                # Clear thinking indicator
                thinking_placeholder.empty()

                # Main answer text
                text_output = str(response.response) if hasattr(response, "response") else str(response)
                
                # Display assistant response
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="assistant-avatar">
                        🤖
                    </div>
                    <div class="assistant-bubble">
                        {text_output}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # === Evaluate Metrics ===
                try:
                    # Gather retrieved context chunks
                    context_chunks = []
                    if hasattr(response, "tool_calls") and response.tool_calls:
                        for tool_call in response.tool_calls:
                            raw_output = getattr(tool_call.tool_output, "raw_output", None)
                            if raw_output and hasattr(raw_output, "source_nodes"):
                                for node in raw_output.source_nodes:
                                    context_chunks.append(node.text)

                    if context_chunks:
                        # Build dataset for Ragas
                        data_samples = {
                            "user_input": [prompt],
                            "response": [text_output],
                            "retrieved_contexts": [context_chunks]
                        }
                        dataset = Dataset.from_dict(data_samples)

                        context_precision_metric = LLMContextPrecisionWithoutReference(llm=llm_wrapper)

                        # Run evaluation
                        results = evaluate(
                            dataset,
                            metrics=[faithfulness, context_precision_metric],
                            llm = llm_wrapper,
                            embeddings = embeddings_model
                        )

                        print(results)

                        faithfulness_score = results["faithfulness"][0]
                        context_precision_score = results['llm_context_precision_without_reference'][0]

                        # Display the score nicely
                        st.markdown(f"""
                        <div style="
                            margin-top:1rem;
                            padding:0.8rem 1rem;
                            background: linear-gradient(135deg, #06beb6 0%, #48b1bf 100%);
                            color: white;
                            border-radius:8px;
                            display:inline-block;
                            box-shadow: 0 3px 15px rgba(72, 177, 191, 0.3);
                            font-size: 1rem;
                        ">
                            <strong>Faithfulness Score:</strong> {faithfulness_score:.3f}
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            <strong>Context Precision (LLM, no reference):</strong> {context_precision_score:.3f}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("No relevant context retrieved. Metrics not computed.")

                except Exception as e:
                    st.warning(f"Metrics evaluation failed: {e}")


                # Display source metadata
                if hasattr(response, "tool_calls") and response.tool_calls:
                    st.markdown("### Sources & References")
                    seen = set()
                    for tool_call in response.tool_calls:
                        raw_output = getattr(tool_call.tool_output, "raw_output", None)
                        if raw_output and hasattr(raw_output, "source_nodes"):
                            for node in raw_output.source_nodes:
                                meta = getattr(node, "metadata", {})
                                if meta:
                                    meta_key = tuple(sorted(meta.items()))
                                    if meta_key in seen:
                                        continue
                                    seen.add(meta_key)
                                    with st.expander(f"{meta.get('case_name', 'Source Metadata')}"):
                                        st.markdown('<div class="source-metadata">', unsafe_allow_html=True)
                                        for k, v in meta.items():
                                            st.markdown(f"**{k.replace('_', ' ').title()}:** {v}")
                                        st.markdown('</div>', unsafe_allow_html=True)

                # Save messages
                st.session_state.messages.append({"role": "assistant", "content": text_output})
                st.session_state.chat_memory.append({"role": "assistant", "content": text_output})
                save_message(str(st.session_state.current_chat_session['_id']), "user", prompt, messages_collection)
                save_message(str(st.session_state.current_chat_session['_id']), "assistant", text_output, messages_collection)

            except Exception as e:
                thinking_placeholder.empty()
                error_msg = f"Error: {str(e)}"
                st.markdown(f"""
                <div class="assistant-message">
                    <div class="assistant-avatar">
                        🤖
                    </div>
                    <div class="assistant-bubble">
                        {error_msg}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.session_state.chat_memory.append({"role": "assistant", "content": error_msg})

    else:
        # Welcome header (minimalist)
        st.markdown(f"""
        <div style="padding: 1rem; border: 0px solid #ddd; border-radius: 8px; text-align: center; margin-bottom: 1rem;">
            <h2 style="margin-bottom: 0.5rem;">⚖️ Legal Case RAG Chatbot</h2>
            <p style="margin: 0;">Welcome, <strong>{st.session_state.username}</strong>! Create a new chat session to get started.</p>
        </div>
        """, unsafe_allow_html=True)

        
        st.info("Use the sidebar to create a new chat session or select from your previous conversations.")