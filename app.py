# app.py
import streamlit as st
from db import init_connection
from ui_pages import login_page, create_account_page, main_page
from chat import load_user_sessions   # import it here

from dotenv import load_dotenv
load_dotenv()

def app():
    # Initialize session state
    default_keys = {
        'logged_in': False,
        'username': "",
        'show_create_account': False,
        'messages': [],
        'current_chat_session': None,
        'chat_sessions': [],
        'last_session_id': None
    }
    for key, val in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Initialize database connection (returns dict of collections)
    db_conn = init_connection()
    if db_conn is None:
        return

    # Route to appropriate page
    if st.session_state.logged_in:
        # load_user_sessions returns (sessions, current_session, messages_stub)
        sessions, current, messages = load_user_sessions(
            st.session_state.username,
            db_conn["sessions"],
            st.session_state.get("last_session_id")
        )
        st.session_state.chat_sessions = sessions
        st.session_state.current_chat_session = current
        # load chat messages if we have a current session
        if current:
            from chat import load_chat_history
            st.session_state.messages = load_chat_history(str(current["_id"]), db_conn["messages"])
        else:
            st.session_state.messages = []
        main_page()
    elif st.session_state.show_create_account:
        create_account_page()
    else:
        login_page()

if __name__ == "__main__":
    app()
