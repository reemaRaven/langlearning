"""Streamlit UI over the LangGraph agent from src/graph.py.

Run:
    streamlit run app.py

Needs ANTHROPIC_API_KEY, either in a local .env (see .env.example) or
pasted into the sidebar for this session only.
"""

import os

import streamlit as st

from src.graph import build_graph

st.set_page_config(page_title="langlearning assistant", page_icon="🦜")
st.title("🦜 langlearning assistant")
st.caption(
    "Ask about LangChain / RAG / tool calling / LangGraph concepts, or ask "
    "for a calculation, the date, or a web search. A LangGraph agent "
    "routes each question to the knowledge base or a tool automatically."
)

with st.sidebar:
    st.subheader("Anthropic API key")
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        st.success("Loaded from .env")
    else:
        entered_key = st.text_input(
            "ANTHROPIC_API_KEY",
            type="password",
            help="Used only for this session, never written to disk.",
        )
        if entered_key:
            os.environ["ANTHROPIC_API_KEY"] = entered_key
            st.success("Key set for this session.")

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.warning(
        "No ANTHROPIC_API_KEY found. Paste one in the sidebar, or copy "
        ".env.example to .env and add it there, to start chatting."
    )
    st.stop()


@st.cache_resource
def get_app():
    return build_graph()


agent = get_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("route"):
            st.caption(f"routed to: {message['route']}")

question = st.chat_input("Ask a question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"question": question})
        st.markdown(result["answer"])
        st.caption(f"routed to: {result['route']}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "route": result["route"],
        }
    )
