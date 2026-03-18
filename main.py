"""
Streamlit chat UI for the Country Information AI Agent.
"""
from __future__ import annotations

import asyncio
import logging
import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.agent.graph import agent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Streamlit page config
st.set_page_config(
    page_title="Country Info AI",
    page_icon=" ",
    layout="centered",
    initial_sidebar_state="expanded",
)


# Custom CSS
st.markdown("""
<style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 2rem; }
    .main-header p  { color: #a8b2d8; margin: 0.5rem 0 0 0; font-size: 0.95rem; }

    /* Chat messages */
    [data-testid="stChatMessage"] { border-radius: 8px; margin-bottom: 0.5rem; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #0d1117; }
    [data-testid="stSidebar"] .stMarkdown { color: #8b949e; }

    /* Input */
    [data-testid="stChatInput"] textarea {
        background: #161b22;
        border: 1px solid #30363d;
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": str, "content": str}


# Sidebar definition
with st.sidebar:
    st.markdown("## Country Info AI")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("""
- Type in any country
- Ask for specific info (capital, currency, etc.)
- You can ask follow-ups!
    """)

    st.markdown("---")
    st.markdown("**Example questions:**")
    examples = [
        "What's the capital of Japan?",
        "What currency do they use in Brazil?",
        "How many people live in Germany?",
        "What do they speak in Switzerland?",
        "Tell me about France capital, pop, and currency",
    ]
    for ex in examples:
        st.markdown(f"• *{ex}*")

    st.markdown("---")
    st.markdown(f"**Session ID:** `{st.session_state.thread_id[:8]}...`")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.markdown(
        "Built with [Streamlit](https://streamlit.io)",
        unsafe_allow_html=False,
    )


# Main header
st.markdown("""
<div class="main-header">
    <h1> Country Information AI</h1>
    <p>Ask me about countries capitals, population, currency, languages & more</p>
</div>
""", unsafe_allow_html=True)


# Render existing chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat input & agent trigger
if prompt := st.chat_input("Ask about any country..."):
    # Show user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # run the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                lc_messages = []
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        lc_messages.append(AIMessage(content=msg["content"]))

                state_input = {
                    "messages": lc_messages,
                }
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                result = asyncio.run(agent.ainvoke(state_input, config=config))
                answer = result.get("answer") or "Couldn't generate an answer. Try again."

            except Exception as exc:
                logger.exception("Agent invocation failed: %s", exc)
                answer = (
                    "Uh oh, something went wrong processing that request. "
                    "Make sure your API key is ok and try again."
                )

        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
