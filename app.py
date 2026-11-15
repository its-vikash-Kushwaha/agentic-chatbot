from backend import (
    chatbot,
    get_all_threads,
    ingest_rag_document
)

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from langgraph.types import Command

import streamlit as st
import uuid
import tempfile
import os


def extract_text(content) -> str:
    """
    Extract plain text from an AIMessage content value.
    Handles both Gemini 3.x list-of-blocks format and plain string format.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


def get_thread_title(thread_id: str) -> str:
    """
    Return a human-readable title for a thread.
    Uses the first HumanMessage as the title (truncated to 35 chars),
    just like ChatGPT does. Falls back to a short thread ID slice.
    """
    # Check in-memory cache first
    cached = st.session_state.get("thread_titles", {})
    if thread_id in cached:
        return cached[thread_id]

    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        messages = state.values.get("messages", [])
        for msg in messages:
            if isinstance(msg, HumanMessage):
                text = extract_text(msg.content).strip()
                if text:
                    title = text[:35] + ("…" if len(text) > 35 else "")
                    # Cache it
                    if "thread_titles" not in st.session_state:
                        st.session_state["thread_titles"] = {}
                    st.session_state["thread_titles"][thread_id] = title
                    return title
    except Exception:
        pass

    # Fallback: short ID
    return f"Chat {thread_id[:8]}…"


# Add a new thread ID to the conversation list
def add_thread(thread_id):

    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


# Create a completely new chat conversation
def reset_chat():

    # Generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # Clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # ========================= HITL ADDED =========================
    # Clear any pending human approval request
    st.session_state["pending_hitl"] = None
    # =============================================================

    # Add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])


# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


# ========================= HITL helper functions =========================

def get_pending_interrupt(thread_id):
    """
    Return the first unresolved LangGraph interrupt for a thread.

    Returns:
        The pending Interrupt object, or None.
    """

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:

        # Read the current checkpoint state
        state_snapshot = chatbot.get_state(config)

        # Some LangGraph versions expose interrupts directly
        direct_interrupts = getattr(
            state_snapshot,
            "interrupts",
            ()
        ) or ()

        if direct_interrupts:
            return direct_interrupts[0]

        # Other LangGraph versions store interrupts inside tasks
        tasks = getattr(
            state_snapshot,
            "tasks",
            ()
        ) or ()

        for task in tasks:

            task_interrupts = getattr(
                task,
                "interrupts",
                ()
            ) or ()

            if task_interrupts:
                return task_interrupts[0]

    except Exception:

        # A newly created thread may not have a checkpoint yet
        return None

    return None


def save_pending_interrupt(thread_id, interrupt_object):
    """
    Save the pending interrupt information inside Streamlit state.
    """

    st.session_state["pending_hitl"] = {
        "thread_id": thread_id,
        "prompt": str(interrupt_object.value)
    }


def sync_pending_interrupt(thread_id):
    """
    Synchronize Streamlit HITL state with the LangGraph checkpoint.

    This allows a pending approval request to reappear after:
    - a Streamlit rerun
    - a browser refresh
    - switching between conversations
    """

    pending_interrupt = get_pending_interrupt(thread_id)

    if pending_interrupt is not None:

        save_pending_interrupt(
            thread_id,
            pending_interrupt
        )

    else:

        current_pending = st.session_state.get(
            "pending_hitl"
        )

        if (
            current_pending is not None
            and current_pending.get("thread_id") == thread_id
        ):
            st.session_state["pending_hitl"] = None


def resume_hitl_execution(decision):
    """
    Resume an interrupted LangGraph execution.

    Args:
        decision:
            "yes" approves the stock purchase.
            "no" rejects the stock purchase.
    """

    pending_hitl = st.session_state.get(
        "pending_hitl"
    )

    if not pending_hitl:

        st.warning(
            "There is no pending action to approve or reject."
        )

        return

    # Get the thread that originally triggered the interrupt
    interrupted_thread_id = pending_hitl["thread_id"]

    # The same thread ID must be used when resuming
    resume_config = {
        "configurable": {
            "thread_id": interrupted_thread_id
        },
        "metadata": {
            "thread_id": interrupted_thread_id
        },
        "run_name": "hitl_resume_trace",
    }

    try:

        # Display the resumed response
        with st.chat_message("assistant", avatar="⚡"):

            status_holder = {
                "box": st.status(
                    "🔄 Resuming the requested action...", expanded=True
                )
            }

            def resumed_ai_only_stream():

                for message_chunk, metadata in chatbot.stream(
                    Command(resume=decision),
                    config=resume_config,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running", expanded=True,
                        )

                    if isinstance(message_chunk, AIMessage):
                        content = message_chunk.content
                        if isinstance(content, str) and content:
                            yield content
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text = block.get("text", "")
                                    if text:
                                        yield text
                                elif isinstance(block, str) and block:
                                    yield block

            resumed_ai_message = st.write_stream(resumed_ai_only_stream())

            # Check whether another interrupt occurred
            next_interrupt = get_pending_interrupt(interrupted_thread_id)

            if next_interrupt is not None:
                save_pending_interrupt(interrupted_thread_id, next_interrupt)
                status_holder["box"].update(
                    label="⚠️ Another approval is required",
                    state="complete", expanded=False
                )
            else:
                st.session_state["pending_hitl"] = None
                status_holder["box"].update(
                    label="✅ Action completed",
                    state="complete", expanded=False
                )

        # Save the assistant response in Streamlit UI history
        if resumed_ai_message:

            st.session_state["message_history"].append({
                "role": "assistant",
                "content": resumed_ai_message
            })

        # Rerun so the response appears in normal chat order
        st.rerun()

    except Exception as error:

        st.error(
            f"Could not resume the requested action: {error}"
        )


# ========================= Page configuration =========================

st.set_page_config(
    page_title="Agentic AI • Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Advanced UI Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Ensure Streamlit top header doesn't overlap or clip our content */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 1.5rem !important;
    z-index: 10;
}

/* App container - comfortable breathing room and spacing */
.block-container {
    max-width: 880px !important;
    padding-top: 3rem !important;
    padding-bottom: 9rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    margin: 0 auto !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f17 0%, #111827 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    min-width: 270px;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Sidebar section header */
.sidebar-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* 1. New Chat button (FIRST button in sidebar) */
[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1rem !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.2s ease !important;
    margin-bottom: 0.6rem !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"]:first-of-type button:hover {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
    transform: translateY(-1px);
}

/* 2. Conversation History buttons (ALL OTHER buttons in sidebar) */
[data-testid="stSidebar"] div[data-testid="stButton"]:not(:first-of-type) button {
    background: rgba(255, 255, 255, 0.02) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 8px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.5rem 0.75rem !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
    margin-bottom: 0.3rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

[data-testid="stSidebar"] div[data-testid="stButton"]:not(:first-of-type) button:hover {
    background: rgba(255, 255, 255, 0.07) !important;
    color: #f1f5f9 !important;
    border-color: rgba(255, 255, 255, 0.14) !important;
    transform: none !important;
}

/* 3. ACTIVE conversation item (primary button among history) */
[data-testid="stSidebar"] div[data-testid="stButton"]:not(:first-of-type) button[kind="primary"] {
    background: rgba(37, 99, 235, 0.18) !important;
    color: #60a5fa !important;
    border: 1px solid rgba(59, 130, 246, 0.45) !important;
    font-weight: 600 !important;
}

/* Top Chat Header Card */
.chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0 1.25rem 0;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.chat-header-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: -0.3px;
}

.chat-header-subtitle {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.15rem;
}

.chat-header-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(34, 197, 94, 0.1);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.25);
    padding: 0.3rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.76rem;
    font-weight: 600;
}

.status-dot {
    width: 7px;
    height: 7px;
    background-color: #4ade80;
    border-radius: 50%;
    box-shadow: 0 0 8px #4ade80;
    display: inline-block;
}

/* Tool Badges Row */
.tool-chips-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.65rem;
}

.tool-chip {
    font-size: 0.72rem;
    padding: 0.22rem 0.6rem;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    color: #94a3b8;
    font-weight: 500;
    transition: all 0.15s ease;
}

.tool-chip:hover {
    background: rgba(255, 255, 255, 0.08);
    color: #f1f5f9;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    padding: 0.85rem 1.15rem !important;
    margin-bottom: 0.75rem !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    background: rgba(17, 24, 39, 0.6) !important;
    backdrop-filter: blur(8px) !important;
    line-height: 1.6 !important;
    font-size: 0.94rem !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
}

/* ChatGPT-style File Attachment Card */
.chat-file-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 10px;
    padding: 0.55rem 0.85rem;
    margin-bottom: 0.55rem;
    max-width: 320px;
}

.chat-file-chip:hover {
    background: rgba(255, 255, 255, 0.08);
    border-color: rgba(59, 130, 246, 0.4);
}

.chat-file-chip-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.3) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 8px;
    font-size: 1.1rem;
}

.chat-file-chip-info {
    display: flex;
    flex-direction: column;
    min-width: 0;
}

.chat-file-chip-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #f1f5f9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 220px;
}

.chat-file-chip-type {
    font-size: 0.68rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

/* Chat Input Bar - Sleek Floating Island */
[data-testid="stChatInput"] {
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    background-color: #111827 !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45) !important;
    transition: all 0.2s ease;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.25), 0 10px 30px rgba(0, 0, 0, 0.6) !important;
}

/* Tool execution status container */
[data-testid="stStatusWidget"] {
    border-radius: 10px !important;
    border: 1px solid rgba(59, 130, 246, 0.25) !important;
    background: rgba(15, 23, 42, 0.65) !important;
    margin-bottom: 0.6rem !important;
}

/* Welcome Hero Cards Grid */
.hero-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.85rem;
    margin: 1.5rem 0 2rem 0;
}

.hero-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 12px;
    padding: 1rem;
    transition: all 0.2s ease;
}

.hero-card:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(59, 130, 246, 0.4);
    transform: translateY(-2px);
}

.hero-card-icon {
    font-size: 1.35rem;
    margin-bottom: 0.4rem;
}

.hero-card-title {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.25rem;
}

.hero-card-desc {
    font-size: 0.76rem;
    color: #64748b;
    line-height: 1.4;
}

/* System specs card in sidebar */
.system-specs-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 0.85rem;
    margin-top: 1.5rem;
}

.specs-title {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.74rem;
    font-weight: 700;
    color: #64748b;
    letter-spacing: 0.6px;
    margin-bottom: 0.55rem;
}

.spec-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.76rem;
    margin-bottom: 0.35rem;
}

.spec-row:last-child {
    margin-bottom: 0;
}

.spec-label {
    color: #64748b;
}

.spec-val {
    color: #cbd5e1;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Top Application Header
st.markdown("""
<div class="chat-header">
    <div>
        <div class="chat-header-title">
            <span>⚡</span> Agentic AI Assistant
        </div>
        <div class="chat-header-subtitle">
            Autonomous multi-tool agent powered by LangGraph & Groq
        </div>
        <div class="tool-chips-container">
            <span class="tool-chip">🔍 Web Search</span>
            <span class="tool-chip">📈 Live Stocks</span>
            <span class="tool-chip">🌤️ Weather</span>
            <span class="tool-chip">📄 PDF RAG</span>
            <span class="tool-chip">🧮 Calculator</span>
            <span class="tool-chip">🛡️ HITL Approval</span>
        </div>
    </div>
    <div class="chat-header-badge">
        <span class="status-dot"></span> Groq • GPT-OSS 120B
    </div>
</div>
""", unsafe_allow_html=True)


# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()


# ========================= HITL ADDED =========================

# Store the currently pending human approval request
if "pending_hitl" not in st.session_state:
    st.session_state["pending_hitl"] = None

# Cache for human-readable thread titles
if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

# =============================================================


# Add the current thread to the conversation list
add_thread(st.session_state["thread_id"])


# ========================= HITL ADDED =========================

# Recover pending approval after page refresh or rerun
sync_pending_interrupt(
    st.session_state["thread_id"]
)

# =============================================================


# ========================= Sidebar threading feature =========================

# Display the sidebar title
st.sidebar.markdown('<div class="sidebar-title">💬 CONVERSATIONS</div>', unsafe_allow_html=True)


# Create a button for starting a new conversation
if st.sidebar.button("➕  New Chat", use_container_width=True):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()


st.sidebar.divider()


# Display all conversation threads in reverse order
# This shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:

    # Build a human-readable title from the first message
    title = get_thread_title(thread_id)

    # Highlight the currently active thread
    is_active = thread_id == st.session_state["thread_id"]
    button_label = f"{'▶ ' if is_active else ''}{title}"

    # Create one sidebar button for every conversation
    if st.sidebar.button(
        button_label,
        key=thread_id,
        use_container_width=True,
        type="primary" if is_active else "secondary",
    ):

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # Loop through all saved messages
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such as ToolMessage
            else:
                continue

            # Convert the LangChain message into a dictionary
            raw_content = message.content
            # Gemini 3+ stores content as a list of blocks
            if isinstance(raw_content, list):
                text_parts = [
                    block.get("text", "") if isinstance(block, dict) else block
                    for block in raw_content
                    if (isinstance(block, dict) and block.get("type") == "text") or isinstance(block, str)
                ]
                raw_content = "".join(text_parts)

            # Check if this message had an attached document tag
            file_name = None
            if isinstance(raw_content, str) and raw_content.startswith("[Attached Document: "):
                closing_idx = raw_content.find("]")
                if closing_idx != -1:
                    file_name = raw_content[19:closing_idx].strip()
                    raw_content = raw_content[closing_idx + 1:].strip()

            temp_messages.append({
                "role": role,
                "content": raw_content,
                "file_name": file_name
            })

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # ========================= HITL ADDED =========================

        # Restore any pending approval for this conversation
        sync_pending_interrupt(thread_id)

        # =============================================================

        # Rerun the application to display the loaded messages
        st.rerun()


# Display sidebar footer
st.sidebar.markdown("""
<div class="system-specs-card">
    <div class="specs-title">⚙️ SYSTEM SPECS</div>
    <div class="spec-row"><span class="spec-label">Engine</span><span class="spec-val">LangGraph</span></div>
    <div class="spec-row"><span class="spec-label">LLM</span><span class="spec-val">Groq • 120B</span></div>
    <div class="spec-row"><span class="spec-label">Memory</span><span class="spec-val">SQLite Saver</span></div>
    <div class="spec-row"><span class="spec-label">Guardrail</span><span class="spec-val">HITL Safe</span></div>
</div>
""", unsafe_allow_html=True)


# ========================= Main chat interface =========================

# If conversation is empty, display welcome hero cards
if not st.session_state["message_history"]:
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0 1.5rem 0;">
        <h2 style="font-size: 1.85rem; font-weight: 700; margin-bottom: 0.5rem; color: #f0f6fc;">
            How can I help you today?
        </h2>
        <p style="color: #8b949e; font-size: 0.95rem; max-width: 550px; margin: 0 auto;">
            I can search the web, track stock prices, execute simulated trades with human approval, check live weather, and analyze your PDFs.
        </p>
    </div>

    <div class="hero-card-grid">
        <div class="hero-card">
            <div class="hero-card-icon">📈</div>
            <div class="hero-card-title">Live Market Data</div>
            <div class="hero-card-desc">"What is the stock price of NVDA and TSLA?"</div>
        </div>
        <div class="hero-card">
            <div class="hero-card-icon">🌤️</div>
            <div class="hero-card-title">Real-Time Weather</div>
            <div class="hero-card-desc">"How's the weather in Tokyo and Paris right now?"</div>
        </div>
        <div class="hero-card">
            <div class="hero-card-icon">🔍</div>
            <div class="hero-card-title">Web Intelligence</div>
            <div class="hero-card-desc">"Search for latest breakthroughs in AI agents."</div>
        </div>
        <div class="hero-card">
            <div class="hero-card-icon">📄</div>
            <div class="hero-card-title">PDF Document RAG</div>
            <div class="hero-card-desc">"Upload a PDF in the chat to extract insights."</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Set avatar based on role
    avatar = "👤" if message["role"] == "user" else "⚡"

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"], avatar=avatar):

        # Display ChatGPT-style file chip if a file was attached to this message
        if message.get("file_name"):
            st.markdown(f"""
            <div class="chat-file-chip">
                <div class="chat-file-chip-icon">📄</div>
                <div class="chat-file-chip-info">
                    <div class="chat-file-chip-name">{message["file_name"]}</div>
                    <div class="chat-file-chip-type">PDF Document</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Display the message content
        if message.get("content"):
            st.markdown(extract_text(message["content"]))


# ========================= HITL approval interface =========================

# Get the currently pending approval request
pending_hitl = st.session_state.get(
    "pending_hitl"
)

# Check whether the pending approval belongs to
# the currently selected conversation
current_thread_has_pending_hitl = (
    pending_hitl is not None
    and pending_hitl.get("thread_id")
    == st.session_state["thread_id"]
)


# Display approval controls
if current_thread_has_pending_hitl:

    st.warning(
        f"🛡️ **Security Guardrail • Human Approval Required**\n\n"
        f"{pending_hitl['prompt']}"
    )

    approve_column, reject_column = st.columns(2)

    # Approve button
    with approve_column:

        if st.button(
            "✅ Approve Purchase",
            key=f"approve_{st.session_state['thread_id']}",
            type="primary",
            use_container_width=True
        ):

            # Send "yes" back to interrupt()
            resume_hitl_execution("yes")

    # Reject button
    with reject_column:

        if st.button(
            "❌ Reject Purchase",
            key=f"reject_{st.session_state['thread_id']}",
            use_container_width=True
        ):

            # Send "no" back to interrupt()
            resume_hitl_execution("no")


# ========================= Fixed chat input with PDF upload =========================

# Keep st.chat_input directly in the main body.
# This keeps it fixed at the bottom of the screen.
#
# accept_file=True adds the attachment button inside the chat input.
# file_type=["pdf"] allows PDF files only.
submission = st.chat_input(
    "Type here",
    accept_file=True,
    file_type=["pdf"],

    # Disable input while waiting for human approval
    disabled=current_thread_has_pending_hitl
)


# Default values
user_input = None
attached_file_name = None


# Process the submitted text and PDF
if submission:

    # Get the text entered by the user
    user_input = submission.text

    # Get the uploaded files
    uploaded_files = submission.files

    # Process the uploaded PDF if one was attached
    if uploaded_files:

        uploaded_pdf = uploaded_files[0]
        attached_file_name = uploaded_pdf.name
        temporary_file_path = None

        try:

            # Save the uploaded PDF as a temporary local file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temporary_file:

                temporary_file.write(
                    uploaded_pdf.getvalue()
                )

                temporary_file_path = temporary_file.name

            # Call the existing backend RAG ingestion function
            with st.spinner(
                f"Indexing {uploaded_pdf.name}..."
            ):

                ingest_rag_document(
                    temporary_file_path
                )

            # Display PDF processing confirmation toast
            st.toast(
                f"{uploaded_pdf.name} indexed successfully.",
                icon="✅"
            )

            # If user uploaded a PDF without typing a question, show it in chat like ChatGPT
            if not user_input:
                # Add user message displaying the uploaded file card
                st.session_state["message_history"].append({
                    "role": "user",
                    "content": "",
                    "file_name": attached_file_name
                })
                # Add assistant message ready to help
                st.session_state["message_history"].append({
                    "role": "assistant",
                    "content": f"I have read and indexed **`{attached_file_name}`**! Feel free to ask me to summarize it, extract projects, skills, or answer any questions about it."
                })
                st.rerun()

        except Exception as error:

            # Display PDF processing error
            st.error(
                f"PDF processing failed: {error}"
            )

        finally:

            # Delete the temporary PDF after indexing
            if (
                temporary_file_path
                and os.path.exists(temporary_file_path)
            ):
                os.remove(temporary_file_path)


# Run this block after the user submits a text message
if user_input:

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
        "file_name": attached_file_name
    })

    # Set chat title from the FIRST user message (like ChatGPT)
    thread_id_now = st.session_state["thread_id"]
    if thread_id_now not in st.session_state["thread_titles"]:
        title = user_input.strip()[:35] + ("…" if len(user_input.strip()) > 35 else "")
        st.session_state["thread_titles"][thread_id_now] = title

    # Display the user's message in the chat interface
    with st.chat_message("user", avatar="👤"):
        if attached_file_name:
            st.markdown(f"""
            <div class="chat-file-chip">
                <div class="chat-file-chip-icon">📄</div>
                <div class="chat-file-chip-info">
                    <div class="chat-file-chip-name">{attached_file_name}</div>
                    <div class="chat-file-chip-type">PDF Document</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(user_input)

    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # Format human message content to include file tag if uploaded together
    prompt_text = f"[Attached Document: {attached_file_name}]\n{user_input}" if attached_file_name else user_input

    # Assistant streaming block
    with st.chat_message("assistant", avatar="⚡"):

        status_holder = {"box": None}

        def ai_only_stream():

            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=prompt_text)]},
                config=CONFIG,
                stream_mode="messages",
            ):

                # Show tool status
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream assistant tokens (Groq returns plain strings)
                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content
                    if isinstance(content, str) and content:
                        yield content
                    elif isinstance(content, list):
                        # Fallback for any list-based model
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    yield text
                            elif isinstance(block, str) and block:
                                yield block

            # ========================= HITL ADDED =========================
            pending_interrupt = get_pending_interrupt(
                st.session_state["thread_id"]
            )
            if pending_interrupt is not None:
                save_pending_interrupt(
                    st.session_state["thread_id"],
                    pending_interrupt
                )
                yield (
                    "\n\n⚠️ This stock purchase requires your approval. "
                    "Use the Approve Purchase or Reject Purchase button below."
                )
            # =============================================================

        ai_message = st.write_stream(ai_only_stream())

        # Finalize tool status box
        if status_holder["box"] is not None:
            if get_pending_interrupt(st.session_state["thread_id"]) is not None:
                status_holder["box"].update(
                    label="⏸️ Waiting for human approval",
                    state="complete", expanded=False
                )
            else:
                status_holder["box"].update(
                    label="✅ Tool finished",
                    state="complete", expanded=False
                )

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })

    # ========================= HITL ADDED =========================

    # Approval controls are rendered earlier in the script.
    # Rerun so they appear immediately after interrupt().
    if (
        st.session_state.get("pending_hitl") is not None
        and st.session_state["pending_hitl"].get("thread_id")
        == st.session_state["thread_id"]
    ):
        st.rerun()

    # =============================================================