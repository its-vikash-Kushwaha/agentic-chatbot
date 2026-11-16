from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import math
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os 
from typing import Any
from langgraph.types import interrupt, Command


load_dotenv()


# LLM 
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3
)


# Embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def get_db_path(thread_id: str | None = None) -> str:
    """Return isolated vector store path for a specific thread."""
    if thread_id:
        return os.path.join("faiss_db", str(thread_id))
    return "faiss_db"


def ingest_rag_document(file_path: str, thread_id: str | None = None):
    """Index PDF document into an isolated vector store for this specific thread."""
    db_path = get_db_path(thread_id)
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(db_path)


def get_retriever(thread_id: str | None = None):
    """Retrieve the vector index for a thread, preventing cross-user data leakage."""
    db_path = get_db_path(thread_id)
    if not os.path.exists(db_path):
        # Fallback to shared faiss_db only if it exists and has an index
        if not (os.path.exists("faiss_db") and os.path.exists(os.path.join("faiss_db", "index.faiss"))):
            return None
        db_path = "faiss_db"

    try:
        vector_store = FAISS.load_local(
            folder_path=db_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        )

        return retriever
    except Exception:
        return None




# rag tool

@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Search and retrieve information from the user's uploaded PDF document, resume, or CV.

    ALWAYS use this tool whenever the user asks about their resume, CV, projects,
    work experience, skills, education, certifications, contact info, or any questions
    about the uploaded file or document contents.

    Args:
        query: The search query or question to retrieve relevant sections from the document.
    """
    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    retriever = get_retriever(thread_id)
    if retriever is None:
        return (
            "No document is currently available in this conversation. "
            "Please instruct the user to upload their PDF document using the attachment icon."
        )

    try:
        documents = retriever.invoke(query)

        if not documents:
            return "No relevant information was found in the uploaded document."

        formatted_documents = []

        for index, document in enumerate(documents, start=1):
            source = os.path.basename(document.metadata.get("source", "Uploaded Document"))
            page = document.metadata.get("page", 0)

            formatted_documents.append(
                f"[Document Section {index} - Page {page + 1}]\n"
                f"{document.page_content}"
            )

        return "\n\n".join(formatted_documents)
    except Exception as e:
        return f"Error reading document: {str(e)}"




# Tools

search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    
    r = requests.get(url)
    return r.json()



@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }




@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        # Step 1: Convert the location name into latitude and longitude
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key,
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )
        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        # Step 2: Get current weather using latitude and longitude
        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"

        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"
    


# Make tool list
tools = [search_tool,calculator, get_stock_price,get_current_weather, rag_tool, purchase_stock]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)




# State
class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]



# Nodes 1
def chat_node(state: ChatState, config: RunnableConfig):
    """LLM node that can answer directly or call an appropriate tool."""

    thread_id = config.get("configurable", {}).get("thread_id") if config else None
    db_path = get_db_path(thread_id)
    has_uploaded_doc = (
        (db_path and os.path.exists(db_path)) or
        (os.path.exists("faiss_db") and os.path.exists(os.path.join("faiss_db", "index.faiss")))
    )

    if has_uploaded_doc:
        doc_status = (
            "DOCUMENT STATUS: [ACTIVE]\n"
            "- A document/resume has been uploaded and indexed in the vector database for this conversation.\n"
            "- CRITICAL: Whenever the user asks about their resume, CV, projects, work experience, "
            "education, skills, certifications, contact details, or document contents, you MUST invoke `rag_tool` "
            "to look up the information.\n"
            "- NEVER say you don't have access to the resume or ask the user to upload it again.\n"
        )
    else:
        doc_status = (
            "DOCUMENT STATUS: [NO DOCUMENT UPLOADED YET]\n"
            "- If the user asks about an uploaded document or resume, politely ask them to upload a PDF "
            "using the attachment icon in the chat input.\n"
        )

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            f"{doc_status}\n"
            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF, resume, or document.\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `purchase_stock` when the user wants to purchase a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"

            "Answer general greetings and conversational questions directly when no tool is needed. "
            "After receiving tool results, provide a clear, well-structured, and helpful answer."
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}




# Nodes 2 - tool node
tool_node = ToolNode(tools)



# Checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)



# graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

#add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpoint)



# Helper functions for Streamlit frontend with multi-user isolation
def get_user_threads(session_id: str | None = None) -> list:
    """
    Return thread IDs for a specific user session to prevent users from seeing
    each other's chat histories.
    """
    if not session_id:
        return []

    user_threads = []
    seen = set()

    try:
        for ckpt in checkpoint.list(None):
            cfg = ckpt.config.get("configurable", {})
            metadata = getattr(ckpt, "metadata", {}) or {}
            ckpt_session = cfg.get("session_id") or metadata.get("session_id")

            if ckpt_session == session_id:
                tid = cfg.get("thread_id")
                if tid and tid not in seen:
                    seen.add(tid)
                    user_threads.append(tid)
    except Exception:
        pass

    return user_threads


def get_all_threads(session_id: str | None = None):
    """Backwards-compatible wrapper that filters by user session."""
    return get_user_threads(session_id)

