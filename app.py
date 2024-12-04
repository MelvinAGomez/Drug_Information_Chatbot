import os
import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from typing import Literal
from typing_extensions import TypedDict
from dataclasses import dataclass
from langgraph.graph import MessagesState
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv("./secrets/secrets.env")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize ChromaDB
PERSIST_DIR = "./db4"
embd = OllamaEmbeddings(model="nomic-embed-text")  # Same embedding model used during storage
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embd)

# Initialize the LLM for Supervisor and Worker Nodes
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-3b-preview")

# Define State Schema
@dataclass
class AgentState(MessagesState):
    """State for multi-agent processing."""
    next: str = None

class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""
    next: Literal["question_answering", "recommendation", "alternatives", "summarization", "FINISH"]

# Supervisor Node
def supervisor_node(state: AgentState) -> AgentState:
    """Determine the next worker node based on the query and current state."""
    system_prompt = (
        "You are a supervisor responsible for coordinating tasks among the following workers: question_answering, recommendation, alternatives, and summarization. "
        "If the query asks for factual information, select 'question_answering'. "
        "If the query requires advice, select 'recommendation'. "
        "If the query asks for substitutes, select 'alternatives'. "
        "If the query requires summarization, select 'summarization'. "
        "Respond with the name of the next worker, or 'FINISH' if no further tasks are required. "
        "Valid responses are: question_answering, recommendation, alternatives, summarization, FINISH."
    )

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    try:
        response = llm.invoke(messages)
        print("Supervisor response:", response.content)  # Debugging: Print response

        # Preprocess response
        response_text = response.content.strip().lower()
        response_text = response_text.split(":")[-1].strip()  # Handle verbose responses

        # Validate response
        if response_text in ["question_answering", "recommendation", "alternatives", "summarization", "finish"]:
            state["next"] = response_text
            print(f"Supervisor selected: {state['next']}")  # Debugging: Print selected worker
        else:
            raise ValueError(f"Unexpected response from supervisor: {response_text}")
    except Exception as e:
        print(f"Error in supervisor_node: {e}")
        raise RuntimeError(f"Supervisor failed: {e}")

    return state

# Worker Nodes
def query_with_llm(task: str, query: str, results: list) -> str:
    """Use LLM to process retrieved data and refine results."""
    prompt = (
        f"You are tasked with {task}. Here are the retrieved results:\n"
        f"{results}\n"
        f"Based on these, generate a suitable response for the user query: {query}"
    )
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content.strip()

def question_answering_node(state: AgentState) -> AgentState:
    """Handles question answering based on user query."""
    query = state["messages"][-1].content
    try:
        results = vectorstore.similarity_search(query, k=3)
        retrieved_data = "\n".join([f"Source: {res.metadata['source']}\n{res.page_content}" for res in results])
        response = query_with_llm("answering the question", query, retrieved_data)
        state["messages"].append(HumanMessage(content=response, name="question_answering"))
    except Exception as e:
        raise RuntimeError(f"Question answering failed: {e}")
    state["next"] = "FINISH"
    return state

def recommendation_node(state: AgentState) -> AgentState:
    """Provides recommendations based on user query."""
    query = state["messages"][-1].content
    try:
        results = vectorstore.similarity_search(query, k=3)
        retrieved_data = "\n".join([f"Source: {res.metadata['source']}\n{res.page_content}" for res in results])
        response = query_with_llm("providing recommendations", query, retrieved_data)
        state["messages"].append(HumanMessage(content=response, name="recommendation"))
    except Exception as e:
        raise RuntimeError(f"Recommendation generation failed: {e}")
    state["next"] = "FINISH"
    return state

def alternatives_node(state: AgentState) -> AgentState:
    """Suggests alternative medications using ChromaDB."""
    query = state["messages"][-1].content
    try:
        results = vectorstore.similarity_search(query, k=3)
        retrieved_data = "\n".join([f"Source: {res.metadata['source']}\n{res.page_content}" for res in results])
        response = query_with_llm("suggesting alternatives", query, retrieved_data)
        state["messages"].append(HumanMessage(content=response, name="alternatives"))
    except Exception as e:
        raise RuntimeError(f"Alternatives generation failed: {e}")
    state["next"] = "FINISH"
    return state

def summarization_node(state: AgentState) -> AgentState:
    """Summarizes the details provided in the user query."""
    query = state["messages"][-1].content
    try:
        results = vectorstore.similarity_search(query, k=3)
        retrieved_data = "\n".join([res.page_content for res in results])
        summary = query_with_llm("summarizing the details", query, retrieved_data)
        state["messages"].append(HumanMessage(content=f"Summary: {summary}", name="summarization"))
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {e}")
    state["next"] = "FINISH"
    return state

# Conditional Edge Function
def determine_next_worker(state: AgentState) -> str:
    """Determine the next worker to invoke based on the supervisor's decision."""
    return state.get("next", END)

# Build the State Graph
def build_graph():
    """Constructs the state graph for the workflow."""
    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("question_answering", question_answering_node)
    builder.add_node("recommendation", recommendation_node)
    builder.add_node("alternatives", alternatives_node)
    builder.add_node("summarization", summarization_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", determine_next_worker)
    return builder.compile()

# Streamlit Chatbot
def main():
    """Runs the Streamlit chatbot interface."""
    st.title("💬 Drug Information Chatbot")
    st.caption("🚀 Powered by LangGraph, ChatGroq, and ChromaDB")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi! How can I assist you today?"}]

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        if prompt.lower() in {"bye", "exit"}:
            goodbye_message = "Goodbye! Have a great day!"
            st.session_state.messages.append({"role": "assistant", "content": goodbye_message})
            st.chat_message("assistant").write(goodbye_message)
            st.stop()

        # Process the query
        try:
            state = AgentState(messages=[{"role": "user", "content": prompt}])
            graph = build_graph()

            # Process the graph
            with st.spinner("Processing your query..."):
                for result in graph.stream(state, subgraphs=True):
                    if isinstance(result, tuple) and len(result) == 2:
                        _, response_data = result
                        for node_name, node_output in response_data.items():
                            if "messages" in node_output:
                                response = node_output["messages"][-1].content
                                # Append response to session state and display it
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.chat_message("assistant").write(response)

                    # Stop processing if finished
                    if state.get("next") == "FINISH":
                        st.success("Query processing complete!")
                        break

        except Exception as e:
            error_message = f"An error occurred while processing your query: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            st.chat_message("assistant").write(error_message)



if __name__ == "__main__":
    main()