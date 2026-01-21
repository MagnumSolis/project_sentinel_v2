"""
Project Sentinel V2 - Agentic Orchestration
Uses LangGraph to coordinate the Scout, Analyst, and Critic agents.
"""

from typing import TypedDict, Annotated, List, Union, Dict, Any, Optional
import operator
import logging

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Import core systems
from app.core.cortex import SentinelCortex
from app.core.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)

# --- State Definition ---
class SentinelState(TypedDict):
    """The shared state of the Sentinel Agent team."""
    messages: Annotated[List[BaseMessage], operator.add]
    current_plan: Optional[str]
    retrieved_evidence: List[Dict[str, Any]]
    confidence_score: float
    missing_info: List[str]
    next_step: str

# --- Mock/Local LLM Setup ---
# In a real deployment, we'd use ChatOpenAI or a local Ollama instance
# For this implementation, we'll try to use OpenAI if key is present, otherwise mock
try:
    llm = ChatOpenAI(model="gpt-4-turbo-preview")
except Exception:
    logger.warning("OpenAI API Key not found/valid. Using Mock LLM behavior.")
    llm = None

# --- Agent Nodes ---

def supervisor_node(state: SentinelState):
    """
    The Supervisor decides which agent to call next based on the user's intent
    and the current state of the investigation.
    """
    last_message = state['messages'][-1]
    last_content = last_message.content if hasattr(last_message, 'content') else ""
    
    logger.info(f"Supervisor processing: {last_content[:50]}...")
    
    # Simple logic for Hackathon MVP:
    # 1. If no evidence, go to Scout.
    # 2. If evidence but no analysis, go to Analyst.
    # 3. If analysis, go to Critic.
    # 4. If Critic approves, END.
    
    if not state.get('retrieved_evidence'):
        return {"next_step": "scout"}
    
    if "analysis" not in str(state.get('messages', [])): # Simplified check
        return {"next_step": "analyst"}
        
    return {"next_step": "critic"}

def scout_agent(state: SentinelState):
    """
    The Scout Agent is responsible for searching Qdrant for evidence.
    """
    logger.info("Scout Agent activating...")
    query = state['messages'][-1].content
    
    # Mock search for now (would call Cortex.search)
    # In real integration, we'd access the Cortex instance here
    evidence = [
        {"id": 1, "text": "Satellite image shows flooded road.", "score": 0.9},
        {"id": 2, "text": "Drone audio detected survivor cry.", "score": 0.85}
    ]
    
    return {
        "retrieved_evidence": evidence,
        "messages": [AIMessage(content=f"Scout found {len(evidence)} pieces of evidence.")]
    }

def analyst_agent(state: SentinelState):
    """
    The Analyst Agent looks for patterns and correlates data.
    """
    logger.info("Analyst Agent activating...")
    evidence = state.get('retrieved_evidence', [])
    
    analysis = f"Based on {len(evidence)} items, I detect a critical situation in Sector 4."
    
    return {
        "messages": [AIMessage(content=analysis)]
    }

def critic_agent(state: SentinelState):
    """
    The Critic Agent verifies the findings and assigns a confidence score.
    """
    logger.info("Critic Agent activating...")
    
    # Mock verification
    confidence = 0.92
    
    final_response = "Confirmed: High-confidence flood detection in Sector 4. Rescue recommended."
    
    return {
        "confidence_score": confidence,
        "messages": [AIMessage(content=final_response)],
        "next_step": "FINISH"
    }

# --- Graph Construction ---

workflow = StateGraph(SentinelState)

# Add nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("scout", scout_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("critic", critic_agent)

# Add edges
# Supervisor decides where to go
workflow.add_conditional_edges(
    "supervisor",
    lambda x: x['next_step'],
    {
        "scout": "scout",
        "analyst": "analyst",
        "critic": "critic",
        "FINISH": END
    }
)

# Agents report back to Supervisor
workflow.add_edge("scout", "supervisor")
workflow.add_edge("analyst", "supervisor")
workflow.add_edge("critic", END) # Critic ends the flow in this simple version

# Entry point
workflow.set_entry_point("supervisor")

# Compile
sentinel_app = workflow.compile()
