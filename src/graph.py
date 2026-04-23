from __future__ import annotations

from typing import Dict, Any, List

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

from src.state import AgentState
from src.intent import classify_intent
from src.rag import build_retriever
from src.tools import mock_lead_capture, is_valid_email
from dotenv import load_dotenv
load_dotenv()


KNOWN_PLATFORMS = {
    "youtube": "YouTube",
    "instagram": "Instagram",
    "tiktok": "TikTok",
    "facebook": "Facebook",
    "x": "X",
    "twitter": "Twitter",
    "linkedin": "LinkedIn",
}


def _looks_like_product_question(text: str) -> bool:
    lowered = text.lower()
    question_hints = ("?", "price", "pricing", "plan", "plans", "feature", "refund", "support")
    return any(hint in lowered for hint in question_hints)


def _extract_platform_value(text: str) -> str:
    lowered = text.lower()
    for platform, canonical in KNOWN_PLATFORMS.items():
        if platform in lowered:
            return canonical
    return text.strip()


def _extract_lead_fields(text: str, lead: Dict[str, str]) -> Dict[str, str]:
    updated = dict(lead)
    value = text.strip()
    if not updated.get("name"):
        updated["name"] = value
    elif not updated.get("email"):
        updated["email"] = value
    elif not updated.get("platform"):
        updated["platform"] = _extract_platform_value(value)
    return updated


def read_user_node(state: AgentState) -> AgentState:
    # No-op placeholder; main loop sets last_user_message/messages.
    return state


def route_after_read_user(state: AgentState) -> str:
    # Keep collecting lead fields once high-intent qualification has started.
    collecting_lead = (
        state.get("intent") == "high_intent"
        and bool(state.get("missing_fields"))
        and not state.get("lead_captured", False)
    )
    if collecting_lead:
        return "lead_progress"
    return "classify"


def classify_node_factory(llm: ChatGoogleGenerativeAI):
    def classify_node(state: AgentState) -> AgentState:
        user_text = state["last_user_message"]
        intent = classify_intent(user_text, llm)
        pending_question = state.get("pending_question", "")
        if intent == "high_intent" and _looks_like_product_question(user_text):
            pending_question = user_text
        elif intent == "product_inquiry":
            pending_question = ""
        return {**state, "intent": intent, "pending_question": pending_question}
    return classify_node


def retrieve_node_factory(retriever):
    def retrieve_node(state: AgentState) -> AgentState:
        user_text = state["last_user_message"]
        docs = retriever.invoke(user_text)
        context = "\n\n".join(d.page_content for d in docs) if docs else ""
        return {**state, "retrieved_context": context}
    return retrieve_node


def lead_progress_node(state: AgentState) -> AgentState:
    lead = _extract_lead_fields(state["last_user_message"], state["lead"])
    missing = [f for f in ["name", "email", "platform"] if not lead.get(f)]
    return {**state, "lead": lead, "missing_fields": missing}


def lead_capture_node_factory(llm: ChatGoogleGenerativeAI, retriever):
    def lead_capture_node(state: AgentState) -> AgentState:
        lead = state["lead"]
        missing = [f for f in ["name", "email", "platform"] if not lead.get(f)]

        # Strict gate: do not call tool with missing details or if already captured.
        if state.get("lead_captured", False) or missing:
            return {**state, "missing_fields": missing}

        email = lead.get("email", "")
        if not is_valid_email(email):
            messages = list(state["messages"])
            messages.append(
                {
                    "role": "assistant",
                    "content": "Please provide a valid email address to continue.",
                }
            )
            return {
                **state,
                "messages": messages,
                "missing_fields": ["email"],
            }

        tool_message = mock_lead_capture(
            name=lead["name"],
            email=lead["email"],
            platform=lead["platform"],
        )
        messages = list(state["messages"])
        messages.append({"role": "assistant", "content": tool_message})

        pending_question = state.get("pending_question", "").strip()
        if pending_question:
            docs = retriever.invoke(pending_question)
            context = "\n\n".join(d.page_content for d in docs) if docs else ""
            prompt = (
                "You are AutoStream assistant. Answer ONLY from provided context.\n"
                "If context is insufficient, say you don't have that detail yet.\n\n"
                f"Context:\n{context}\n\n"
                f"User: {pending_question}"
            )
            answer = llm.invoke(prompt).content.strip()
            messages.append({"role": "assistant", "content": answer})

        return {
            **state,
            "messages": messages,
            "lead_captured": True,
            "missing_fields": [],
            "pending_question": "",
        }

    return lead_capture_node


def respond_node_factory(llm: ChatGoogleGenerativeAI):
    def respond_node(state: AgentState) -> AgentState:
        intent = state["intent"]

        if intent == "greeting":
            answer = "Hey! I can help with AutoStream pricing, features, and plans."
        elif intent == "product_inquiry":
            ctx = state.get("retrieved_context", "")
            prompt = (
                "You are AutoStream assistant. Answer ONLY from provided context.\n"
                "If context is insufficient, say you don't have that detail yet.\n\n"
                f"Context:\n{ctx}\n\n"
                f"User: {state['last_user_message']}"
            )
            answer = llm.invoke(prompt).content.strip()
        else:  # high_intent
            missing = state.get("missing_fields", [])
            if not missing:
                answer = "Thanks, processing your lead details now."
            else:
                next_field = missing[0]
                if next_field == "email":
                    answer = "Awesome - please share your email address."
                elif next_field == "platform":
                    answer = "Great - which creator platform do you use (YouTube, Instagram, etc.)?"
                else:
                    answer = "Awesome - to continue, please share your name."

        messages = list(state["messages"])
        messages.append({"role": "assistant", "content": answer})
        return {**state, "messages": messages}
    return respond_node


def route_after_classify(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "product_inquiry":
        return "retrieve"
    if intent == "high_intent":
        return "lead_progress"
    return "respond"  # greeting


def route_after_lead_progress(state: AgentState) -> str:
    if not state.get("missing_fields", []):
        return "capture_lead"
    return "respond"


def build_graph() -> Any:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    retriever = build_retriever()

    graph = StateGraph(AgentState)

    graph.add_node("read_user", read_user_node)
    graph.add_node("classify", classify_node_factory(llm))
    graph.add_node("retrieve", retrieve_node_factory(retriever))
    graph.add_node("lead_progress", lead_progress_node)
    graph.add_node("capture_lead", lead_capture_node_factory(llm, retriever))
    graph.add_node("respond", respond_node_factory(llm))

    graph.set_entry_point("read_user")
    graph.add_conditional_edges(
        "read_user",
        route_after_read_user,
        {
            "lead_progress": "lead_progress",
            "classify": "classify",
        },
    )

    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "retrieve": "retrieve",
            "lead_progress": "lead_progress",
            "respond": "respond",
        },
    )

    graph.add_edge("retrieve", "respond")
    graph.add_conditional_edges(
        "lead_progress",
        route_after_lead_progress,
        {
            "capture_lead": "capture_lead",
            "respond": "respond",
        },
    )
    graph.add_edge("capture_lead", END)
    graph.add_edge("respond", END)

    return graph.compile()