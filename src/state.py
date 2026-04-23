from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    messages: List[Dict[str, str]]        # [{"role":"user/assistant","content":"..."}]
    last_user_message: str
    intent: str                           # greeting | product_inquiry | high_intent
    pending_question: str
    retrieved_context: str
    lead: Dict[str, str]                  # {"name":"","email":"","platform":""}
    missing_fields: List[str]             # ["name","email","platform"]
    lead_captured: bool