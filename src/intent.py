# src/intent.py
from langchain_google_genai import ChatGoogleGenerativeAI

HIGH_INTENT_HINTS = {
    "sign up", "signup", "get started", "start", "buy", "purchase",
    "try pro", "subscribe", "trial", "book demo"
}

def rule_high_intent(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in HIGH_INTENT_HINTS)

def classify_intent(text: str, llm: ChatGoogleGenerativeAI) -> str:
    if rule_high_intent(text):
        return "high_intent"

    prompt = (
        "Classify user intent into exactly one label:\n"
        "greeting | product_inquiry | high_intent\n"
        f"User: {text}\n"
        "Return only the label."
    )
    try:
        label = llm.invoke(prompt).content.strip().lower()
    except Exception:
        # Fallback keeps the conversation responsive if model call fails.
        return "product_inquiry"
    if label not in {"greeting", "product_inquiry", "high_intent"}:
        return "product_inquiry"
    return label