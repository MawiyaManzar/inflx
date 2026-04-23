import src.graph as graph_mod


class DummyMsg:
    def __init__(self, content: str):
        self.content = content


class DummyLLM:
    def __init__(self, classify_label: str):
        self.classify_label = classify_label

    def invoke(self, prompt: str):
        if "Classify user intent into exactly one label" in prompt:
            return DummyMsg(self.classify_label)
        if "Answer ONLY from provided context" in prompt:
            return DummyMsg("From KB: Pro plan is $79/month.")
        return DummyMsg("ok")


class DummyRetriever:
    def invoke(self, _query: str):
        return []


def base_state():
    return {
        "messages": [],
        "last_user_message": "",
        "intent": "",
        "pending_question": "",
        "retrieved_context": "",
        "lead": {"name": "", "email": "", "platform": ""},
        "missing_fields": ["name", "email", "platform"],
        "lead_captured": False,
    }


def test_tool_not_called_when_required_fields_missing(monkeypatch):
    monkeypatch.setattr(graph_mod, "build_retriever", lambda: DummyRetriever())
    monkeypatch.setattr(graph_mod, "ChatGoogleGenerativeAI", lambda model: DummyLLM("high_intent"))

    calls = []

    def fake_capture(name, email, platform):
        calls.append((name, email, platform))
        return "captured"

    monkeypatch.setattr(graph_mod, "mock_lead_capture", fake_capture)

    app = graph_mod.build_graph()
    state = base_state()

    state["last_user_message"] = "Mawiya"
    state["messages"].append({"role": "user", "content": "Mawiya"})
    state = app.invoke(state)

    assert calls == []
    assert state["missing_fields"] == ["email", "platform"]
    assert state["lead_captured"] is False


def test_invalid_email_blocks_tool_call(monkeypatch):
    monkeypatch.setattr(graph_mod, "build_retriever", lambda: DummyRetriever())
    monkeypatch.setattr(graph_mod, "ChatGoogleGenerativeAI", lambda model: DummyLLM("high_intent"))

    calls = []

    def fake_capture(name, email, platform):
        calls.append((name, email, platform))
        return "captured"

    monkeypatch.setattr(graph_mod, "mock_lead_capture", fake_capture)

    app = graph_mod.build_graph()
    state = base_state()

    for text in ["Mawiya", "not-an-email", "YouTube"]:
        state["last_user_message"] = text
        state["messages"].append({"role": "user", "content": text})
        state = app.invoke(state)

    assert calls == []
    assert state["lead_captured"] is False
    assert state["missing_fields"] == ["email"]
    assert "valid email" in state["messages"][-1]["content"].lower()


def test_tool_called_once_when_all_fields_valid(monkeypatch):
    monkeypatch.setattr(graph_mod, "build_retriever", lambda: DummyRetriever())
    monkeypatch.setattr(graph_mod, "ChatGoogleGenerativeAI", lambda model: DummyLLM("high_intent"))

    calls = []

    def fake_capture(name, email, platform):
        calls.append((name, email, platform))
        return f"captured {name} {email} {platform}"

    monkeypatch.setattr(graph_mod, "mock_lead_capture", fake_capture)

    app = graph_mod.build_graph()
    state = base_state()

    for text in ["Mawiya", "mawiya@example.com", "YouTube"]:
        state["last_user_message"] = text
        state["messages"].append({"role": "user", "content": text})
        state = app.invoke(state)

    assert len(calls) == 1
    assert calls[0] == ("Mawiya", "mawiya@example.com", "YouTube")
    assert state["lead_captured"] is True
    assert state["missing_fields"] == []

    # Extra user message should not trigger capture again.
    state["last_user_message"] = "Thanks"
    state["messages"].append({"role": "user", "content": "Thanks"})
    state = app.invoke(state)
    assert len(calls) == 1


def test_lead_collection_skips_reclassification(monkeypatch):
    monkeypatch.setattr(graph_mod, "build_retriever", lambda: DummyRetriever())
    monkeypatch.setattr(graph_mod, "ChatGoogleGenerativeAI", lambda model: DummyLLM("product_inquiry"))

    def fail_classify(_text, _llm):
        raise AssertionError("classify_intent should not be called during lead collection")

    monkeypatch.setattr(graph_mod, "classify_intent", fail_classify)

    app = graph_mod.build_graph()
    state = base_state()
    # Simulate active lead collection after first high-intent turn.
    state["lead"]["name"] = "Mawiya"
    state["missing_fields"] = ["email", "platform"]
    state["intent"] = "high_intent"
    state["last_user_message"] = "mawiya@example.com"
    state["messages"].append({"role": "user", "content": "mawiya@example.com"})

    out = app.invoke(state)
    assert out["lead"]["email"] == "mawiya@example.com"
    assert out["missing_fields"] == ["platform"]


def test_classification_failure_falls_back_to_inquiry(monkeypatch):
    monkeypatch.setattr(graph_mod, "build_retriever", lambda: DummyRetriever())

    class ClassificationFailingLLM:
        def invoke(self, prompt: str):
            if "Classify user intent into exactly one label" in prompt:
                raise RuntimeError("simulated llm outage")
            return DummyMsg("Fallback response path still works")

    monkeypatch.setattr(graph_mod, "ChatGoogleGenerativeAI", lambda model: ClassificationFailingLLM())

    app = graph_mod.build_graph()
    state = base_state()
    state["last_user_message"] = "Tell me about your plans"
    state["messages"].append({"role": "user", "content": state["last_user_message"]})
    out = app.invoke(state)

    assert out["intent"] == "product_inquiry"


def test_mixed_intent_question_is_answered_after_capture(monkeypatch):
    class RetrieverWithPricing:
        def invoke(self, _query: str):
            class Doc:
                def __init__(self, page_content):
                    self.page_content = page_content

            return [Doc("Basic: $29/month. Pro: $79/month.")]

    class PricingLLM:
        def __init__(self, classify_label: str):
            self.classify_label = classify_label

        def invoke(self, prompt: str):
            if "Classify user intent into exactly one label" in prompt:
                return DummyMsg(self.classify_label)
            if "Answer ONLY from provided context" in prompt:
                return DummyMsg("The Pro plan is $79/month and Basic is $29/month.")
            return DummyMsg("ok")

    monkeypatch.setattr(graph_mod, "build_retriever", lambda: RetrieverWithPricing())
    monkeypatch.setattr(graph_mod, "ChatGoogleGenerativeAI", lambda model: PricingLLM("high_intent"))

    calls = []

    def fake_capture(name, email, platform):
        calls.append((name, email, platform))
        return "Lead captured successfully"

    monkeypatch.setattr(graph_mod, "mock_lead_capture", fake_capture)

    app = graph_mod.build_graph()
    state = base_state()

    # Mixed intent in first turn.
    state["last_user_message"] = "Hi, what is your pricing? I want to buy."
    state["messages"].append({"role": "user", "content": state["last_user_message"]})
    state = app.invoke(state)
    assert state["pending_question"] != ""

    # Continue lead collection.
    for text in ["mawiya@example.com", "Instagram, please answer my question"]:
        state["last_user_message"] = text
        state["messages"].append({"role": "user", "content": text})
        state = app.invoke(state)

    assert len(calls) == 1
    assert calls[0][2] == "Instagram"
    assert state["lead_captured"] is True
    assert state["pending_question"] == ""
    assert "79/month" in state["messages"][-1]["content"]
