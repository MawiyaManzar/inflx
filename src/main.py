from src.graph import build_graph

def initial_state():
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


def main() -> None:
    app = build_graph()
    state = initial_state()
    print("AutoStream Agent CLI")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Session closed.")
            break
        if not user_input:
            print("Agent: Please enter a message.")
            continue

        state["last_user_message"] = user_input
        state["messages"].append({"role": "user", "content": user_input})
        state = app.invoke(state)
        print(
            f"[debug] intent={state['intent']} missing={state['missing_fields']} "
            f"lead_captured={state['lead_captured']}"
        )
        print(f"Agent: {state['messages'][-1]['content']}")


if __name__ == "__main__":
    main()