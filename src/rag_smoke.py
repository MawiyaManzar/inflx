from rag import build_retriever

def run():
    retriever = build_retriever()

    queries = [
        "What is the Pro plan price?",
        "What is your refund policy?",
        "Which plan has 24/7 support?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        docs = retriever.invoke(q)
        for i, d in enumerate(docs, 1):
            print(f"[{i}] {d.page_content[:180]}...")

if __name__ == "__main__":
    run()