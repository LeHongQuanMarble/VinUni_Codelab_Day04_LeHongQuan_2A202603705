"""Chat realtime trong terminal với ToolCallingAgent (template.py).

Chạy:
    python3 chat.py            # chat với Agent (có tool calling)
    python3 chat.py --baseline # chat với ChatbotBaseline (không tool)

Gõ 'exit' hoặc 'quit' (hoặc Ctrl+C / Ctrl+D) để thoát.
"""

import sys

from template import ChatbotBaseline, ToolCallingAgent


def main() -> None:
    use_baseline = "--baseline" in sys.argv

    if use_baseline:
        bot = ChatbotBaseline()
        label = "ChatbotBaseline"
    else:
        bot = ToolCallingAgent(max_iterations=5)
        label = "ToolCallingAgent"

    print(f"=== Chat với {label} ===")
    print("Gõ 'exit' để thoát.\n")

    while True:
        try:
            user_input = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Tạm biệt!")
            break

        if use_baseline:
            result = bot.query(user_input)
        else:
            result = bot.run(user_input)

        print(f"Bot: {result['answer']}\n")


if __name__ == "__main__":
    main()
