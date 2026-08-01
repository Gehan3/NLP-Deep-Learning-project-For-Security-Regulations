from prompt import answer_question  # استيراد دالة معالجة السؤال والـ LLM من ملف prompt.py

def main():
    print("=" * 80)
    print("🤖 ISO/IEC 27002:2022 Compliance Chatbot (Powered by BGE-M3, Reranker & OpenRouter LLM)")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 80)

    while True:
        # استقبال السؤال من المستخدم
        question = input("\n📝 Enter your compliance question: ").strip()

        # أمر الخروج من البرنامج
        if question.lower() in ["exit", "quit"]:
            print("\nGoodbye! Stay secure and compliant. 🔒")
            break

        if not question:
            print("Please enter a valid question.")
            continue

        print("\n⏳ Processing search, reranking, and generating auditor-grade answer via LLM...")

        try:
            # استدعاء دالة answer_question التي تبحث، تعيد الترتيب، تبني البرومبت، وتتحدث مع الـ LLM
            answer, sources = answer_question(question, k=10, max_sources=3)

            # طباعة إجابة الـ LLM النهائية
            print("\n" + "=" * 40 + " AI Auditor Answer " + "=" * 40)
            print(answer)
            print("=" * 99)

            # طباعة المصادر للتحقق من الشفافية
            print("\n📌 --- Retrieved Sources Used ---")
            if sources:
                for idx, src in enumerate(sources, start=1):
                    score_val = src.get('rerank_score', src.get('score', 0.0))
                    print(f"  [Source {idx}] Control: {src['control_id']} | Section: {src['section'].upper()} | Score: {score_val:.4f}")
            else:
                print("  No sources were utilized.")

        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()