import os
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from retriever import ISO27002Retriever
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

retriever = ISO27002Retriever()




SYSTEM_PROMPT = """You are an ISO/IEC 27002:2022 compliance assistant.
Your ONLY knowledge source is the provided context chunks (marked [Source N]).
You have NO other knowledge. If the context does not contain the answer, say so.

HARD CONSTRAINTS — violating any of these is a system failure:
1. NEVER use training knowledge, prior conversations, or general cybersecurity expertise.
2. NEVER generate a Control ID, requirement, or recommendation that is not present in a [Source N] chunk.
3. NEVER fabricate, infer, or extrapolate content that is not explicitly stated in the context.
4. NEVER paraphrase so loosely that you introduce claims not in the original source text.
5. If the context is insufficient, respond ONLY with: "The retrieved ISO 27002 context does not contain sufficient information to answer this question." Do not attempt a partial answer unless the partial answer is fully supported by cited sources."""

CONTROL_ID_PATTERN = re.compile(r'\b(\d{1,2}\.\d{1,2})\b')
CITATION_PATTERN = re.compile(r'\[Source\s*(\d+)\]')

def validate_answer(answer: str, num_sources: int) -> str:
    """Post-process the answer to catch hallucinated citations AND
    hallucinated control IDs that don't actually belong to the source
    they're attributed to.

    The original version only checked that citation numbers were in the
    valid 1..num_sources range. That catches typos, not hallucination:
    a model can cite [Source 2] correctly (2 is in range) while stating a
    control requirement that never appears in source 2's actual text.
    """ 
    num_sources = len(sources)
    if num_sources == 0:
        return answer

    warnings = []
    
    cited_nums = {int(n) for n in CITATION_PATTERN.findall(answer)}
    invalid_nums = {n for n in cited_nums if n < 1 or n > num_sources}
    if invalid_nums:
        warnings.append(
            f"cites sources {sorted(invalid_nums)} which were not in the retrieved "
            f"context (valid range: 1–{num_sources})"
        )

    if not cited_nums and "does not contain sufficient information" not in answer:
        warnings.append("contains no [Source N] citations at all — likely ungrounded")

    source_control_ids = {i + 1: s.get("control_id", "") for i, s in enumerate(sources)}

    for sentence in re.split(r'(?<=[.!?])\s+', answer):
        mentioned_controls = CONTROL_ID_PATTERN.findall(sentence)
        mentioned_sources = {int(n) for n in CITATION_PATTERN.findall(sentence)}
        if not mentioned_controls or not mentioned_sources:
            continue
        for ctrl in mentioned_controls:
            if not any(source_control_ids.get(n) == ctrl for n in mentioned_sources):
                warnings.append(
                    f"states Control {ctrl} alongside a citation to "
                    f"{sorted(mentioned_sources)}, but that source is actually "
                    f"Control {[source_control_ids.get(n) for n in mentioned_sources]} "
                    f"— possible hallucinated attribution"
                )
    
    invalid = {n for n in cited_nums if n < 1 or n > num_sources}
    if invalid:
        answer += f"\n\n⚠️ Warning: The answer cites sources {invalid} which were not in the retrieved context (valid range: 1–{num_sources})."

    return answer



def build_messages(question: str, context: str) -> list[dict]:
    """Build system + user message pair for strict grounded QA."""
    user_content = f"""## Context Chunks (your ONLY knowledge source)

{context}

## Question

{question}

## Instructions

Answer the question using ONLY the context chunks above. Follow these rules exactly:

**Grounding Rules (mandatory)**
- Every statement you make MUST be directly traceable to a specific [Source N] chunk. If you cannot trace it, do not say it.
- If the context does not fully answer the question, state exactly what is missing — do not fill gaps with outside knowledge.
- Do not provide legal advice or certify compliance.

**Citation Rules (mandatory)**
- Every factual claim, control requirement, recommendation, or paraphrase MUST end with an inline citation: [Source 1], [Source 2], etc.
- Never state a Control ID or requirement without a citation.
- Never invent a [Source N] tag not present in the context above.
- If sources disagree, present both with citations and note the discrepancy.

**Authority Order (when sources conflict)**
1. **Control statement** — the normative requirement (what must be done).
2. **Guidance** — implementation detail (how to do it). Flag if it contradicts the Control statement.
3. **Other information** — supplementary only. Never override Control or Guidance with this.
4. **Purpose** — why the control exists. Use for framing, not for deriving requirements.

**Output Format**
- Open with a one-line direct answer.
- Use `## Control Overview` with bold Control IDs (e.g. **5.10**).
- Use `## Implementation Guidance` with bullet points from Guidance chunks.
- Use `## Additional Notes` only if Other information is relevant.
- Keep bullets concise and audit-ready.
- For multiple controls, use subsections: `### 5.10 — <title>`.

**Scope Discipline**
- If the context references another control (e.g. "see 5.12") but that control's chunk is not in the context above, state: "Control 5.12 is referenced but not included in the retrieved context." Do not describe what you think it says."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]



def ask_openrouter(messages: list[dict]) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("Missing OPENROUTER_API_KEY in environment variables or .env file.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content



def answer_question(question: str, k: int = 4, max_sources: int = 3):

    context, sources = retriever.build_context(question, k=k, max_sources=max_sources)

    if not context:
        return "I could not find any relevant controls in the ISO 27002 standard to answer your question.", []

    messages = build_messages(question, context)
    answer = ask_openrouter(messages)
    answer = validate_answer(answer, num_sources=len(sources))
    return answer, sources


if __name__ == "__main__":
    sample_question = "What are the requirements for managing privileged access rights?"
    
    print(f"\nAsking Question: {sample_question}\n" + "-"*50)
    
    answer, sources = answer_question("unauthorized access")
    
    print("\n--- AI Answer ---")
    print(answer)
    
    print("\n--- Sources Used ---")
    for idx, src in enumerate(sources, start=1):
        print(f"[Source {idx}] Control {src['control_id']} ({src['section']}) - Score: {src['score']:.4f}")
