from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .retriever import ISO27002Retriever
except ImportError:
    # Supports running prompt.py directly or importing it from Streamlit.
    from retriever import ISO27002Retriever

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

INSUFFICIENT_CONTEXT_RESPONSE = (
    "The retrieved ISO 27002 context does not contain sufficient information "
    "to answer this question."
)

retriever = ISO27002Retriever()

SYSTEM_PROMPT = f"""You are an ISO/IEC 27002:2022 compliance assistant.
Your ONLY knowledge source is the supplied context chunks marked [SOURCE N].
You have no authority to use training knowledge, previous conversations, or
outside cybersecurity knowledge.

HARD CONSTRAINTS:
1. Use only facts explicitly stated in the supplied [SOURCE N] chunks.
2. Never invent a Control ID, control title, requirement, recommendation, or citation.
3. Every factual claim and every mentioned Control ID must have an inline [SOURCE N] citation.
4. You may give a partial answer when the supported portion is useful and fully cited.
5. If none of the supplied sources directly addresses the query, respond only with:
   "{INSUFFICIENT_CONTEXT_RESPONSE}"
6. A short keyword or topic query, such as "unauthorized access", means:
   identify which supplied controls address that topic and summarize only what
   the supplied chunks explicitly say about it.
7. Do not provide legal advice, certify compliance, or infer unstated obligations.
"""

CONTROL_ID_PATTERN = re.compile(r"\b(\d{1,2}\.\d{1,2})\b")
CITATION_PATTERN = re.compile(r"\[SOURCE\s*(\d+)\]", re.IGNORECASE)


def validate_answer(answer: str, sources: list[dict[str, Any]]) -> str:
    """Return the answer with explicit diagnostics when citation checks fail."""
    cleaned_answer = (answer or "").strip()
    if not cleaned_answer:
        return INSUFFICIENT_CONTEXT_RESPONSE

    if cleaned_answer == INSUFFICIENT_CONTEXT_RESPONSE:
        return cleaned_answer

    number_of_sources = len(sources)
    if number_of_sources == 0:
        return INSUFFICIENT_CONTEXT_RESPONSE

    warnings: list[str] = []
    cited_numbers = {int(value) for value in CITATION_PATTERN.findall(cleaned_answer)}
    invalid_numbers = {
        value
        for value in cited_numbers
        if value < 1 or value > number_of_sources
    }

    if invalid_numbers:
        warnings.append(
            "Cites source numbers not included in the retrieved context: "
            f"{sorted(invalid_numbers)}."
        )

    if not cited_numbers:
        warnings.append("The generated answer contains no [SOURCE N] citation.")

    source_control_ids = {
        index + 1: str(source.get("control_id", ""))
        for index, source in enumerate(sources)
    }

    for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned_answer):
        mentioned_controls = CONTROL_ID_PATTERN.findall(sentence)
        mentioned_sources = {
            int(value) for value in CITATION_PATTERN.findall(sentence)
        }
        if not mentioned_controls or not mentioned_sources:
            continue

        for control_id in mentioned_controls:
            valid_attribution = any(
                1 <= source_number <= number_of_sources
                and source_control_ids.get(source_number) == control_id
                for source_number in mentioned_sources
            )
            if not valid_attribution:
                cited_control_ids = [
                    source_control_ids.get(source_number, "unknown")
                    for source_number in sorted(mentioned_sources)
                ]
                warnings.append(
                    f"Control {control_id} is cited against source control(s) "
                    f"{cited_control_ids}."
                )

    if warnings:
        unique_warnings = list(dict.fromkeys(warnings))
        warning_text = "\n".join(f"- {warning}" for warning in unique_warnings)
        cleaned_answer += f"\n\n> Validation warning\n{warning_text}"

    return cleaned_answer


def build_messages(question: str, context: str) -> list[dict[str, str]]:
    """Build a grounded system/user message pair."""
    user_content = f"""## Context Chunks

{context}

## User Query

{question}

## Required Reasoning Policy

First determine whether at least one supplied chunk directly addresses the
query or topic. Keyword overlap can be evidence of relevance, but only answer
with the meaning explicitly supported by the chunk.

- If one or more chunks directly address the query, answer the supported part.
- Do not refuse merely because the query is a short phrase rather than a full question.
- If the chunks cover only part of a broader question, answer that supported part
  and briefly state which requested detail is not present.
- If no supplied chunk directly addresses the query, output exactly:
  {INSUFFICIENT_CONTEXT_RESPONSE}

## Citation Rules

- End every factual claim or paraphrase with its supporting [SOURCE N] citation.
- Never cite a source number that does not appear above.
- Never mention a Control ID without citing the source containing that same ID.
- If multiple sources support one sentence, cite all applicable sources.

## Authority Order

When supplied chunks conflict, prioritize in this order:
1. Control statement
2. Guidance
3. Other information
4. Purpose

Do not infer missing content from another control merely because it is referenced.

## Output Format

- Begin with a direct one-sentence answer.
- For a keyword/topic query, identify the relevant retrieved control(s) and state
  exactly how the supplied chunk relates to the topic.
- Use `## Control Overview` when at least one control is relevant.
- Use `## Implementation Guidance` only when a supplied Guidance chunk supports it.
- Use `## Additional Notes` only when supplied Other information or Purpose text is relevant.
- Omit unsupported sections instead of refusing the whole answer.
- Keep the answer concise and audit-ready.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def ask_openrouter(messages: list[dict[str, str]]) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "Missing OPENROUTER_API_KEY in environment variables, .env, "
            "or Streamlit secrets."
        )

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

    content = response.choices[0].message.content
    return content.strip() if content else ""


def answer_question(
    question: str,
    k: int = 12,
    max_sources: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    clean_question = question.strip()
    if not clean_question:
        return "Please enter a question about ISO/IEC 27002:2022.", []

    # update for retriever: use a wider candidate pool and let the reranker
    # select sources after reranking, filtering, and deduplication.
    context, sources = retriever.build_context(
        clean_question,
        k=k,
        max_sources=max_sources,
    
    )

    if not context or not sources:
        return INSUFFICIENT_CONTEXT_RESPONSE, []

    messages = build_messages(clean_question, context)
    answer = ask_openrouter(messages)
    return validate_answer(answer, sources), sources


if __name__ == "__main__":
    sample_question = "unauthorized access"
    print(f"\nAsking Question: {sample_question}\n" + "-" * 50)

    generated_answer, used_sources = answer_question(sample_question)

    print("\n--- AI Answer ---")
    print(generated_answer)

    print("\n--- Sources Used ---")
    for index, source in enumerate(used_sources, start=1):
        print(
            f"[SOURCE {index}] Control {source['control_id']} "
            f"({source['section']}) | embedding={source['embedding_score']:.4f} "
            f"| rerank_logit={source['rerank_logit']:.4f} "
            f"| rerank_score={source['rerank_score']:.4f}"
        )