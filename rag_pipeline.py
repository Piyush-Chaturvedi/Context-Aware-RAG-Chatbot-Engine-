"""
Core RAG pipeline: retrieve relevant chunks, then (optionally) generate a
grounded answer with an LLM using strict anti-hallucination prompting.

Guardrails implemented here:
1. Relevance threshold -- if the best retrieved chunk scores too low,
   we don't even call the LLM; we tell the user we couldn't find it.
2. Strict system prompt -- the model is instructed to answer ONLY from the
   provided context and to explicitly say when it doesn't know.
3. Mandatory source citation -- every answer must reference which document
   it came from, so a human can verify it.
4. Retrieval-only fallback mode -- if no LLM API key is configured, the
   pipeline still works: it returns the best-matching manual excerpt
   directly, so the demo runs with zero API cost/setup.
"""
from src.config import (
    MODEL_PROVIDER,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
    MIN_RELEVANCE_SCORE,
)

SYSTEM_PROMPT = """You are a technical support assistant for an electronics company.

STRICT RULES (do not break these under any circumstances):
1. Answer ONLY using the information in the "CONTEXT" section below. Do not use any outside knowledge, even if you know the answer.
2. If the CONTEXT does not contain enough information to answer the question, say clearly: "I don't have this information in the available documents." Do NOT guess or make up an answer.
3. Every answer must end with a "Source:" line naming the document(s) the answer came from (given in the context metadata).
4. Be concise and practical. Use numbered steps for procedures/instructions.
5. Do not invent part numbers, specifications, voltages, or safety information that is not explicitly present in the context.
"""


def _format_context(retrieved):
    """retrieved: list of (chunk_dict, score)"""
    blocks = []
    for chunk, score in retrieved:
        blocks.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    return "\n\n---\n\n".join(blocks)


def _call_anthropic(query, context):
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_openai(query, context):
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {query}"
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


def answer_query(vector_store, query, top_k=TOP_K):
    """
    Main entry point. Returns a dict:
    {
        "answer": str,
        "sources": [source filenames used],
        "retrieved_chunks": [...],   # for debugging/UI display
        "mode": "llm" | "retrieval_only" | "no_match"
    }
    """
    retrieved = vector_store.query(query, top_k=top_k)

    if not retrieved or retrieved[0][1] < MIN_RELEVANCE_SCORE:
        return {
            "answer": (
                "I couldn't find anything relevant to this question in the available "
                "documents. Could you rephrase, or is this something support/engineering "
                "should look into directly?"
            ),
            "sources": [],
            "retrieved_chunks": retrieved,
            "mode": "no_match",
        }

    context = _format_context(retrieved)
    sources = sorted(set(c["source"] for c, _ in retrieved))

    if MODEL_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        answer = _call_anthropic(query, context)
        mode = "llm"
    elif MODEL_PROVIDER == "openai" and OPENAI_API_KEY:
        answer = _call_openai(query, context)
        mode = "llm"
    else:
        # Retrieval-only fallback: no LLM configured, so we return the
        # best-matching excerpt directly instead of generating free text.
        # This keeps the guarantee "never say something not in the docs".
        best_chunk, best_score = retrieved[0]
        answer = (
            f"(Retrieval-only mode — no LLM API key configured)\n\n"
            f"Most relevant excerpt (match score: {best_score:.2f}):\n\n"
            f"\"{best_chunk['text']}\"\n\n"
            f"Source: {best_chunk['source']}"
        )
        mode = "retrieval_only"

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved,
        "mode": mode,
    }
