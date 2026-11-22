"""
Model testing report for EduCore.
Tests each model with a minimal call and records success/latency/errors.
Runs with the ml env Python (has all dependencies).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(".env")

RESULTS: dict[str, dict] = {}


def record(model: str, mtype: str, status: str, latency_ms: float | None = None,
           error: str | None = None, sample: str | None = None,
           detail: dict | None = None) -> None:
    RESULTS[model] = {
        "type": mtype,
        "status": status,
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "error": error,
        "sample": sample,
        "detail": detail or {},
        "tested_at": datetime.utcnow().isoformat() + "Z",
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. Groq LLM — primary model
# ═══════════════════════════════════════════════════════════════════════════
def test_groq_primary() -> None:
    model = "openai/gpt-oss-120b"
    try:
        os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY", "")
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )
        start = time.perf_counter()
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'ok' in one word."}],
            max_completion_tokens=10,
            temperature=0.0,
        )
        latency = (time.perf_counter() - start) * 1000
        content = completion.choices[0].message.content or ""
        record(
            model, "llm", "PASS", latency_ms=latency,
            sample=content,
            detail={
                "model": getattr(completion, "model", model),
                "usage": getattr(completion, "usage", None),
            },
        )
    except Exception as e:
        record(model, "llm", "FAIL", error=f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Groq LLM — fallback model
# ═══════════════════════════════════════════════════════════════════════════
def test_groq_fallback() -> None:
    model = "openai/gpt-oss-20b"
    try:
        os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY", "")
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )
        start = time.perf_counter()
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'ok' in one word."}],
            max_completion_tokens=10,
            temperature=0.0,
        )
        latency = (time.perf_counter() - start) * 1000
        content = completion.choices[0].message.content or ""
        record(
            model, "llm", "PASS", latency_ms=latency,
            sample=content,
            detail={"model": getattr(completion, "model", model)},
        )
    except Exception as e:
        record(model, "llm", "FAIL", error=f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Embedding model (local, sentence-transformers)
# ═══════════════════════════════════════════════════════════════════════════
def test_embedding() -> None:
    model = "all-MiniLM-L6-v2"
    try:
        from sentence_transformers import SentenceTransformer
        start = time.perf_counter()
        emb = SentenceTransformer(model, device="cpu")
        load_ms = (time.perf_counter() - start) * 1000
        texts = ["What is NLP?", "The cat sat on the mat.", "العربية لغة جميلة."]
        start = time.perf_counter()
        vectors = emb.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        enc_ms = (time.perf_counter() - start) * 1000
        dim = int(vectors.shape[1])
        record(
            model, "embedding", "PASS",
            latency_ms=round(load_ms + enc_ms, 2),
            detail={
                "dimension": dim,
                "texts_encoded": len(texts),
                "encode_ms": round(enc_ms, 2),
                "load_ms": round(load_ms, 2),
                "shape": list(vectors.shape),
            },
            sample=f"{dim}-dim vectors for {len(texts)} texts",
        )
    except Exception as e:
        record(model, "embedding", "FAIL", error=f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Gemini image model (google-genai)
# ═══════════════════════════════════════════════════════════════════════════
def test_gemini_image() -> None:
    model = os.getenv("GOOGLE_VISION_MODEL", "gemini-flash-latest")
    try:
        from PIL import Image
        import io
        from google import genai
        from google.genai import types

        # Build a tiny valid PNG image in-memory (no file download)
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        start = time.perf_counter()
        response = client.models.generate_content(
            model=model,
            contents=["What color is this image?", types.Part.from_bytes(data=img_bytes, mime_type="image/png")],
            config=types.GenerateContentConfig(max_output_tokens=20, temperature=0.0),
        )
        latency = (time.perf_counter() - start) * 1000
        text = getattr(response, "text", None) or ""
        record(
            model, "image", "PASS", latency_ms=latency,
            sample=text,
            detail={"image_bytes": len(img_bytes), "image_size": "64x64"},
        )
    except Exception as e:
        record(model, "image", "FAIL", error=f"{type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Provider lock — OpenRouter rejected for text generation
# ═══════════════════════════════════════════════════════════════════════════
def test_provider_lock() -> None:
    model = "provider-lock (Groq-only)"
    try:
        from educore.provider import provider_settings
        provider, key, base = provider_settings()
        if provider == "groq":
            record(
                model, "config", "PASS",
                detail={"provider": provider, "base_url": base,
                        "key_present": bool(key)},
                sample="Text generation locked to Groq provider",
            )
        else:
            record(model, "config", "FAIL",
                   error=f"Provider is '{provider}', expected 'groq'")
    except Exception as e:
        record(model, "config", "FAIL", error=f"{type(e).__name__}: {e}")


def main() -> None:
    print("=" * 60)
    print("EduCore Model Testing Report")
    print("=" * 60)

    tests = [
        ("Groq primary", test_groq_primary),
        ("Groq fallback", test_groq_fallback),
        ("Embedding", test_embedding),
        ("Gemini image", test_gemini_image),
        ("Provider lock", test_provider_lock),
    ]

    for name, fn in tests:
        print(f"\n[TEST] {name} ...")
        try:
            fn()
        except Exception as e:
            print(f"  CRASH: {type(e).__name__}: {e}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<40} {'Type':<14} {'Status':<8} {'Latency(ms)':>12}")
    print("-" * 74)
    for model, r in RESULTS.items():
        print(f"{model:<40} {r['type']:<14} {r['status']:<8} "
              f"{str(r['latency_ms']):>12}")
        if r["error"]:
            print(f"{'':<40}   ERROR: {r['error']}")
        if r["sample"]:
            print(f"{'':<40}   SAMPLE: {r['sample'][:80]}")

    # Write markdown report
    lines = ["# EduCore Model Testing Report", "",
             f"Generated: {datetime.utcnow().isoformat()}Z", "",
             "| Model | Type | Status | Latency (ms) | Error | Sample |",
             "|-------|------|--------|--------------|-------|--------|"]
    for model, r in RESULTS.items():
        error = (r["error"] or "").replace("|", "\\|")
        sample = (r["sample"] or "").replace("|", "\\|")[:60]
        lines.append(
            f"| {model} | {r['type']} | {r['status']} | "
            f"{r['latency_ms'] or '-'} | {error} | {sample} |"
        )
    lines.append("")
    report = "\n".join(lines)
    out = Path("MODEL_TEST_REPORT.md")
    out.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out.resolve()}")


if __name__ == "__main__":
    main()
