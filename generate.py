"""
AccessTrap-CS — Corpus Generation Script (FIXED)
Generates responses: 30 concepts × 3 conditions × N models
"""

import csv
import json
import time
import os
import sys
import subprocess
import requests
from datetime import datetime

# CONFIG 

MODELS = [
    "mistral:7b",
    "ministral-3:14b",
    "gemma2:27b",          
    "llama3.3:70b",
    "deepseek-r1:70b",     
    "llama4:scout",        
    "qwen3:235b",
]

# Prompt templates — 3 conditions
PROMPTS = {
    "A_standard": (
        "Explain the following computer science concept accurately and clearly, "
        "suitable for a first-year computer science student.\n\n"
        "Concept: {concept}"
    ),
    "B_simplified": (
        "Explain the following computer science concept to a student who has "
        "reading difficulties. Use simplified language, short sentences, and "
        "concrete analogies instead of code or formal definitions. Avoid "
        "technical jargon where possible.\n\n"
        "Concept: {concept}"
    ),
    "C_highly_simplified": (
        "Explain the following computer science concept to a student with "
        "significant attention difficulties. Keep the explanation within five "
        "sentences. Use one single, vivid analogy that captures the core idea. "
        "Make it memorable and engaging.\n\n"
        "Concept: {concept}"
    ),
}

INPUT_CSV  = "ground_truth.csv"
OUTPUT_CSV = "corpus.csv"
LOG_FILE   = "generate_log.txt"

# Retry settings
MAX_RETRIES       = 3
RETRY_DELAY       = 5
MIN_RESPONSE_LEN  = 20    # responses shorter than this are treated as empty

# ── HELPERS ───────────────────────────────────────────────────────────────────

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_available_models():
    """Query Ollama for the list of actually installed models."""
    try:
        resp = requests.get(OLLAMA_TAGS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {m["name"] for m in data.get("models", [])}
    except Exception as e:
        log(f"FATAL: Cannot reach Ollama at {OLLAMA_TAGS}: {e}")
        log("Make sure Ollama is running: `ollama serve`")
        sys.exit(1)


def verify_models(available):
    """Check that every configured model is actually installed."""
    missing = [m for m in MODELS if m not in available]
    if missing:
        log("=" * 70)
        log("ERROR: The following configured models are NOT installed:")
        for m in missing:
            log(f"   - {m}")
        log("")
        log("Available models on this machine:")
        for m in sorted(available):
            log(f"   + {m}")
        log("")
        log("Fix: either `ollama pull <model>` or edit the MODELS list.")
        log("=" * 70)
        sys.exit(1)
    log(f"Pre-flight OK: all {len(MODELS)} models are available.")


def load_concepts(path):
    concepts = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            concepts.append({
                "id":           row["ID"].strip(),
                "domain":       row["Domain"].strip(),
                "concept":      row["Concept"].strip(),
                "ground_truth": row["Ground_Truth"].strip(),
                "reference":    row["Reference"].strip(),
            })
    return concepts


def load_done(path):
    """Return set of (concept_id, condition, model) already generated
    with a NON-EMPTY response."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resp = (row.get("response") or "").strip()
            if len(resp) >= MIN_RESPONSE_LEN:
                done.add((row["concept_id"], row["condition"], row["model"]))
    return done


def query_ollama(model, prompt, timeout=600):
    """Send a single request to Ollama. Returns response text or raises."""
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 512,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return (data.get("response") or "").strip()


def query_with_retry(model, prompt):
    """Try up to MAX_RETRIES times to get a non-empty response."""
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text = query_ollama(model, prompt)
            if len(text) >= MIN_RESPONSE_LEN:
                return text
            log(f"  attempt {attempt}/{MAX_RETRIES}: empty/short response "
                f"({len(text)} chars), retrying...")
            last_err = f"empty response ({len(text)} chars)"
        except Exception as e:
            log(f"  attempt {attempt}/{MAX_RETRIES}: {e}")
            last_err = str(e)
        time.sleep(RETRY_DELAY)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_err}")


def append_row(path, row, write_header):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


#  MAIN 

def main():
    available = get_available_models()
    verify_models(available)

    concepts = load_concepts(INPUT_CSV)
    done     = load_done(OUTPUT_CSV)

    total = len(concepts) * len(PROMPTS) * len(MODELS)

    log(f"Plan: {len(concepts)} concepts x {len(PROMPTS)} conditions x "
        f"{len(MODELS)} models = {total} responses")
    log(f"Already done (with valid response): {len(done)}")

    write_header = not os.path.exists(OUTPUT_CSV)
    generated    = 0
    errors       = 0

    for concept in concepts:
        for condition, template in PROMPTS.items():
            prompt = template.format(concept=concept["concept"])

            for model in MODELS:
                key = (concept["id"], condition, model)

                if key in done:
                    continue

                log(f"Generating -> {concept['id']} | {condition} | {model}")

                try:
                    response = query_with_retry(model, prompt)

                    row = {
                        "concept_id":   concept["id"],
                        "domain":       concept["domain"],
                        "concept":      concept["concept"],
                        "condition":    condition,
                        "model":        model,
                        "prompt":       prompt,
                        "response":     response,
                        "ground_truth": concept["ground_truth"],
                        "reference":    concept["reference"],
                        "timestamp":    datetime.now().isoformat(),
                    }
                    append_row(OUTPUT_CSV, row, write_header)
                    write_header = False
                    generated += 1
                    time.sleep(1)

                except Exception as e:
                    log(f"  FAILED {key}: {e}")
                    errors += 1

    log("=" * 70)
    log(f"DONE.  Generated: {generated}  |  Errors: {errors}")
    log(f"Output: {OUTPUT_CSV}")
    log("=" * 70)


if __name__ == "__main__":
    main()