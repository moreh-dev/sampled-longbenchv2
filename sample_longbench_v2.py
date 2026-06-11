#!/usr/bin/env python3
"""Sample real long-context prompts from LongBench-v2 for `vllm bench serve`.

Output is the vLLM **custom** dataset format: a JSONL file with one
``{"prompt": "..."}`` object per line. Each prompt is built from the official
LongBench-v2 0-shot template (context + question + 4 choices) and the context is
truncated so the prompt tokenizes to *exactly* the target input length (ISL)
under the GLM-5 tokenizer, with no special tokens added -- matching what
``vllm bench serve --dataset-name custom --skip-chat-template`` measures.

The dataset controls ISL only. Output length (OSL) is NOT encoded here; set it at
serve time with ``--custom-output-len <OSL>`` (combined with ``--ignore-eos``).

Why `custom` and not `sharegpt`: vLLM's ShareGPTDataset hard-filters prompts to
<=1024 tokens (`is_valid_sequence`), so it silently drops every long-context
sample. CustomDataset applies no length filter.

Memory note: data.json is ~465MB but the host has little RAM, so the dataset is
streamed with ijson (one entry in memory at a time) and prompts are written
incrementally.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import ijson
from transformers import AutoTokenizer, PreTrainedTokenizerFast

# Official LongBench-v2 0-shot prompt, split around the {context} slot.
TEMPLATE_PREFIX = "Please read the following text and answer the questions below.\n\n"
TEMPLATE_SUFFIX = (
    "\n\nWhat is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n\n"
    'Format your response as follows: "The correct answer is (insert answer here)".'
)

# name, target ISL (tokens), recommended OSL (serve-time only), num prompts
DEFAULT_CONFIGS = [
    ("8k", 8192, 1024, 256),
    ("10k", 10000, 500, 256),
    ("100k", 100000, 500, 100),
    ("1M", 1000000, 500, 32),
]

# Conservative lower bound on tokens/word, used only to pre-filter candidate
# entries cheaply. The real token count is always verified by the tokenizer.
TOK_PER_WORD_MIN = 1.5
CHARS_PER_TOK = 4.5  # GLM-5 on this data ~4.48; used to size the context pre-slice


def load_tokenizer(path: str):
    if os.path.isfile(path) and path.endswith(".json"):
        return PreTrainedTokenizerFast(tokenizer_file=path)
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


def build_suffix(entry: dict) -> str:
    return TEMPLATE_SUFFIX.format(
        question=entry.get("question", ""),
        choice_A=entry.get("choice_A", ""),
        choice_B=entry.get("choice_B", ""),
        choice_C=entry.get("choice_C", ""),
        choice_D=entry.get("choice_D", ""),
    )


def make_prompt(tok, prefix_len: int, entry: dict, isl: int, tol: int = 0, max_iters: int = 6):
    """Return (prompt, achieved_tokens) tokenizing to ~exactly ``isl`` tokens, or None.

    Refinement stops as soon as the re-tokenized prompt is within ``tol`` tokens of
    ``isl``; otherwise the closest result across all iterations is returned.
    None means the entry's context is too short to fill the ISL budget.
    """
    context = entry.get("context") or ""
    suffix = build_suffix(entry)
    suffix_len = n_tokens(tok, suffix)
    budget = isl - prefix_len - suffix_len
    if budget <= 0:
        return None

    # Pre-slice the context by characters so we never tokenize a multi-million
    # token document just to keep its first `budget` tokens. Grow until we have
    # a small margin over budget, or until we've consumed the whole context.
    need_chars = int(budget * CHARS_PER_TOK * 1.3) + 2048
    while True:
        ctx_ids = tok(context[:need_chars], add_special_tokens=False).input_ids
        if len(ctx_ids) >= budget + 64 or need_chars >= len(context):
            break
        need_chars = min(len(context), need_chars * 2)

    if len(ctx_ids) < budget:
        return None  # context genuinely too short for this ISL

    # Refine: decode the first `b` context tokens, reassemble, re-tokenize the
    # whole prompt, and nudge `b` by the residual until the *re-tokenized* prompt
    # hits the target exactly (decode->encode can drift a few tokens at edges).
    b = budget
    best = None
    for _ in range(max_iters):
        b = max(1, min(b, len(ctx_ids)))
        ctx_text = tok.decode(ctx_ids[:b])
        prompt = TEMPLATE_PREFIX + ctx_text + suffix
        n = n_tokens(tok, prompt)
        if best is None or abs(isl - n) < abs(isl - best[1]):
            best = (prompt, n)
        if abs(isl - n) <= tol:
            return prompt, n
        b += isl - n
    return best


def scan_word_counts(dataset_path: str) -> list[int]:
    """Pass 1: stream the file and record per-entry word counts (cheap)."""
    counts: list[int] = []
    with open(dataset_path, "rb") as f:
        for obj in ijson.items(f, "item"):
            ctx = obj.get("context") or ""
            counts.append(ctx.count(" ") + 1)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="/remote/vast0/share-mv/zai-org/LongBench-v2/data.json")
    ap.add_argument("--output-dir", default=os.path.expanduser("~/workspace/data"))
    ap.add_argument("--tokenizer", default="/remote/vast0/share-mv/zai-org/GLM-5-FP8/tokenizer.json",
                    help="Path to a tokenizer.json file or a HF model dir.")
    ap.add_argument("--prefix", default="longbenchv2",
                    help="Output filename prefix: <prefix>-<name>.jsonl")
    ap.add_argument("--tolerance", type=int, default=-1,
                    help="Accept a prompt if |achieved_tokens - ISL| <= this. "
                         "-1 (default) = auto per-config: max(4, round(ISL*0.0005)) (~0.05%%). "
                         "Use 0 to keep only token-exact prompts.")
    args = ap.parse_args()

    configs = DEFAULT_CONFIGS
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tok = load_tokenizer(args.tokenizer)
    prefix_len = n_tokens(tok, TEMPLATE_PREFIX)

    print(f"Pass 1/2: scanning word counts in {args.dataset} ...", flush=True)
    words = scan_word_counts(args.dataset)
    print(f"  {len(words)} entries", flush=True)

    # Candidate index sets per config (pre-filter on word count).
    cand = {}
    for name, isl, _osl, _n in configs:
        min_words = int(isl / TOK_PER_WORD_MIN)
        idxs = {i for i, w in enumerate(words) if w >= min_words}
        cand[name] = idxs
        print(f"  [{name}] ISL={isl}: {len(idxs)} candidate entries (>= {min_words} words)", flush=True)
    needed = set().union(*cand.values()) if cand else set()

    # Open output files and stream pass 2.
    files = {name: open(out_dir / f"{args.prefix}-{name}.jsonl", "w", encoding="utf-8")
             for name, *_ in configs}
    counts = {name: 0 for name, *_ in configs}
    achieved = {name: [] for name, *_ in configs}
    target = {name: (isl, osl, n) for name, isl, osl, n in configs}
    done = {name: False for name, *_ in configs}

    print("Pass 2/2: building prompts ...", flush=True)
    with open(args.dataset, "rb") as f:
        for i, obj in enumerate(ijson.items(f, "item")):
            if i not in needed:
                continue
            if all(done.values()):
                break
            for name, isl, osl, npr in configs:
                if done[name] or i not in cand[name]:
                    continue
                tol = args.tolerance if args.tolerance >= 0 else max(4, round(isl * 0.0005))
                res = make_prompt(tok, prefix_len, obj, isl, tol=tol)
                if res is None:
                    continue
                prompt, got = res
                if abs(got - isl) > tol:
                    continue
                rec = {
                    "prompt": prompt,
                    "input_tokens": got,
                    "target_isl": isl,
                    "_id": obj.get("_id"),
                    "domain": obj.get("domain"),
                    "sub_domain": obj.get("sub_domain"),
                    "difficulty": obj.get("difficulty"),
                    "source_length": obj.get("length"),
                    "source_words": words[i],
                }
                files[name].write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[name] += 1
                achieved[name].append(got)
                if counts[name] >= npr:
                    done[name] = True
            collected = {k: counts[k] for k in counts}
            if i % 50 == 0:
                print(f"  scanned {i+1} entries | collected {collected}", flush=True)

    for fh in files.values():
        fh.close()

    # Manifest + summary
    manifest = []
    print("\n=== Summary ===", flush=True)
    for name, isl, osl, npr in configs:
        a = achieved[name]
        path = str(out_dir / f"{args.prefix}-{name}.jsonl")
        info = {
            "name": name,
            "file": path,
            "target_isl": isl,
            "recommended_osl": osl,
            "requested_prompts": npr,
            "produced_prompts": counts[name],
            "achieved_tokens_min": min(a) if a else None,
            "achieved_tokens_max": max(a) if a else None,
            "candidate_pool": len(cand[name]),
        }
        manifest.append(info)
        warn = "" if counts[name] >= npr else "  <-- fewer than requested (candidate pool exhausted)"
        rng = f"{min(a)}..{max(a)}" if a else "n/a"
        print(f"  {name:>5}: {counts[name]:>4}/{npr} prompts | ISL exact={isl} got={rng} "
              f"| OSL(serve)={osl}{warn}", flush=True)

    man_path = out_dir / f"{args.prefix}-manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {man_path}", flush=True)
    print("\nServe example (set OSL here, not in the dataset):", flush=True)
    print(
        "  vllm bench serve --backend vllm --dataset-name custom \\\n"
        f"      --dataset-path {out_dir / (args.prefix + '-8k.jsonl')} \\\n"
        "      --skip-chat-template --custom-output-len 1024 --ignore-eos \\\n"
        "      --model <served-model> --tokenizer "
        f"{args.tokenizer} \\\n"
        "      --num-prompts 256 --max-concurrency <C>",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
