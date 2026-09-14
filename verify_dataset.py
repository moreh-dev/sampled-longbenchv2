#!/usr/bin/env python3
"""Verify the JSONL files load through vLLM's own CustomDataset (the exact class
`vllm bench serve --dataset-name custom` uses) and produce the target ISL.
OSL is set by `output_len` (== the serve-time `--custom-output-len`)."""
import os
from transformers import PreTrainedTokenizerFast
from vllm.benchmarks.datasets import CustomDataset

BASE = os.path.dirname(os.path.abspath(__file__))  # dataset files live next to this script

def find(name):
    for p in (os.path.join(BASE, f"longbenchv2-{name}.jsonl"),
              os.path.join(BASE, "data", f"longbenchv2-{name}.jsonl")):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"longbenchv2-{name}.jsonl not found near {BASE}")

tok = PreTrainedTokenizerFast(tokenizer_file="/remote/vast0/share-mv/zai-org/GLM-5-FP8/tokenizer.json")

for name, isl in [("4k", 4096), ("8k", 8192), ("10k", 10000), ("64k", 65536),
                  ("100k", 100000), ("1M", 1000000)]:
    ds = CustomDataset(dataset_path=find(name), disable_shuffle=True)
    reqs = ds.sample(tokenizer=tok, num_requests=8, output_len=500, skip_chat_template=True)
    plen = [r.prompt_len for r in reqs]
    print(f"{name:>5}: {len(reqs)} reqs | prompt_len {min(plen)}..{max(plen)} (target {isl}) "
          f"| expected_output_len={reqs[0].expected_output_len}")
