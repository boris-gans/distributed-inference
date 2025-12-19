## Prompts
We have 2 files with 20 prompts each, one file for prompts of about 2k tokens and the other for prompts of 4k tokens. Each prompt asks for the exact same operation, but the 4k versions are just slighly more verbose. This way we can explore model performance with larger attention heads


## Deterministic Correctness Checks

1. Logit match test on 20 fixed prompts: compare per-token logits against the baseline run, DeepSeek v3.1 hosted on DeepSeek servers
    - Using the same hyperparameters: Greedy decoding (temperature=0, top_p=1, top_k=0, do_sample=False), dropout disabled, fixed seeds, identical tokenizer

    - Report mean absolute error and max absolute error.

2. Batch-size invariance: same prompts run as batch=1 vs batch=8 should produce identical outputs under greedy.

3. Multi-node invariance: 1-node vs 2-node sharding yields identical outputs.