We use Meta's 3B parameter model from their OpenLlama series. You can download it from: https://huggingface.co/openlm-research/open_llama_3b_v2 , or via the HuggingFace CLI with 

```bash
hf download openlm-research/open_llama_3b_v2 --include="*" --local-dir ~/[path-to-project]/models
```

*Note: you must authenticate your device with HF first*


New option: https://huggingface.co/facebook/opt-125m