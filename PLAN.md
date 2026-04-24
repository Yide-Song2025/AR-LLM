# Plan: OpenRouter API Evaluation Script for Telecom Datasets

## Context

This project evaluates LLM routing strategies. The existing codebase uses pre-computed accuracy from HuggingFace leaderboards (`extract_openllm_leaderboard_data.py`). This new script evaluates the same 16 candidate models via OpenRouter's API on two telecom-domain datasets, producing output compatible with the existing pipeline.

## Output Format

Matches `data/model_data/extracted_dataset_samples.jsonl`:
```json
{"query": "...", "answer": "...", "model": "...", "dataset": "...", "subset": "...", "correct": 1.0, "query_id": "..."}
```

## Output Files

| File | Content |
|------|---------|
| `data/model_data/teleqna_openrouter_results.jsonl` | All TeleQnA results (train + test) |
| `data/model_data/teleqna_test_openrouter_results.jsonl` | TeleQnA test split only (separate) |
| `data/model_data/telequad_openrouter_results.jsonl` | All TeleQuAD results |

## Candidate Models

All 16 models from `data/model_data/models_info.json`. Used as OpenRouter model IDs (OpenRouter hosts open-source models like Qwen, Llama, DeepSeek).

```python
MODELS = [
    "MaziyarPanahi/calme-3.2-instruct-78b",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]
```

## Dataset Structures

### TeleQnA (`ymoslem/TeleQnA-processed` from HuggingFace)

- **Splits:** `train` (9000 rows), `test` (1000 rows) — both collected
- **Columns:**
  - `question` (string) — question text
  - `choices` (List[string]) — answer choices
  - `answer` (int64) — 0-based index of correct choice
  - `subject` (string) — subject category → maps to `subset` in output
  - `explanation` (string) — not used in input
  - `prompt` (string) — NOT used; we use our own template
- **Input:** question + choices only
- **Ground truth answer:** `choices[answer]` (the choice text)

### TeleQuAD (`data/TeleQuAD-v4-full.json`)

- **Structure:** `data[].paragraphs[].qas[]`
- Each paragraph has `context` (passage text) and `qas` (questions array)
- Each QA: `{id, question, answers: [{text}], is_impossible}`
- **Input:** context + question
- **Ground truth answer:** `answers[0]["text"]`
- **Subset:** source field or `"telequad"`

## Prompt Templates

Conversational and concise — no overly specific instructions.

```python
# TeleQnA — multiple choice
TELEQNA_TEMPLATE = """{question}

{choices_formatted}

Reply with just the answer inside /box{{}}."""

# choices_formatted = "\n".join(f"{i}. {choice}" for i, choice in enumerate(choices))

# TeleQuAD — reading comprehension
TELEQUAD_TEMPLATE = """Context: {context}

Question: {question}

Reply with just the answer inside /box{{}}."""
```

## Implementation: `evaluate_openrouter_models.py`

### Architecture

```
Load datasets
    ↓
For each model (sequential):
    ├── Probe model availability → skip if 404/unavailable
    │
    ├── Process TeleQnA train+test (parallel, 32 threads):
    │   ├── Format prompt: question + numbered choices
    │   ├── Call OpenRouter API
    │   ├── Extract /box{} answer
    │   └── Direct match: compare extracted answer to choices[answer]
    │       (match by index number or by choice text)
    │
    └── Process TeleQuAD (parallel, 32 threads):
        ├── Format prompt: context + question
        ├── Call OpenRouter API
        ├── Extract /box{} answer
        └── DeepSeek API evaluates semantic match with reference
    ↓
Write 3 output JSONL files
```

### Core Functions

| Function | Purpose |
|----------|---------|
| `load_teleqna()` | Load both splits from HuggingFace, return list of items with split tag |
| `load_telequad()` | Parse local JSON, flatten to (context, question, answer_text, qa_id) list |
| `call_openrouter(model, prompt)` | API call via OpenAI SDK with retry (base_url=openrouter) |
| `extract_boxed_answer(text)` | Parse `/box{...}` from model output |
| `evaluate_teleqna_item(item, model)` | Single TeleQnA: call model, extract answer, direct match |
| `evaluate_telequad_item(item, model)` | Single TeleQuAD: call model, extract answer, DeepSeek judge |
| `judge_with_deepseek(question, ref, response)` | Semantic match via DeepSeek API |
| `process_model(model, datasets)` | Process all items for one model across both datasets |
| `main()` | Load data, iterate models, write outputs |

### Key Implementation Details

1. **OpenRouter client** (OpenAI SDK):
   ```python
   openrouter_client = OpenAI(
       base_url="https://openrouter.ai/api/v1",
       api_key=os.getenv("OPENROUTER_API_KEY"),
   )
   ```

2. **DeepSeek client** (for TeleQuAD evaluation):
   ```python
   deepseek_client = OpenAI(
       base_url="https://api.deepseek.com",
       api_key=os.getenv("DEEPSEEK_API_KEY"),
   )
   ```

3. **Answer extraction** — regex for `/box{...}`:
   ```python
   def extract_boxed_answer(text: str) -> str | None:
       match = re.search(r'/box\{([^}]*)\}', text)
       return match.group(1).strip() if match else None
   ```

4. **TeleQnA evaluation** — direct matching:
   - Extract answer from `/box{}`
   - Try matching as integer index against `answer` field
   - Also try matching as text against `choices[answer]`
   - Normalize: strip whitespace, case-insensitive

5. **TeleQuAD evaluation** — DeepSeek judge:
   - Send reference answer + model response to DeepSeek
   - Prompt: "Is the model's answer semantically equivalent to the reference? Reply 1 or 0."
   - `temperature=0.0`, `max_tokens=10`

6. **Model availability check** — single test call per model before evaluation; catch 404/403 to skip

7. **Parallelism** — `ThreadPoolExecutor(max_workers=32)` for item-level API calls within each model

8. **Incremental saves** — write results after each model completes (append mode) so progress isn't lost

### Output Record Construction

```python
# TeleQnA record
{
    "query": item["question"],           # or formatted question+choices
    "answer": item["choices"][item["answer"]],  # ground truth choice text
    "model": model_name,
    "dataset": "teleqna",
    "subset": item["subject"],           # e.g. "Research publications"
    "correct": 1.0 or 0.0,
    "query_id": f"teleqna_{split}_{idx}"  # e.g. "teleqna_train_0"
}

# TeleQuAD record
{
    "query": question_text,
    "answer": ref_answer_text,
    "model": model_name,
    "dataset": "telequad",
    "subset": "telequad",
    "correct": 1.0 or 0.0,
    "query_id": f"telequad_{idx}"
}
```

## Key Files to Reference

| File | Purpose |
|------|---------|
| `extract_openllm_leaderboard_data.py` | Pattern for OpenAI SDK + ThreadPoolExecutor + DeepSeek evaluation |
| `data/model_data/models_info.json` | Source of 16 candidate model names |
| `data/model_data/extracted_dataset_samples.jsonl` | Target output format |
| `data/TeleQuAD-v4-full.json` | TeleQuAD dataset source |

## Verification

1. Run with 1 model, 10 items from each dataset — verify output format
2. Check `/box{}` extraction handles edge cases (nested braces, whitespace)
3. Verify TeleQnA direct matching works for both index and text answers
4. Spot-check 5-10 DeepSeek judgments for TeleQuAD
5. Confirm 32-thread parallelism doesn't hit rate limits
6. Verify test split is correctly separated into its own file
7. Run: `conda activate noesis && python evaluate_openrouter_models.py`
