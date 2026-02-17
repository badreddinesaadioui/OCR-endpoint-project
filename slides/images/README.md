# Images for soutenance slides (OCR & Parsing)

## OCR

| File | Description |
|------|-------------|
| `01-cvtheque-database.png` | Our CVtheque – database of 16 CVs, navigation, CV cards |
| `02-ocr-benchmark-ui.png` | OCR Benchmark – model selection, input, ground truth, Analyze |
| `03-mistral-api-keys.png` | Mistral API keys page (key-mistral) |
| `04-logo-mistral.png` | Mistral AI logo |
| `05-replicate-ocr-page.png` | Replicate OCR page (text-extract-ocr, marker, deepseek-ocr) |
| `06-logo-replicate.png` | Replicate logo |
| `08-parallel-ocr-results-verdict.png` | Parallel OCR test: summary table + verdict (Mistral vs Replicate) |
| `09-ocr-charts-cer-time.png` | Bar charts: CER by layout, time by language, avg time |
| `16-ocr-charts-cer-layout-language.png` | OCR: CER by layout, CER by language, avg CER by model, layout accuracy |
| `17-ocr-charts-time-cost.png` | OCR: avg time per run, total cost per model |

## LLM Parsing

| File | Description |
|------|-------------|
| `11-llm-parsing-benchmark-ui.png` | LLM Parsing Benchmark – models (GPT 4.1 nano, GPT 5 mini, Gemini 2.5 Flash, Claude 4.5 Haiku), input/ground truth JSON |
| `12-parallel-llm-parsing-results-verdict.png` | Parallel LLM test: summary (Claude 4.5 Haiku, Gemini 2.5 Flash, Gemini 4o nano) + verdict |
| `13-llm-parsing-charts-accuracy-time.png` | LLM: accuracy by layout, time by language, avg time per run |
| `14-llm-parsing-all-runs-table.png` | LLM parsing: “All runs” table (cv_filename, model, accuracy, time, cost, json_valid, schema_valid) |

## Summary & deliverable

| File | Description |
|------|-------------|
| `15-results-summary-ocr-llm.png` | Results summary – recommended OCR (Mistral OCR 3) and LLM (Claude 4.5 Haiku) with metrics |
| `18-cv-parser-svelte-app-ui.png` | CV Parser Svelte app (forvis mazars) – upload CV, Parse CV, result area |

## Placeholders (optional to replace)

| File | Description |
|------|-------------|
| `07-placeholder.png` | Small/placeholder |
| `10-placeholder-black.png` | Black/empty – e.g. OCR “All runs” table if you want a dedicated screenshot |

---

**Sourcing for slides/report:**  
- **OCR:** Mistral OCR 3 (Mistral API), Replicate text-extract-ocr (Replicate). GPT-4o Vision (OpenAI) and others were benchmarked then excluded (e.g. speed &gt; 20 s).  
- **LLM parsing:** GPT 4.1 nano, GPT 5 mini (OpenAI API); Gemini 2.5 Flash, Claude 4.5 Haiku (Replicate). GPT 5 mini excluded (&gt; 15 s). **Chosen:** Claude 4.5 Haiku (best accuracy, valid JSON/schema, reasonable cost).
