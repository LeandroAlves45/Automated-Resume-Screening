# Automated Resume Screener — Technical Reference

**Version**: 1.0.0 (v1.0 Pipeline)
**Status**: Complete — 63/63 tests passing
**Last Updated**: April 2026
**Author**: Leandro Alves

> This document is the internal technical reference for the v1.0 pipeline codebase.
> It describes every module, class, and function: its responsibility, design decisions,
> inputs, outputs, and side effects.
>
> For system-level architecture, deployment, and API design, see `ARCHITECTURE.md`.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Pipeline Execution Order](#2-pipeline-execution-order)
3. [config.py](#3-configpy)
4. [main.py](#4-mainpy)
5. [parser/resume_parser.py](#5-parserresume_parserpy)
6. [parser/jd_parser.py](#6-parserjd_parserpy)
7. [nlp/preprocessor.py](#7-nlppreprocessorpy)
8. [nlp/extractor.py](#8-nlpextractorpy)
9. [scoring/scorer.py](#9-scoringscorerpy)
10. [reports/reporter.py](#10-reportsreporterpy)
11. [tests/conftest.py](#11-testsconftestpy)
12. [Unit Tests Reference](#12-unit-tests-reference)
13. [Integration Tests Reference](#13-integration-tests-reference)

---

## 1. Project Structure

```
backend/
  config.py                   # Central configuration constants
  main.py                     # CLI entry point and pipeline orchestrator
  nlp/
    __init__.py
    preprocessor.py           # Raw text cleaning and tokenisation
    extractor.py              # Feature extraction (skills, exp, edu, NER)
  parser/
    __init__.py
    resume_parser.py          # PDF / DOCX / TXT file parsing
    jd_parser.py              # Job Description feature extraction
  scoring/
    __init__.py
    scorer.py                 # Weighted scoring engine
  reports/
    __init__.py
    reporter.py               # Terminal, CSV, JSON, TXT output
  tests/
    conftest.py               # Shared pytest fixtures (session + function scope)
    unit/
      test_resume_parser.py   # UT-01 to UT-03 + additional
      test_jd_parser.py       # UT-04 to UT-06 + additional
      test_extractor.py       # UT-07 to UT-09 + additional
      test_scorer.py          # UT-10 to UT-13 + additional
    integration/
      test_pipeline.py        # IT-01 to IT-04 + sort order tests
```

---

## 2. Pipeline Execution Order

```
main.py
  |
  ├─ 1. load_spacy_model()
  │       Loads TEXT_CONFIG["spacy_model"] once.
  │       Result shared via Dependency Injection.
  │
  ├─ 2. JobDescriptionParser(nlp).parse(jd_text)
  │       → { skills, min_experience, education_level, keywords, raw_text }
  │
  ├─ 3. ResumeParser().parse_folder(folder_path)
  │       → list[ { name, file, text, error } ]
  │
  └─ 4. For each valid parse result:
          │
          ├─ TextPreprocessor(nlp).process(raw_text)
          │       → { clean_text, tokens }
          │
          ├─ ResumeFeatureExtractor(nlp).extract(clean_text, jd_skills)
          │       → { matched_skills, experience_years, education_level,
          │           keywords, entities }
          │
          ├─ features["keywords"] = preprocessor tokens  (override)
          │
          ├─ ResumeScorer().score(features, jd_criteria)
          │       → { total_score, breakdown, category,
          │           matched_skills, missing_skills, experience_years_found }
          │
          └─ score_result["name"] = candidate_name  (augment)

  ├─ 5. sort_results(scored_results)
  │       Sort by (-total_score, name.lower())
  │
  └─ 6. ReportGenerator(output_dir)
          .print_terminal(ranked)
          .save_csv(ranked)     → Path
          .save_json(ranked)    → Path
          .save_text(ranked)    → Path
```

---

## 3. config.py

### Responsibility

Single source of truth for all configurable values in the v1.0 pipeline. No module hardcodes values. Everything tunable — scoring weights, education levels, classification thresholds, text limits — is defined here.

### Design Decisions

- Centralises all configuration in one file. Changing a weight requires editing exactly one location.
- Uses typed constants so IDEs enforce correct usage at the call site.
- Comments explain the business meaning of each value, not just its type.

### Constants

#### `SCORING_WEIGHTS: dict[str, float]`

Relative weights for the four scoring criteria. **Must sum to 1.0.**

```python
SCORING_WEIGHTS = {
    "skills_match":     0.40,
    "experience_years": 0.25,
    "keyword_density":  0.20,
    "education":        0.15,
}
```

#### `EDUCATION_LEVELS: dict[str, float]`

Maps education keyword strings (lowercase) to a normalised level `[0.0, 1.0]`.

| Level | Value | Keywords |
|-------|-------|---------|
| High School | 0.2 | high school, secondary school, gcse, a-level |
| Sub-degree | 0.4 | associate, foundation degree, hnd, hnc |
| Bachelor | 0.6 | bachelor, bsc, ba, undergraduate, licenciatura |
| Master | 0.8 | master, msc, mba, postgraduate, mestrado |
| Doctorate | 1.0 | phd, doctorate, ph.d., doctoral, doutoramento |

#### `THRESHOLDS: dict[str, int]`

```python
THRESHOLDS = {
    "strong_match":    75,
    "potential_match": 50,
    # anything below potential_match is Weak Match
}
```

#### `TEXT_CONFIG: dict`

```python
TEXT_CONFIG = {
    "min_resume_length": 100,       # characters — below this, CV is invalid
    "max_experience_years": 50,     # cap on regex-detected experience years
    "spacy_model": "en_core_web_sm" # spaCy model name to load
}
```

#### `SUPPORTED_EXTENSIONS: list[str]`

```python
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]
```

Files with other extensions are silently skipped by `ResumeParser.parse_folder()`.

---

## 4. main.py

### Responsibility

CLI entry point and pipeline orchestrator. Parses arguments, initialises all modules (loading spaCy once), and calls each pipeline step in the correct order. Contains no business logic.

### Design Decisions

- **Thin orchestrator pattern**: `main.py` only knows the call sequence. Implementation details live in the modules.
- **Dependency Injection**: spaCy is loaded once and passed as a constructor argument to every NLP module. This avoids reloading a ~15MB model multiple times.
- Each pipeline step is a standalone function so integration tests can import and call them directly without invoking the CLI.

### `build_arg_parser() -> argparse.ArgumentParser`

| Argument | Default | Description |
|----------|---------|-------------|
| `--jd` | `job_description.txt` | Path to the JD file |
| `--resumes` | `./resumes` | Path to the CV folder |
| `--output` | `./output` | Path to the output folder |

### `load_spacy_model() -> spacy.Language`

Loads the model defined in `TEXT_CONFIG["spacy_model"]`. Called once at startup. If the model is not installed, logs the installation command and calls `sys.exit(1)`.

### `load_job_description(jd_path: str, jd_parser: JobDescriptionParser) -> dict | None`

Reads the JD file from disk with UTF-8/Latin-1 fallback, then calls `jd_parser.parse()`. Returns `None` if the file does not exist.

### `process_resumes(folder_path, jd_criteria, resume_parser, preprocessor, extractor, scorer) -> tuple[list[dict], list[dict]]`

Runs the full extraction and scoring pipeline for all CVs in a folder. Returns `(scored_results, failed_files)`. CVs that fail to parse are collected without aborting the batch.

### `sort_results(results: list[dict]) -> list[dict]`

Sort key: `(-total_score, name.lower())`. Negating the score achieves descending order without `reverse=True`, which would also reverse the alphabetical tie-breaking.

### `report_failures(failed_files: list[dict]) -> None`

Logs each failed file at WARNING level.

### `main() -> None`

Orchestrates the full pipeline. Parses CLI args, loads spaCy, initialises all modules, processes JD, processes all CVs, sorts results, generates all four reports, logs failures.

---

## 5. parser/resume_parser.py

### Responsibility

Extract plain text from CV files in PDF, DOCX, and TXT format. The caller receives a uniform result dict regardless of format or failure.

### Design Decisions

- **Facade Pattern**: pdfminer.six, python-docx, and standard file I/O are hidden behind one interface.
- **SRP**: `parse()` handles routing. Each `_parse_*()` handles one format. `_clean_text()` handles normalisation.
- **Error containment**: exceptions are captured in the result dict, never raised to callers.

### Class: `ResumeParser`

#### `parse(file_path: str) -> dict`

**Returns**:
```python
{
    "name":  str,        # filename without extension
    "file":  str,        # full path as string
    "text":  str,        # extracted plain text (empty on failure)
    "error": str | None  # error message, or None on success
}
```

Steps: existence check → extension check → dispatch to format parser → `_clean_text()` → minimum length validation → exception capture.

#### `parse_folder(folder_path: str) -> list[dict]`

Iterates all files in a folder (non-recursive), skips unsupported extensions silently. Files are sorted before processing to guarantee consistent cross-platform ordering. Returns `[]` if the folder does not exist.

#### `_parse_pdf(path: str) -> str`

Uses pdfminer.six `extract_text()`. Returns empty string if the PDF has no text layer.

#### `_parse_docx(path: str) -> str`

Uses python-docx. Iterates `doc.paragraphs` and joins with `\n`. Known limitation: text inside tables, headers, footers, and text boxes is not captured.

#### `_parse_txt(path: str) -> str`

Reads with UTF-8 (`errors="strict"`). On `UnicodeDecodeError`, retries with Latin-1.

#### `_clean_text(text: str) -> str`

Splits into lines → collapses intra-line whitespace → rejoins → collapses 3+ consecutive newlines to 2 → strips.

---

## 6. parser/jd_parser.py

### Responsibility

Parse a Job Description text and extract structured evaluation criteria for the scoring engine.

### Design Decisions

- **Dependency Injection**: spaCy injected via constructor, not loaded internally.
- `_SKILLS_REFERENCE` list is converted to `_SKILLS_SET` for O(1) membership lookup.

### Class: `JobDescriptionParser`

#### `parse(jd_text: str) -> dict`

**Returns**:
```python
{
    "skills":          list[str],  # skills from reference list found in JD
    "min_experience":  int,        # minimum years required (0 if not stated)
    "education_level": float,      # required education level [0.0, 1.0]
    "keywords":        list[str],  # lemmatised, deduplicated keywords
    "raw_text":        str         # original JD text (used by scorer for TF-IDF)
}
```

#### `_extract_skills(text: str) -> list[str]`

Single-character skills use word boundary regex `\b`. Multi-character skills use `in` check. Text is lowercased once outside the loop: O(n + m).

#### `_extract_min_experience(text: str) -> int`

Applies four regex patterns. Returns the maximum value found. Values above `max_experience_years` are excluded.

#### `_extract_education_level(text: str) -> float`

Scans for keywords from `EDUCATION_LEVELS`. Returns the maximum numeric value found. Returns `0.0` if nothing found.

#### `_extract_keywords(doc: spacy.Doc) -> list[str]`

Filters stop words, punctuation, whitespace, tokens shorter than 2 chars. Lemmatises and lowercases. Deduplicates using `dict.fromkeys()` (preserves insertion order).

---

## 7. nlp/preprocessor.py

### Responsibility

Transform raw CV text into two representations: a cleaned plain-text string (for regex-based extraction) and a list of lemmatised tokens (for TF-IDF vectorisation).

### Class: `TextPreprocessor`

#### `process(raw_text: str) -> dict`

**Returns**:
```python
{
    "clean_text": str,       # normalised plain text
    "tokens":     list[str]  # lemmatised, filtered tokens (with duplicates)
}
```

#### `_clean_text(text: str) -> str`

Applied in order: remove non-printable control characters (except `\n`, `\t`) → normalise Unicode punctuation to ASCII → replace emails with `EMAIL` placeholder → replace URLs with `URL` placeholder → collapse multiple spaces/tabs → collapse 3+ newlines to 2 → strip.

**Why placeholders instead of deletion**: removing emails/URLs completely would collapse surrounding context. Placeholders preserve sentence structure while eliminating noise tokens.

#### `_tokenise(doc: spacy.Doc) -> list[str]`

Filters: stop words, punctuation/whitespace, numeric-only, tokens shorter than 2 chars. After filtering: `token.lemma_.lower().strip()`. Duplicates are retained — TF-IDF needs term frequency.

---

## 8. nlp/extractor.py

### Responsibility

Extract quantifiable features from preprocessed CV text: matched skills, experience years, education level, keywords, NER entities.

### Design Decisions

- Each feature is extracted by its own private method — independent, can fail or be improved separately.
- **Two-strategy experience extraction**: taking the maximum of regex and date-range strategies gives the most candidate-favourable result.
- spaCy `Doc` is created once in `extract()` and shared across methods.

### Class: `ResumeFeatureExtractor`

#### `extract(resume_text: str, jd_skills: list[str]) -> dict`

**Returns**:
```python
{
    "matched_skills":   list[str],  # JD skills found in the CV
    "experience_years": int,        # years detected (max of both strategies)
    "education_level":  float,      # highest education level [0.0, 1.0]
    "keywords":         list[str],  # lemmatised tokens (with duplicates)
    "entities":         dict        # NER entities: ORG, GPE, DATE, PERSON
}
```

#### `_strategy_explicit_phrases(text: str) -> int`

Scans for natural language patterns: "5 years of experience", "3 years experience", "over 3 years", "more than 4 years". Returns maximum found.

#### `_strategy_date_ranges(text: str) -> int`

Normalises "present/current/now" to current year. Pattern: `(?:\w+\s+)?(\d{4})\s*[-–]\s*(?:\w+\s+)?(\d{4})`. Sanity checks: `start >= 1970`, `end <= current_year`, `start < end`. Sums durations across all valid matches.

#### `_extract_education(text: str) -> float`

Scans for keywords from `EDUCATION_LEVELS`. Returns the maximum value found.

#### `_extract_entities(doc: spacy.Doc) -> dict`

spaCy NER. Categories: ORG, GPE, DATE, PERSON. Returns empty lists if using blank model.

---

## 9. scoring/scorer.py

### Responsibility

Calculate a final weighted score `[0, 100]` for each candidate by combining four independent scoring criteria.

### Design Decisions

- **OCP**: new criteria are added by defining a new private method and registering it in `score()`.
- `numpy.clip()` enforces the `[0, 100]` range regardless of floating point rounding.
- TF-IDF operates on lemmatised token lists, not raw text.
- Neutral values are used when a criterion cannot be evaluated — the candidate is not penalised for missing JD information.

### Class: `ResumeScorer`

#### `score(resume_features: dict, jd_criteria: dict) -> dict`

Steps: calculate four criterion scores → apply weights → scale to 100 → clip and round → build breakdown → classify → compute missing_skills.

#### `_score_skills(matched: list, required: list) -> float`

`score = len(matched) / len(required)`. Neutral: `required` empty → returns `0.5`.

#### `_score_experience(candidate_years: int, min_required: int) -> float`

`score = min(candidate_years / (2 × min_required), 1.0)`. Someone with 2× the required experience gets full credit. Neutral: `min_required == 0` → returns `0.5`.

#### `_score_education(candidate_level: float, required_level: float) -> float`

`score = min(candidate_level / required_level, 1.0)`. Both `0.0` → returns `0.6` (neutral positive). `required_level == 0`, candidate has edu → bonus, no penalty. `candidate_level == 0`, JD requires edu → `0.0`.

#### `_score_keywords(resume_keywords: list[str], jd_keywords: list[str]) -> float`

TF-IDF + Cosine Similarity. Joins both token lists into space-separated strings → `TfidfVectorizer().fit_transform([cv_text, jd_text])` → `cosine_similarity(matrix[0], matrix[1])[0][0]`. Returns `0.0` if either list is empty or on any exception.

#### `_classify(score: float) -> str`

Returns `"Strong Match"`, `"Potential Match"`, or `"Weak Match"` based on `THRESHOLDS` from config.py.

---

## 10. reports/reporter.py

### Responsibility

Format and export scoring results in four output formats: terminal table, CSV, JSON, TXT narrative.

### Design Decisions

- **OCP**: adding a new format requires one new method. Nothing existing changes.
- A single timestamp generated at `__init__` time is shared across all four output files.
- CSV uses `utf-8-sig` (UTF-8 with BOM) so Microsoft Excel opens it correctly on Windows.
- JSON uses `ensure_ascii=False` to preserve accented characters in candidate names.

### Class: `ReportGenerator`

#### `__init__(output_dir: str) -> None`

Creates the output directory. Generates `_timestamp_str` (for filenames) and `_timestamp_display` (for headers), both from the same `datetime.now()` call.

#### `print_terminal(results: list[dict]) -> None`

Header shows: total candidates, timestamp, count per category. Each row: rank indicator (`***`, `*`, `---`), rank number, name, score, category, skills matched/total.

#### `save_csv(results: list[dict]) -> Path`

Columns: Rank, Candidate, Total Score, Category, Skills Match (%), Experience (%), Education (%), Keyword Density (%), Matched Skills, Missing Skills, Experience Years Found. Skills lists serialised as semicolon-separated strings.

#### `save_json(results: list[dict]) -> Path`

Top-level wrapper with `metadata` (generated_at, total, strong_matches, potential_matches, weak_matches) and `candidates` array. `indent=2`, `ensure_ascii=False`.

#### `save_text(results: list[dict]) -> Path`

Human-readable narrative for hiring managers. Each candidate gets a section with score breakdown, matched/missing skills, experience, and a written recommendation from `_generate_recommendation()`.

#### `_generate_recommendation(result: dict) -> str`

Strong Match → "Recommended for immediate interview." Potential Match → lists up to 3 missing skills, recommends second review. Weak Match → "Does not meet minimum requirements."

---

## 11. tests/conftest.py

### Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `nlp_model` | `session` | `spacy.blank("en")`. Created once per test session. Supports tokenisation, not lemmatisation or NER. |
| `sample_jd_text` | `function` | Realistic JD with 6 skills, 5 years experience, Bachelor education. |
| `strong_cv_text` | `function` | CV designed as Strong Match: all 6 skills, 7 years exp, MSc. |
| `weak_cv_text` | `function` | CV designed as Weak Match: no skills, High School, no tech experience. |
| `short_cv_text` | `function` | CV shorter than `min_resume_length`. Tests minimum length validation. |
| `tmp_cv_folder` | `function` | Temp folder with 2 valid `.txt` CVs and 1 unsupported `.xyz` file. |
| `tmp_jd_file` | `function` | Temp JD text file from `sample_jd_text`. |
| `sample_jd_criteria` | `function` | Pre-built JD criteria dict. Bypasses parser in scorer tests. |
| `strong_candidate_features` | `function` | Feature dict: all 6 skills, 8 years, 0.8 edu. |
| `weak_candidate_features` | `function` | Feature dict: no skills, 0 years, 0.2 edu. |

---

## 12. Unit Tests Reference

### test_resume_parser.py — TestParseSingleFile

| Test | ID | What it verifies |
|------|-----|-----------------|
| `test_ut01_valid_txt_file_returns_text` | UT-01 | Valid TXT file → `error` is None, `text` length ≥ 100. |
| `test_ut02_nonexistent_file_returns_error` | UT-02 | Missing file → `error` not None, `text` empty. No exception raised. |
| `test_ut03_cv_too_short_returns_error` | UT-03 | File shorter than 100 chars → `error` mentions "too short". |
| `test_unsupported_extension_returns_error` | — | `.xyz` extension → `error` mentions "unsupported". |
| `test_result_dict_always_has_all_keys` | — | All four keys always present, even on failure. |
| `test_txt_encoding_fallback_latin1` | — | Latin-1 file parsed without error. |

### test_resume_parser.py — TestParseFolder

| Test | What it verifies |
|------|-----------------|
| `test_folder_returns_only_supported_files` | Only `.txt` and `.pdf` processed; `.xyz` skipped. Results count = 2. |
| `test_folder_nonexistent_returns_empty_list` | Non-existent folder → empty list, no exception. |
| `test_folder_results_sorted_alphabetically` | Names in results are alphabetically sorted. |
| `test_bad_file_does_not_abort_batch` | One invalid file → 2 results total, 1 success, 1 failure. |

### test_jd_parser.py

| Test | ID | What it verifies |
|------|-----|-----------------|
| `test_ut04_known_skills_are_detected` | UT-04 | "python" and "docker" in JD → both in `skills`. |
| `test_ut05_extracts_explicit_years` | UT-05 | "3+ years" → `min_experience == 3`. |
| `test_ut06_no_experience_requirement_returns_zero` | UT-06 | No mention of years → `min_experience == 0`. |
| `test_skills_are_case_insensitive` | — | "PYTHON" and "POSTGRESQL" → both detected. |
| `test_returns_maximum_when_multiple_values_found` | — | "3 years" and "5 years" in JD → returns 5. |

### test_extractor.py

| Test | ID | What it verifies |
|------|-----|-----------------|
| `test_ut08_explicit_phrase_extracted` | UT-08 | "5 years of experience" → `experience_years == 5`. |
| `test_ut09_date_ranges_summed_correctly` | UT-09 | "2019 - 2023" → `experience_years >= 4`. |
| `test_takes_maximum_of_both_strategies` | — | Date range gives 7, explicit gives 3 → result is 7. |
| `test_present_keyword_uses_current_year` | — | "2020 - Present" → experience ≥ current_year - 2020. |

### test_scorer.py

| Test | ID | What it verifies |
|------|-----|-----------------|
| `test_ut10_total_score_is_within_bounds` | UT-10 | `0.0 <= total_score <= 100.0` always. |
| `test_score_with_perfect_candidate_approaches_100` | — | All skills, PhD, high experience → score ≥ 85.0. |
| `test_breakdown_values_sum_to_approximately_total` | — | Reconstructed weighted sum equals `total_score` within 0.1. |
| `test_ut13_tfidf_with_empty_keywords_returns_zero_no_exception` | UT-13 | Empty keyword lists → `keyword_density == 0.0`, no exception. |

---

## 13. Integration Tests Reference

All integration tests import `process_resumes`, `sort_results`, and `load_job_description` from `main.py` directly. They use real files on disk via `tmp_path`.

| Test | ID | What it verifies |
|------|-----|-----------------|
| `test_it01_strong_candidate_ranks_above_weak` | IT-01 | Full pipeline: strong ranks above weak, alice is first, carol is last. |
| `test_it02_csv_has_header_plus_one_row_per_candidate` | IT-02 | CSV has exactly N + 1 rows. |
| `test_it03_json_is_parseable_and_has_expected_structure` | IT-03 | JSON has `metadata` and `candidates` keys. All required fields present. |
| `test_it04_invalid_file_does_not_block_valid_cvs` | IT-04 | Invalid file in `failed_files`. Valid CV is scored. |
| `test_sort_by_score_descending` | — | Scores: 45, 80, 60 → sorted: 80, 60, 45. |
| `test_sort_alphabetically_on_score_tie` | — | Three candidates with score 70.0 → sorted: alice, mike, zara. |
