# Automated Resume Screener

A Python NLP application that automates CV screening by scoring candidates
against a Job Description using weighted multi-criterion analysis.

## How It Works

The system extracts structured features from each CV (skills, experience,
education level) and compares them against the Job Description using:

- **Skill matching** — presence of required technical skills (40%)
- **Experience scoring** — years of experience vs. minimum required (25%)
- **Education scoring** — education level vs. required level (15%)
- **Semantic similarity** — TF-IDF + Cosine Similarity between CV and JD (20%)

Each candidate receives a score from 0–100 and is classified as
**Strong Match**, **Potential Match**, or **Weak Match**.

## Tech Stack

- **Python 3.11+**
- **spaCy** — NLP pipeline (tokenisation, lemmatisation, NER)
- **scikit-learn** — TF-IDF vectorisation and Cosine Similarity
- **pdfminer.six** — PDF text extraction
- **python-docx** — DOCX text extraction

## Project Structure

```text
resume_screener/
├── main.py               # Pipeline orchestrator — entry point
├── config.py             # Weights, thresholds, and global config
├── parser/
│   ├── resume_parser.py  # PDF, DOCX, TXT text extraction
│   └── jd_parser.py      # Job Description feature extraction
├── nlp/
│   ├── preprocessor.py   # Text cleaning and normalisation
│   └── extractor.py      # CV feature extraction
├── scoring/
│   └── scorer.py         # Weighted scoring engine
└── reports/
    └── reporter.py       # CSV, JSON, TXT report generation
```

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/resume-screener.git
cd resume-screener

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy language model
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Basic usage (uses default paths)
python main.py

# Custom paths
python main.py --jd ./jobs/senior_dev.txt --resumes ./candidates --output ./results
```

### Arguments

| Argument    | Default               | Description                      |
| ----------- | --------------------- | -------------------------------- |
| `--jd`      | `job_description.txt` | Path to the Job Description file |
| `--resumes` | `./resumes`           | Folder containing candidate CVs  |
| `--output`  | `./output`            | Destination folder for reports   |

## Output

The screener generates three report formats in the output folder:

- `screening_results_<timestamp>.csv` — ranked candidates with all scores
- `screening_results_<timestamp>.json` — structured data for integrations
- `screening_report_<timestamp>.txt` — narrative report per candidate

### Example Terminal Output

```text
======================================================================
  AUTOMATED RESUME SCREENER — RESULTS
======================================================================
  Total candidates analysed: 3

  Strong Match (>= 75):    1 candidate(s)
  Potential Match (>= 50): 1 candidate(s)
  Weak Match (< 50):       1 candidate(s)

  #    Candidate             Score    Category          Skills
  -------------------------------------------------------------------
  +++ 1  john_smith          82.4     Strong Match      8/10
  +   2  ana_costa           51.3     Potential Match   4/10
  --- 3  pedro_alves         11.7     Weak Match        0/10
```

## Configuration

All scoring weights and thresholds are configurable in `config.py`:

```python
SCORING_WEIGHTS = {
    "skills_match":      0.40,
    "experience_years":  0.25,
    "education":         0.15,
    "keyword_density":   0.20,
}
```

## License

MIT
