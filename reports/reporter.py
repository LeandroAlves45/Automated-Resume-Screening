# =============================================================================
# reporter.py — Report Generator
# =============================================================================
# Responsibility: Format and export scoring results in four output formats:
#   1. Terminal  — ranked table printed to stdout during execution
#   2. CSV       — spreadsheet-compatible export for further analysis
#   3. JSON      — structured export for integration with other tools
#   4. TXT       — human-readable narrative report for archiving
#
# Design principle: Open/Closed Principle (OCP).
# Each output format is an independent public method. Adding a new format
# (e.g. HTML, XLSX) requires adding one new method — nothing existing changes.
#
# All four formats include a timestamp (NFR-08: Traceability) so multiple
# runs can be distinguished even if stored in the same output folder.
#
# Dependencies: only Python standard library (csv, json, datetime, pathlib).
# No third-party imports — the reporter must never be a source of failures.
# =============================================================================

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from config import THRESHOLDS

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Formats and exports ranked candidate results in multiple output formats.

    Expects a list of result dicts produced by ResumeScorer.score(),
    each augmented with the candidate's name from the parser output.

    Expected input structure per candidate:
        {
            "name":                   str,
            "total_score":            float,
            "category":               str,
            "breakdown":              dict,
            "matched_skills":         list[str],
            "missing_skills":         list[str],
            "experience_years_found": int,
        }
    """

    def __init__(self, output_dir: str) -> None:
        """
        Initialise the report generator with an output directory.

        Creates the output directory if it does not exist, so the caller
        never needs to handle directory creation separately.

        Args:
            output_dir (str): Path to the folder where reports will be saved.
        """
        self._output_dir = Path(output_dir)

        # Create the output directory (and any missing parents) if needed.
        # exist_ok=True means no error is raised if the folder already exists.
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Generate a single timestamp for this run, shared across all formats.
        # Using one timestamp ensures that all four output files from the same
        # run have exactly the same timestamp — important for NFR-08 traceability.
        self._timestamp = datetime.now()
        self._timestamp_str = self._timestamp.strftime("%Y%m%d_%H%M%S")
        self._timestamp_display = self._timestamp.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug(f"ReportGenerator initialised. Output: {self._output_dir}")

    # -----------------------------------------------------------------------------
    # PUBLIC METHODS - one per output format
    # -----------------------------------------------------------------------------

    def print_terminal(self, results: list[dict]) -> None:
        """
        Print a formatted ranking table to the terminal (stdout).

        This is always the first output shown — it provides immediate
        feedback during a batch run without needing to open a file.
        Candidates are shown ranked by score (highest first).

        Uses only print() here intentionally — terminal output IS the
        purpose of this method, unlike logging which is for diagnostics.

        Args:
            results (list[dict]): Scored and ranked candidate results.
                                Must be pre-sorted by total_score descending.
        """
        # Count candidates in each category for the summary header
        strong = sum(1 for r in results if r["category"] == "Strong Match")
        potential = sum(1 for r in results if r["category"] == "Potential Match")
        weak = sum(1 for r in results if r["category"] == "Weak Match")

        separator = "=" * 70
        divider = "-" * 70

        print(f"\n{separator}")
        print("  AUTOMATED RESUME SCREENING - RESULTS")
        print(separator)
        print(f"  Total candidates analised :  {len(results)}")
        print(f"   Date / Time              :  {self._timestamp_display}")
        print(separator)
        print(f"  Strong Match  (>= {THRESHOLDS['strong_match']}) :  {strong} candidate(s)")
        print(f"  Potential Match (>= {THRESHOLDS['potential_match']}) :  {potential} candidate(s)")
        print(f"  Weak Match    (< {THRESHOLDS['potential_match']}) :  {weak} candidate(s)")
        print(f"{divider}")

        # Column header — fixed-width formatting with f-strings.
        # The column widths (25, 7, 18, 10) are chosen to fit typical
        # candidate names and score values without wrapping on most terminals.
        print(f"  {'#':<4} {'Candidate':<25} {'Score':<7} {'Category':<18} {'Skills'}")
        print(f"{divider}")

        for rank, result in enumerate(results, start=1):
            indicator = {
                "Strong Match":     "***",
                "Potential Match":  "*",
                "Weak Match":       "---",
            }.get(result["category"], " ")

            # Skills matched display: "5/8" format.
            matched_count = len(result.get("matched_skills", []))
            missing_count = len(result.get("missing_skills", []))
            total_skills = matched_count + missing_count
            skills_display = f"{matched_count}/{total_skills}" if total_skills > 0 else "N/A"

            print(
                f"  {indicator} {rank:<3}"
                f"{result['name']:<25}"
                f"{result['total_score']:>6.1f}  "
                f"{result['category']:<18}  "
                f"{skills_display}"
            )
        print(f"{divider}\n")

    def save_csv(self, results: list[dict]) -> Path:
        """
        Export results to a CSV file for spreadsheet analysis.

        Each row represents one candidate with all scoring fields.
        The header row uses human-readable column names.
        The file is UTF-8 encoded with a BOM so Excel opens it correctly
        on Windows without character encoding issues.

        Args:
            results (list[dict]): Scored and ranked candidate results.

        Returns:
            Path: Absolute path to the generated CSV file.
        """
        # Build a timestamped filename so multiple runs don't overwrite each other.
        filename = self._output_dir / f"results_{self._timestamp_str}.csv"

        # utf-8-sig writes a UTF-8 BOM (Byte Order Mark) at the start of the file.
        # This is invisible to most tools but tells Microsoft Excel to open the
        # file as UTF-8 rather than the Windows default encoding (CP1252),
        # which would corrupt special characters in candidate names.
        with open(filename, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile)

            # Write the header row with human-readable column names.
            writer.writerow([
                "Rank",
                "Candidate",
                "Total Score",
                "Category",
                "Skills Match (%)",
                "Experience (%)",
                "Education (%)",
                "Keyword Density (%)",
                "Matched Skills",
                "Missing Skills",
                "Experience Years Found",
            ])

            # Write one row per candidate
            for rank, result in enumerate(results, start=1):
                breakdown = result.get("breakdown", {})

                # Skills are joined into a single semicolon-separated string.
                # We use semicolons (not commas) because the outer delimiter
                # is already a comma — using commas inside a field would require
                # quoting and could confuse some CSV parsers.
                matched_str = "; ".join(result.get("matched_skills", []))
                missing_str = "; ".join(result.get("missing_skills", []))

                writer.writerow([
                    rank,
                    result.get("name", ""),
                    result.get("total_score", 0),
                    result.get("category", ""),
                    breakdown.get("skills_match", 0),
                    breakdown.get("experience_years", 0),
                    breakdown.get("education", 0),
                    breakdown.get("keyword_density", 0),
                    matched_str,
                    missing_str,
                    result.get("experience_years_found", 0),
                ])
        logger.info(f"CSV report saved: {filename}")
        return filename

    def save_json(self, results: list[dict]) -> Path:
        """
        Export results to a JSON file for programmatic consumption.

        The JSON structure is a dict with metadata at the top level
        and a 'candidates' array containing the full result for each.
        This structure is more useful than a bare array because it
        includes the run timestamp and summary counts alongside the data.

        Args:
            results (list[dict]): Scored and ranked candidate results.

        Returns:
            Path: Absolute path to the generated JSON file.
        """
        filename = self._output_dir / f"results_{self._timestamp_str}.json"

        # Build a top-level wrapper object with metadata.
        # A bare list would also be valid JSON, but this structure is
        # self-documenting — anyone reading the file knows when it was
        # generated and how many candidates it contains
        output = {
            "metadata" : {
                "generated_at":   self._timestamp_display,
                "total_candidates": len(results),
                "strong_matches": sum(1 for r in results if r["category"] == "Strong Match"),
                "potential_matches": sum(1 for r in results if r["category"] == "Potential Match"),
                "weak_matches": sum(1 for r in results if r["category"] == "Weak Match"),
            },
            "candidates": []
        }

        for rank, result in enumerate(results, start=1):
            # Build a clean candidate entry.
            # We add 'rank' explicitly since it is not part of the scorer output
            # but is important context for anyone consuming this JSON.
            candidate_entry = {
                "rank": rank,
                "name": result.get("name", ""),
                "total_score": result.get("total_score", 0),
                "category": result.get("category", ""),
                "breakdown": result.get("breakdown", {}),
                "matched_skills": result.get("matched_skills", []),
                "missing_skills": result.get("missing_skills", []),
                "experience_years_found": result.get("experience_years_found", 0),
            }
            output["candidates"].append(candidate_entry)

        # indent=2 produces human-readable JSON with 2-space indentation.
        # ensure_ascii=False preserves non-ASCII characters (accented names, etc.)
        # rather than escaping them as \uXXXX sequences.
        with open(filename, "w", encoding="utf-8") as jsonfile:
            json.dump(output, jsonfile, indent=2, ensure_ascii=False)

        logger.info(f"JSON report saved: {filename}")
        return filename

    def save_text(self, results: list[dict]) -> Path:
        """
        Export a human-readable narrative TXT report for archiving.

        Unlike the CSV (data-focused) and JSON (machine-focused), the TXT
        report is written for a human reader — a hiring manager or recruiter
        who wants a detailed written summary of each candidate's profile.

        Each candidate gets a dedicated section with their score breakdown,
        matched and missing skills, and a written recommendation.

        Args:
            results (list[dict]): Scored and ranked candidate results.

        Returns:
            Path: Absolute path to the generated TXT file.
        """
        filename = self._output_dir / f"results_{self._timestamp_str}.txt"

        with open(filename, "w", encoding="utf-8") as txtfile:

            # -- Report header --
            txtfile.write("=" * 70 + "\n")
            txtfile.write("  AUTOMATED RESUME SCREENING - DETAILED REPORT\n")
            txtfile.write("=" * 70 + "\n")
            txtfile.write(f"Generated at: {self._timestamp_display}\n")
            txtfile.write(f"Candidates: {len(results)}\n")
            txtfile.write("-" * 70 + "\n\n")

            # -- One section per candidate --
            for rank, result in enumerate(results, start=1):
                breakdown = result.get("breakdown", {})

                txtfile.write(f"{'─' * 70}\n")
                txtfile.write(
                    f"  #{rank}  {result.get('name', 'Unknown')}\n"
                    f"[{result.get('category', '')}]"
                    f"Score: {result.get('total_score', 0):.1f}/100\n"
                )
                txtfile.write(f"{'─' * 70}\n")

                # Score breakdown
                txtfile.write("SCORE BREAKDOWN:\n")
                txtfile.write(f"  {'Skills Match':<22}: {breakdown.get('skills_match', 0):.1f}% (weights: 40%)\n")
                txtfile.write(f"  {'Experience':<22}: {breakdown.get('experience_years', 0):.1f}% (weights: 25%)\n")
                txtfile.write(f"  {'Education':<22}: {breakdown.get('education', 0):.1f}% (weights: 15%)\n")
                txtfile.write(f"  {'Keyword Density':<22}: {breakdown.get('keyword_density', 0):.1f}% (weights: 20%)\n")

                # Skills detail section.
                txtfile.write("\n  SKILLS\n")
                matched = result.get("matched_skills", [])
                missing = result.get("missing_skills", [])
                txtfile.write(f"  Matched : {', '.join(matched) if matched else 'None'}\n")
                txtfile.write(f"  Missing : {', '.join(missing) if missing else 'None'}\n")

                # Experience line
                txtfile.write(f"\n  EXPERIENCE\n")
                txtfile.write(
                    f"  Years detected in CV: "
                    f"{result.get('experience_years_found', 0)}\n"
                )

                # Recommendation line based on category
                txtfile.write(f"\n  RECOMMENDATION\n")
                txtfile.write(f"  {self._generate_recommendation(result)}\n")
                txtfile.write("\n")

            # -- Report footer --
            txtfile.write("=" * 70 + "\n")
            txtfile.write("END OF REPORT\n")
            txtfile.write("=" * 70 + "\n")

        logger.info(f"TXT report saved: {filename}")
        return filename

    # -----------------------------------------------------------------------------
    # PRIVATE HELPER METHODS
    # -----------------------------------------------------------------------------

    def _generate_recommendation(self, result: dict) -> str:
        """
        Generate a plain-language recommendation string for a candidate.

        Written recommendations make the TXT report useful to hiring managers
        who may not want to interpret numeric scores directly. The text is
        templated but includes dynamic values (score, missing skills) to
        make it feel specific rather than generic.

        Args:
            result (dict): A single scored candidate result dict.

        Returns:
            str: A one-sentence recommendation string.
        """
        category = result.get("category", "")
        score = result.get("total_score", 0)
        missing = result.get("missing_skills", [])
        name = result.get("name", "This candidate")

        if category == "Strong Match":
            return (
                f"{name} is a strong match with a score of {score:.1f}/100. "
                "Recommended for immediate interview."
            )
        elif category == "Potential Match":
            if missing:
                missing_str = ", ".join(missing[:3]) 
                return (
                    f"{name} is a potential match with a score of {score:.1f}/100. "
                    f"Key missing skills: {missing_str}. Consider for a second review."
                )
            else:
                return (
                    f"{name} is a potential match with a score of {score:.1f}/100. "
                    "No significant skill gaps identified. Consider for a second review."
                )
        else:
            return (
                    f"{name} does not meet the minimum requirements for this role. "
                    f"(score: {score:.1f}/100). Not recommended for advancement."
                )

