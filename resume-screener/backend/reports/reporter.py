"""
Gerador de relatórios.

Formata e exporta resultados em terminal, CSV, JSON e TXT.
Cada ficheiro recebe timestamp para distinguir execuções.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from backend.api.scoring_config import THRESHOLDS

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Gera relatórios a partir da lista de candidatos já pontuada e ordenada.
    """

    def __init__(self, output_dir: str) -> None:
        """
        Inicializa o gerador e garante que a pasta de saída existe.
        """
        self._output_dir = Path(output_dir)

        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Um timestamp partilhado mantém os ficheiros da mesma execução alinhados.
        self._timestamp = datetime.now()
        self._timestamp_str = self._timestamp.strftime("%Y%m%d_%H%M%S")
        self._timestamp_display = self._timestamp.strftime("%Y-%m-%d %H:%M:%S")

        logger.debug("ReportGenerator initialised. Output: %s", self._output_dir)

    def print_terminal(self, results: list[dict]) -> None:
        """
        Imprime no terminal uma tabela resumida de ranking.
        """
        # Contagem por categoria para o cabeçalho.
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

        # Larguras fixas para manter a tabela legível na maioria dos terminais.
        print(f"  {'#':<4} {'Candidate':<25} {'Score':<7} {'Category':<18} {'Skills'}")
        print(f"{divider}")

        for rank, result in enumerate(results, start=1):
            indicator = {
                "Strong Match":     "***",
                "Potential Match":  "*",
                "Weak Match":       "---",
            }.get(result["category"], " ")

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
        Exporta resultados para CSV compatível com folhas de cálculo.
        """
        filename = self._output_dir / f"results_{self._timestamp_str}.csv"

        # utf-8-sig ajuda o Excel no Windows a detetar UTF-8 corretamente.
        with open(filename, "w", newline="", encoding="utf-8-sig") as csvfile:
            writer = csv.writer(csvfile)

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

            for rank, result in enumerate(results, start=1):
                breakdown = result.get("breakdown", {})

                # Semicolons evitam conflito visual com o delimitador CSV.
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
        logger.info("CSV report saved: %s", filename)
        return filename

    def save_json(self, results: list[dict]) -> Path:
        """
        Exporta resultados para JSON com metadados da execução.
        """
        filename = self._output_dir / f"results_{self._timestamp_str}.json"

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
            # O rank é adicionado aqui porque o scorer não conhece a ordenação final.
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

        # ensure_ascii=False preserva nomes com acentos.
        with open(filename, "w", encoding="utf-8") as jsonfile:
            json.dump(output, jsonfile, indent=2, ensure_ascii=False)

        logger.info("JSON report saved: %s", filename)
        return filename

    def save_txt(self, results: list[dict]) -> Path:
        """
        Exporta um relatório TXT detalhado e legível por humanos.
        """
        filename = self._output_dir / f"results_{self._timestamp_str}.txt"

        with open(filename, "w", encoding="utf-8") as txtfile:

            # Cabeçalho do relatório.
            txtfile.write("=" * 70 + "\n")
            txtfile.write("  AUTOMATED RESUME SCREENING - DETAILED REPORT\n")
            txtfile.write("=" * 70 + "\n")
            txtfile.write(f"Generated at: {self._timestamp_display}\n")
            txtfile.write(f"Candidates: {len(results)}\n")
            txtfile.write("-" * 70 + "\n\n")

            # Uma secção por candidato.
            for rank, result in enumerate(results, start=1):
                breakdown = result.get("breakdown", {})

                txtfile.write(f"{'â”€' * 70}\n")
                txtfile.write(
                    f"  #{rank}  {result.get('name', 'Unknown')}\n"
                    f"[{result.get('category', '')}]"
                    f"Score: {result.get('total_score', 0):.1f}/100\n"
                )
                txtfile.write(f"{'â”€' * 70}\n")

                txtfile.write("SCORE BREAKDOWN:\n")
                txtfile.write(
                    f"  {'Skills Match':<22}: {breakdown.get('skills_match', 0):.1f}% "
                    "(weights: 40%)\n"
                )
                txtfile.write(
                    f"  {'Experience':<22}: {breakdown.get('experience_years', 0):.1f}% "
                    "(weights: 25%)\n"
                )
                txtfile.write(
                    f"  {'Education':<22}: {breakdown.get('education', 0):.1f}% "
                    "(weights: 15%)\n"
                )
                txtfile.write(
                    f"  {'Keyword Density':<22}: {breakdown.get('keyword_density', 0):.1f}% "
                    "(weights: 20%)\n"
                )

                txtfile.write("\n  SKILLS\n")
                matched = result.get("matched_skills", [])
                missing = result.get("missing_skills", [])
                txtfile.write(f"  Matched : {', '.join(matched) if matched else 'None'}\n")
                txtfile.write(f"  Missing : {', '.join(missing) if missing else 'None'}\n")

                txtfile.write("\n  EXPERIENCE\n")
                txtfile.write(
                    f"  Years detected in CV: "
                    f"{result.get('experience_years_found', 0)}\n"
                )

                txtfile.write("\n  RECOMMENDATION\n")
                txtfile.write(f"  {self._generate_recommendation(result)}\n")
                txtfile.write("\n")

            # Rodapé do relatório.
            txtfile.write("=" * 70 + "\n")
            txtfile.write("END OF REPORT\n")
            txtfile.write("=" * 70 + "\n")

        logger.info("TXT report saved: %s", filename)
        return filename

    def _generate_recommendation(self, result: dict) -> str:
        """
        Gera uma recomendação textual curta para o candidato.
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
