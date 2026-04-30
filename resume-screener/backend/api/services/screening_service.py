# screening_service.py

from matplotlib import category

from backend.api.db.database import SessionLocal
from backend.api.db.models import Result, Candidate, Process
from backend.scoring.scorer import ResumeScorer
from backend.parser.jd_parser import JobDescriptionParser
from backend.api.utils.errors import ScreeningNotAllowedInStateError, NoCandidatesError


class ScreeningService:
  """Orquestra o screening: parsing JD, extração features, scoring."""

  def __init__(self, db_session):
    self.db = db_session
    self.scorer = ResumeScorer()
    self.jd_parser = JobDescriptionParser()

  def run_screening(self, process_id: str, candidate_ids: list[str]):
    """Executa screening para múltiplos candidatos."""
    process = self.db.query(Process).filter(Process.id == process_id).first()

    if not process:
      raise ValueError(f"Process with ID {process_id} not found")
    
    # Extrai as skills, experiência e educação da JD
    jd_criteria = self.jd_parser.parse(process.jd_text)

    # Para cada candidato, score e persiste resultado
    for candidate_id in candidate_ids:
      candidate = self.db.query(Candidate).filter(
        Candidate.id == candidate_id).first()
      
      if not candidate:
        continue  # ou logar erro

      try:
        # Extrai features do CV
        resume_features = self._extract_features(candidate.raw_text)

        # Calcula score
        result_dict = self.scorer.score(resume_features, jd_criteria)

        # Persiste resultado com required_skills para calcular missing_skills
        result_db = Result(
          candidate_id=candidate.id,
          total_score=result_dict["total_score"],
          breakdown=result_dict["breakdown"],
          category=result_dict["category"],
          matched_skills=result_dict["matched_skills"],
          required_skills=jd_criteria["skills"],
          experience_years_found=result_dict["experience_years_found"]
        )
        self.db.add(result_db)

      except Exception as e:
        # Log do erro sem quebrar o batch
        print(f"Error processing candidate {candidate_id}: {e}")
        continue
    
    self.db.commit()

  def _extract_features(self, resume_text: str, jd_criteria: dict):
    """Extrai features do CV."""
    # Chama o extrator
    from backend.nlp.extractor import ResumeFeatureExtractor
    pass