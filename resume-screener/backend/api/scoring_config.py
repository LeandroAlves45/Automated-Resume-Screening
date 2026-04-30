# ==================================================================
# config.py - Módulo central de configuração
# ==================================================================
# Este ficheiro concentra os valores configuráveis do sistema.
# Os restantes módulos devem importar estes valores em vez de duplicar
# constantes ou regras diretamente no código.

# ------------------------------------------------------------------
# PESOS DE PONTUAÇÃO
# ------------------------------------------------------------------

# Define a importância relativa de cada métrica na pontuação final.
# A soma dos pesos deve ser 1.0 para manter o cálculo equilibrado.

SCORING_WEIGHTS: dict[str, float] = {
    "skills_match": 0.4,        # Percentagem de competências da vaga encontradas no CV
    "experience_years": 0.25,   # Anos totais de experiência relevante
    "keyword_density": 0.2,     # Similaridade TF-IDF entre palavras-chave do CV e da vaga
    "education": 0.15           # Nível de formação do candidato face ao requisito
}

# ------------------------------------------------------------------
# NÍVEIS DE FORMAÇÃO
# ------------------------------------------------------------------
# Mapeia palavras associadas a formação académica para valores normalizados
# entre 0.0 e 1.0.
#
# Hierarquia usada na comparação:
#  0.2 = ensino secundário
#  0.4 = curso intermédio ou licenciatura incompleta
#  0.6 = licenciatura
#  0.8 = mestrado
#  1.0 = doutoramento
#
# As palavras estão em minúsculas; o texto analisado deve passar por .lower().
# -------------------------------------------------------------------
EDUCATION_LEVELS: dict[str, float] = {
    # Equivalentes ao ensino secundário
    "high school": 0.2,
    "secondary school": 0.2,
    "gcse": 0.2,
    "a-level": 0.2,

    # Nível intermédio
    "associate": 0.4,
    "foundation degree": 0.4,
    "hnd": 0.4,
    "hnc": 0.4,

    # Licenciatura
    "bachelor": 0.6,
    "bsc": 0.6,
    "ba": 0.6,
    "b.sc": 0.6,
    "b.a": 0.6,
    "undergraduate": 0.6,
    "licenciatura": 0.6,

    # Mestrado
    "master": 0.8,
    "msc": 0.8,
    "mba": 0.8,
    "m.sc": 0.8,
    "postgraduate": 0.8,
    "mestrado": 0.8,

    # Doutoramento
    "phd": 1.0,
    "doctorate": 1.0,
    "ph.d.": 1.0,
    "doctoral": 1.0,
    "doutoramento": 1.0
}


# ------------------------------------------------------------------
# LIMIARES DE CLASSIFICAÇÃO
# ------------------------------------------------------------------
# Define os limites que determinam a categoria final do candidato.
# As pontuações ficam no intervalo [0, 100].
#
# Categorias:
#   - "Strong Match": >= 75: perfil muito alinhado
#   - "Potential Match": >= 50: perfil parcial, requer revisão manual
#   - "Weak Match": < 50: perfil pouco alinhado
THRESHOLDS: dict[str, int] = {
    "strong_match": 75,
    "potential_match": 50,
    # Qualquer valor abaixo de potential_match é classificado como Weak Match.
}

# ------------------------------------------------------------------
# CONFIGURAÇÃO DE TEXTO
# ------------------------------------------------------------------
# Parâmetros comuns para validação, extração e processamento NLP.
# -------------------------------------------------------------------
TEXT_CONFIG: dict = {
    "min_resume_length": 100,  # Caracteres mínimos para considerar um CV válido
    "max_experience_years": 50,  # Limite de segurança para anos de experiência
    "spacy_model": "en_core_web_sm"  # Modelo spaCy usado nas tarefas de NLP
}

# -------------------------------------------------------------------
# EXTENSÕES SUPORTADAS
# -------------------------------------------------------------------
# Extensões que o parser de CVs tenta processar.
# --------------------------------------------------------------------
SUPPORTED_EXTENSIONS: list[str] = [".pdf", ".docx", ".txt"]
