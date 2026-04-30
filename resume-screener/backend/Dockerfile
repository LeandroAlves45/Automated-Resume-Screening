FROM python:3.11-slim

WORKDIR /app

# Set Python path to include /app
ENV PYTHONPATH=/app

# Instala dependências de sistema necessárias para psycopg2, spaCy e python-magic
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copia o requirements.txt e instala as dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descarrega o modelo spaCy usado pelo pipeline NLP
RUN python -m spacy download en_core_web_sm

# Copia o código da aplicação para o container
COPY . .

# Expõe a porta usada pela aplicação
EXPOSE 8000

# Comando de arranque da aplicação
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
