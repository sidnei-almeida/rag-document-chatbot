FROM python:3.11-slim

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Criar usuário não-root para segurança
RUN useradd -m -u 1000 user

WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY --chown=user:user . .

# Mudar para usuário não-root
USER user

# Hugging Face Spaces (Docker SDK) usa porta 7860 por padrão; PORT pode ser injetado
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]

