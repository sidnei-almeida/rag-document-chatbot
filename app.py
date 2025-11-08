# Entrypoint para Hugging Face Spaces
# Este arquivo é usado pelo Dockerfile para iniciar a aplicação
from main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

