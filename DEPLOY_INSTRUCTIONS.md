# Deploy no Hugging Face Spaces

Space existente: **[salmeida/my-rag-chatbot](https://huggingface.co/spaces/salmeida/my-rag-chatbot)**  
URL da API: **https://salmeida-my-rag-chatbot.hf.space**

Este repositório usa **Docker SDK** (não Gradio SDK). O Space executa `uvicorn main:app` na porta **7860**.

---

## Checklist antes do push

| Item | Obrigatório |
|------|-------------|
| `Dockerfile` na raiz | Sim |
| `requirements.txt` | Sim |
| `main.py` + pacote `app/` | Sim |
| `sample_documents/ai-document-intelligence-report.pdf` | Sim (demo `/demo/load-sample`) |
| Secret **`GROQ_API_KEY`** no Space | Sim |
| `faiss_index/` no Git | Não (opcional; use upload ou load-sample) |

---

## 1. Configurar secrets no Space

1. Abra [Settings → Repository secrets](https://huggingface.co/spaces/salmeida/my-rag-chatbot/settings).
2. Adicione:

| Secret | Descrição |
|--------|-----------|
| `GROQ_API_KEY` | Chave da [Groq Console](https://console.groq.com/) — **obrigatório** |

Secrets opcionais (têm default no código): `GROQ_MODEL`, `GROQ_TEMPERATURE`, `MAX_FILE_SIZE_MB`, `MAX_PAGES`, `ALLOWED_ORIGINS`, etc. Ver `.env.example`.

---

## 2. Configurar remote Git (uma vez)

```bash
git remote add hf https://huggingface.co/spaces/salmeida/my-rag-chatbot
# ou SSH:
# git remote add hf git@hf.co:spaces/salmeida/my-rag-chatbot
```

Token: https://huggingface.co/settings/tokens (permissão **Write**).

---

## 3. Fazer push

```bash
git add -A
git commit -m "Prepare Space deploy"
bash deploy_to_hf.sh
# ou diretamente:
git push hf main
```

O build no Space leva **5–15 min** na primeira vez (PyTorch + sentence-transformers).

---

## 4. Verificar após o build

| URL | Uso |
|-----|-----|
| https://salmeida-my-rag-chatbot.hf.space/health | Health + limites |
| https://salmeida-my-rag-chatbot.hf.space/docs | Swagger |
| https://salmeida-my-rag-chatbot.hf.space/status | Estado do índice |

```bash
# Carregar documento de demonstração
curl -X POST https://salmeida-my-rag-chatbot.hf.space/demo/load-sample

# Pergunta
curl -X POST https://salmeida-my-rag-chatbot.hf.space/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main benefits?"}'
```

---

## Troubleshooting

| Problema | Solução |
|----------|---------|
| Space “missing app file” | Confirme `Dockerfile` na raiz e push na branch `main`. |
| Build falha | Veja **Logs**; confira `requirements.txt` e memória do Space. |
| App reinicia em loop | `GROQ_API_KEY` ausente ou inválida. |
| 503 / initializing | Aguarde download do modelo de embeddings (1ª requisição). |
| `/ask` sem documento | `POST /demo/load-sample` ou `POST /upload` antes. |
| Disco cheio no Space | Não commite `faiss_index/` grande; use load-sample. |

---

## Git LFS (opcional)

Se commitar `faiss_index/`:

```bash
git lfs install
git add faiss_index/
git push hf main
```

Para a demo pública, **não é necessário** — `POST /demo/load-sample` recria o índice em memória.
