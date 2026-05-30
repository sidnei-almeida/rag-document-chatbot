# Relatório de Avaliação RAG — DocMind

**Projeto:** `rag-document-chatbot`  
**Data da execução:** 30 de maio de 2026  
**Ambiente:** API local (`http://127.0.0.1:7860`) com variáveis do arquivo `.env`  
**Modo de avaliação:** `python evals/run_eval.py --mode api`  
**Documento de teste:** `sample_documents/ai-document-intelligence-report.pdf`

---

## 1. Resumo executivo

A avaliação automatizada verificou se o pipeline RAG **recupera fontes** e produz respostas **alinhadas ao documento sample**, usando critérios de palavras-chave e quantidade mínima de trechos (`sources`).

| Indicador | Resultado |
|-----------|-----------|
| Total de casos | 8 |
| Aprovados | **8** |
| Reprovados | **0** |
| Taxa de sucesso | **100%** |
| Status geral | **APROVADO** |

O índice FAISS foi carregado via `POST /demo/load-sample` antes das perguntas. Cada pergunta foi enviada a `POST /ask` com o modelo Groq configurado (`llama-3.3-70b-versatile`).

---

## 2. Configuração do ambiente

| Parâmetro | Valor |
|-----------|--------|
| Modelo LLM | `llama-3.3-70b-versatile` |
| Temperatura | `0.15` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retriever | MMR (`k=6`, `fetch_k=20`, `lambda=0.7`) |
| Chunk size / overlap | `1200` / `180` |
| Documento sample | 7 páginas → **7 chunks** indexados |

**Limites da demo pública:** PDF até 8 MB, 40 páginas, perguntas até 1000 caracteres.

---

## 3. Metodologia

Para cada item em `evals/questions.json`:

1. **Fontes:** `sources.length >= expected_min_sources`
2. **Keywords:** pelo menos `min_keyword_matches` termos de `expected_keywords` presentes na resposta e/ou nos previews das fontes (`match_in`: `both` ou `sources`)
3. **Sem alucinação vazia:** respostas do tipo *"could not find enough evidence"* contam como falha

Não foram usados LangSmith, RAGAS ou serviços pagos de avaliação.

---

## 4. Resultados por pergunta

| ID | Pergunta | Fontes | Keywords | Resultado |
|----|----------|--------|----------|-----------|
| `about` | What is this document about? | 6 | 4/4 critérios | PASS |
| `benefits` | What are the main benefits? | 6 | 4 | PASS |
| `limitations` | What are the limitations mentioned in the document? | 6 | 4 | PASS |
| `technologies` | What technologies are used? | 6 | 4 | PASS |
| `embeddings` | What embedding model does the system use? | 6 | 3 | PASS |
| `summarize` | Summarize the document. | 6 | 4 | PASS |
| `deployment` | Where is the system intended to be deployed? | 6 | 3 | PASS |
| `confidence` | Is the confidence score a calibrated probability? | 6 | 3 | PASS |

Todas as perguntas recuperaram **6 trechos** (valor configurado em `RETRIEVAL_K`), indicando retrieval estável após o load do sample.

---

## 5. Histórico da sessão de testes

### 5.1 Primeira execução (chave Groq placeholder)

- **Problema:** `GROQ_API_KEY` ainda era `your_groq_api_key_here`
- **Efeito:** `POST /ask` retornou HTTP 500 (401 Invalid API Key do Groq)
- **Eval API:** 0/8
- **Eval local (só retrieval):** 8/8 — confirmou que ingestão e FAISS estavam corretos

### 5.2 Segunda execução (chave real, antes do fix)

- **Eval API:** 7/8
- **Falha:** pergunta `about` — detectada erroneamente como “conversa geral” porque a substring `"hi"` aparecia em **"th**is**"**
- **Correção:** `is_general_question()` passou a usar limites de palavra (`\b`)

### 5.3 Execução final (após correção + chave real)

- **Eval API:** **8/8** — relatório salvo em `evals/results.json`

---

## 6. Conclusões

**Pontos fortes**

- Pipeline completo funcional: sample → chunks → FAISS → retrieval → Groq → resposta com `sources` estruturadas
- Cobertura consistente de temas do PDF (benefícios, limitações, stack, deploy, confidence heurística)
- Avaliação reproduzível e gratuita (`evals/run_eval.py`)

**Riscos / limitações conhecidas**

- Avaliação por keywords não mede qualidade semântica da resposta do LLM
- Dependência de API Groq na execução `--mode api`
- Perguntas muito genéricas precisam de detecção cuidadosa (corrigido o caso `hi` ⊂ `this`)

---

## 7. Como reproduzir

```bash
# Terminal 1 — API
set -a && source .env && set +a
uvicorn main:app --host 127.0.0.1 --port 7860

# Terminal 2 — Avaliação
python evals/run_eval.py --mode api --api-url http://127.0.0.1:7860 --output evals/results.json
```

Avaliação só retrieval (sem Groq):

```bash
python evals/run_eval.py --mode local
```

---

## 8. Artefatos

| Arquivo | Descrição |
|---------|-----------|
| `evals/questions.json` | Casos de teste |
| `evals/results.json` | Último resultado JSON (8/8) |
| `evals/scoring.py` | Lógica de pontuação |
| `evals/run_eval.py` | Script executor |

---

*Relatório gerado para documentação de portfólio do projeto DocMind.*
