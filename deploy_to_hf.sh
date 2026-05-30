#!/usr/bin/env bash
# Push do repositório para o Hugging Face Space (Docker SDK).
# Uso: bash deploy_to_hf.sh
set -euo pipefail

HF_SPACE="salmeida/my-rag-chatbot"
HF_REMOTE="${HF_REMOTE:-hf}"

echo "==> DocMind — deploy para Hugging Face Space: ${HF_SPACE}"

required_files=(
  Dockerfile
  requirements.txt
  main.py
  app.py
  app/main.py
  sample_documents/ai-document-intelligence-report.pdf
)

for f in "${required_files[@]}"; do
  if [[ ! -e "$f" ]]; then
    echo "ERRO: arquivo obrigatório ausente: $f"
    exit 1
  fi
done

echo "==> Arquivos obrigatórios OK"

if ! git remote get-url "$HF_REMOTE" &>/dev/null; then
  echo "==> Remote '${HF_REMOTE}' não encontrado. Adicionando..."
  git remote add "$HF_REMOTE" "https://huggingface.co/spaces/${HF_SPACE}"
fi

echo "==> Remote ${HF_REMOTE}: $(git remote get-url "$HF_REMOTE")"
echo "==> Branch atual: $(git branch --show-current)"
echo ""
echo "Próximo passo: git push ${HF_REMOTE} main"
echo "  Username: seu usuário HF (ex: salmeida)"
echo "  Password: token com permissão WRITE (https://huggingface.co/settings/tokens)"
echo ""
read -r -p "Executar push agora? [y/N] " confirm
if [[ "${confirm,,}" == "y" ]]; then
  git push "$HF_REMOTE" main
  echo ""
  echo "Deploy iniciado. Acompanhe:"
  echo "  https://huggingface.co/spaces/${HF_SPACE}"
  echo "  Logs → Build / Running"
fi
