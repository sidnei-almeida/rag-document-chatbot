#!/bin/bash

# Script para fazer push APENAS para Hugging Face Space
# NÃO vai para GitHub

echo "🚀 Fazendo push para Hugging Face Space..."
echo ""

# Verificar se está na branch main
CURRENT_BRANCH=$(git branch --show-current)
echo "📍 Branch atual: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "⚠️  Mudando para branch main..."
    git checkout main
fi

# Listar arquivos que serão enviados
echo ""
echo "📋 Arquivos que serão enviados:"
git ls-files | grep -E "app.py|main.py|Dockerfile|requirements.txt|faiss_index" | head -10
echo ""

# Verificar remote
echo "🔗 Remote configurado:"
git remote get-url hf
echo ""

# Fazer push FORÇADO para o Hugging Face (isso vai SOBRESCREVER o que está lá)
echo "📤 Fazendo push para Hugging Face Space..."
echo "   ⚠️  Isso vai sobrescrever o conteúdo atual do Space"
echo ""
echo "   Você precisará inserir:"
echo "   Username: salmeida"
echo "   Password: Seu token do Hugging Face"
echo ""
echo "   Para criar token: https://huggingface.co/settings/tokens"
echo ""

# Push forçado para garantir que vai
git push --force hf main

echo ""
if [ $? -eq 0 ]; then
    echo "✅ Push concluído com sucesso!"
    echo "🌐 Space: https://huggingface.co/spaces/salmeida/my-rag-chatbot"
    echo "⏳ Aguarde o build completar (5-10 minutos)"
else
    echo "❌ Erro no push. Tente manualmente:"
    echo "   git push --force hf main"
fi

