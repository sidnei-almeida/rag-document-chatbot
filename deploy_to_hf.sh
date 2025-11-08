#!/bin/bash

# Script para fazer deploy no Hugging Face Spaces
# Execute: bash deploy_to_hf.sh

echo "🚀 Preparando deploy para Hugging Face Spaces..."
echo ""

# Verificar se está na branch main
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo "⚠️  Você está na branch $CURRENT_BRANCH. Mudando para main..."
    git checkout main
fi

# Verificar se há mudanças não commitadas
if [ -n "$(git status --porcelain)" ]; then
    echo "⚠️  Há mudanças não commitadas. Adicionando e fazendo commit..."
    git add .
    git commit -m "Update before deploy"
fi

# Verificar se os arquivos essenciais existem
echo "📋 Verificando arquivos essenciais..."
FILES=("app.py" "main.py" "Dockerfile" "requirements.txt")
MISSING_FILES=()

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "❌ Arquivos faltando: ${MISSING_FILES[*]}"
    exit 1
fi

echo "✅ Todos os arquivos essenciais estão presentes"
echo ""

# Verificar Git LFS
echo "📦 Verificando Git LFS..."
if git lfs ls-files | grep -q "faiss_index"; then
    echo "✅ Git LFS configurado corretamente"
else
    echo "⚠️  Configurando Git LFS..."
    git lfs track "faiss_index/*"
    git add .gitattributes
    git commit -m "Configure Git LFS" || true
fi

echo ""
echo "📤 Fazendo push para Hugging Face Spaces..."
echo "   Você precisará inserir suas credenciais do Hugging Face"
echo "   Username: salmeida"
echo "   Password: Seu token do Hugging Face (não sua senha)"
echo ""
echo "   Para criar um token: https://huggingface.co/settings/tokens"
echo ""

# Tentar fazer push
git push hf main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Deploy concluído com sucesso!"
    echo "🌐 Seu Space estará disponível em:"
    echo "   https://huggingface.co/spaces/salmeida/my-rag-chatbot"
    echo ""
    echo "⏳ Aguarde alguns minutos para o build completar..."
else
    echo ""
    echo "❌ Erro ao fazer push. Verifique suas credenciais."
    echo ""
    echo "💡 Alternativa: Use o token diretamente:"
    echo "   git push https://salmeida:SEU_TOKEN@huggingface.co/spaces/salmeida/my-rag-chatbot main"
fi

