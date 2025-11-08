# Instruções para Deploy no Hugging Face Spaces

## Problema
O Space está mostrando: "This Space is missing an app file" porque os arquivos estão apenas no GitHub, não foram enviados para o Hugging Face Space.

## Solução: Fazer Push dos Arquivos

### Opção 1: Usando o Script Automatizado

Execute no terminal:

```bash
bash deploy_to_hf.sh
```

O script vai:
- Verificar se todos os arquivos estão presentes
- Configurar Git LFS se necessário
- Fazer push para o Hugging Face Space

### Opção 2: Manual (Passo a Passo)

#### 1. Verificar se você está na branch main:
```bash
git branch
# Deve mostrar: * main
```

#### 2. Verificar se o remote está configurado:
```bash
git remote -v
# Deve mostrar o remote 'hf' apontando para seu Space
```

#### 3. Obter Token do Hugging Face:
1. Acesse: https://huggingface.co/settings/tokens
2. Clique em "New token"
3. Dê um nome (ex: "space-deploy")
4. Selecione "Write" como permissão
5. Copie o token gerado

#### 4. Fazer Push (escolha uma opção):

**Opção A: Push interativo (vai pedir credenciais)**
```bash
git push hf main
```
- Username: `salmeida`
- Password: Cole o token que você copiou (não sua senha)

**Opção B: Push com token no comando**
```bash
git push https://salmeida:SEU_TOKEN_AQUI@huggingface.co/spaces/salmeida/my-rag-chatbot main
```
Substitua `SEU_TOKEN_AQUI` pelo token que você copiou.

**Opção C: Configurar credenciais uma vez**
```bash
git config credential.helper store
git push hf main
# Digite username e token uma vez, será salvo
```

### 5. Verificar o Deploy

Após o push:
1. Acesse: https://huggingface.co/spaces/salmeida/my-rag-chatbot
2. Vá na aba "Logs" para ver o progresso do build
3. Aguarde alguns minutos (primeiro build pode levar 5-10 minutos)
4. Quando aparecer "Running", sua API estará pronta!

## Arquivos que DEVEM estar no Space

Verifique se estes arquivos estão commitados:
- ✅ `app.py` - Entrypoint da aplicação
- ✅ `main.py` - Código principal da API
- ✅ `Dockerfile` - Configuração Docker
- ✅ `requirements.txt` - Dependências Python
- ✅ `faiss_index/` - Índice vetorial (via Git LFS)
- ✅ `.gitattributes` - Configuração Git LFS

Para verificar:
```bash
git ls-files | grep -E "app.py|main.py|Dockerfile|requirements.txt"
```

## Troubleshooting

### Erro: "could not read Username"
- Você precisa autenticar. Use uma das opções acima.

### Erro: "authentication failed"
- Verifique se o token está correto
- Certifique-se de que o token tem permissão "Write"

### Erro: "refusing to allow a non-fast-forward"
- Use: `git push --force hf main` (cuidado: isso sobrescreve o que está no Space)

### Build falha no Hugging Face
- Verifique os logs na aba "Logs" do Space
- Certifique-se de que o secret `GROQ_API_KEY` está configurado
- Verifique se o Dockerfile está correto

## Após o Deploy Bem-Sucedido

Sua API estará disponível em:
- **URL**: https://huggingface.co/spaces/salmeida/my-rag-chatbot
- **Swagger UI**: https://huggingface.co/spaces/salmeida/my-rag-chatbot/docs
- **Health Check**: https://huggingface.co/spaces/salmeida/my-rag-chatbot/health

## Testando a API

```bash
# Verificar status
curl https://huggingface.co/spaces/salmeida/my-rag-chatbot/

# Upload PDF (exemplo)
curl -X POST https://huggingface.co/spaces/salmeida/my-rag-chatbot/upload \
  -F "file=@documento.pdf"

# Fazer pergunta
curl -X POST https://huggingface.co/spaces/salmeida/my-rag-chatbot/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

