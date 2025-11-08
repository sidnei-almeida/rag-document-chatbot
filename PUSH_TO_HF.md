# 🚀 Como Fazer Push para Hugging Face Space

## ⚠️ IMPORTANTE: Use o remote `hf`, NÃO `origin`

O `origin` vai para GitHub. O `hf` vai para Hugging Face Space.

---

## Método 1: Script Automatizado (RECOMENDADO)

```bash
bash push_to_hf.sh
```

Este script vai:
- Verificar que você está na branch main
- Fazer push FORÇADO para o Hugging Face Space
- Mostrar o progresso

---

## Método 2: Manual (Passo a Passo)

### Passo 1: Obter Token do Hugging Face

1. Acesse: https://huggingface.co/settings/tokens
2. Clique em **"New token"**
3. Nome: `space-deploy`
4. Permissão: **"Write"** (importante!)
5. Clique em **"Generate token"**
6. **COPIE O TOKEN** (você só verá uma vez!)

### Passo 2: Fazer Push

**IMPORTANTE**: Use `hf` não `origin`!

```bash
# Opção A: Push interativo (vai pedir username e password)
git push --force hf main

# Quando pedir:
# Username: salmeida
# Password: COLE O TOKEN (não sua senha!)
```

**OU Opção B: Push com token no comando (mais fácil)**

```bash
# Substitua SEU_TOKEN_AQUI pelo token que você copiou
git push --force https://salmeida:SEU_TOKEN_AQUI@huggingface.co/spaces/salmeida/my-rag-chatbot main
```

**OU Opção C: Configurar credenciais uma vez**

```bash
# Configurar para salvar credenciais
git config credential.helper store

# Fazer push (vai pedir uma vez, depois salva)
git push --force hf main
# Username: salmeida
# Password: SEU_TOKEN
```

---

## ✅ Verificar se Funcionou

Após o push, você deve ver algo como:

```
Enumerating objects: 19, done.
Counting objects: 100% (19/19), done.
Delta compression using up to 4 threads
Compressing objects: 100% (16/16), done.
Writing objects: 100% (16/16), 9.23 KiB | 4.61 MiB/s, done.
Total 16 (delta 0), reused 0 (delta 0)
To https://huggingface.co/spaces/salmeida/my-rag-chatbot
 + abc1234...def5678 main -> main (forced update)
```

---

## 🔍 Verificar no Hugging Face

1. Acesse: https://huggingface.co/spaces/salmeida/my-rag-chatbot
2. Vá na aba **"Files"** - você deve ver:
   - ✅ `app.py`
   - ✅ `main.py`
   - ✅ `Dockerfile`
   - ✅ `requirements.txt`
   - ✅ `faiss_index/` (pasta)

3. Vá na aba **"Logs"** - você verá o build em progresso

---

## 🐛 Troubleshooting

### Erro: "could not read Username"
- Você precisa autenticar. Use uma das opções acima.

### Erro: "authentication failed"
- Verifique se o token está correto
- Certifique-se que o token tem permissão **"Write"**
- Tente criar um novo token

### Erro: "remote: error: GH013" (GitHub push protection)
- Isso é do GitHub, ignore. Estamos fazendo push para Hugging Face, não GitHub.

### Nada aparece no Hugging Face após push
- Aguarde 1-2 minutos e atualize a página
- Verifique a aba "Logs" para ver se o build começou
- Certifique-se que usou `hf` e não `origin`

### "This Space is missing an app file"
- Significa que o push não foi feito ou não funcionou
- Execute novamente: `git push --force hf main`
- Verifique se `app.py` aparece na aba "Files"

---

## 📝 Checklist Antes do Push

- [ ] Está na branch `main`? (`git branch`)
- [ ] Todos os arquivos estão commitados? (`git status`)
- [ ] Tem o token do Hugging Face com permissão Write?
- [ ] Vai usar `hf` e não `origin`?

---

## 🎯 Comando Rápido (Copie e Cole)

```bash
# 1. Verificar branch
git branch

# 2. Verificar arquivos
git ls-files | grep -E "app.py|main.py|Dockerfile"

# 3. Fazer push (SUBSTITUA SEU_TOKEN)
git push --force https://salmeida:SEU_TOKEN@huggingface.co/spaces/salmeida/my-rag-chatbot main
```

---

## 💡 Dica

Se você sempre quer fazer push para ambos (GitHub e HF), pode fazer:

```bash
# Push para GitHub
git push origin main

# Push para Hugging Face
git push hf main
```

Mas para o deploy do Space, você PRECISA fazer push para `hf`!

