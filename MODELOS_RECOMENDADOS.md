# Modelos Recomendados para Conta Free do Hugging Face

Esta lista contém modelos que geralmente funcionam bem com contas gratuitas do Hugging Face Inference API.

## Modelos Pequenos e Rápidos (Recomendados)

### 1. microsoft/phi-2 ⭐ RECOMENDADO
- **Tamanho**: 2.7B parâmetros
- **Tipo**: text-generation
- **Vantagens**: Pequeno, rápido, bom desempenho
- **Uso**: Não precisa de `task` parameter
- **Status**: Geralmente disponível

### 2. google/flan-t5-base
- **Tamanho**: 250M parâmetros  
- **Tipo**: text2text-generation
- **Vantagens**: Muito leve, rápido
- **Uso**: Precisa de `task="text2text-generation"`
- **Status**: Pode estar instável

### 3. google/flan-t5-small
- **Tamanho**: 60M parâmetros
- **Tipo**: text2text-generation
- **Vantagens**: Extremamente leve
- **Uso**: Precisa de `task="text2text-generation"`
- **Status**: Pode estar instável

## Modelos Médios

### 4. mistralai/Mistral-7B-Instruct-v0.2
- **Tamanho**: 7B parâmetros
- **Tipo**: text-generation
- **Vantagens**: Boa qualidade, instruções
- **Uso**: Não precisa de `task` parameter
- **Status**: Pode ter limitações na conta free

### 5. meta-llama/Llama-2-7b-chat-hf
- **Tamanho**: 7B parâmetros
- **Tipo**: text-generation
- **Vantagens**: Muito popular, boa qualidade
- **Uso**: Pode precisar de acesso especial
- **Status**: Pode requerer aprovação

## Como Trocar o Modelo

Edite o arquivo `main.py` e altere a linha:

```python
REPO_ID_MODELO = "microsoft/phi-2"
```

Para modelos text2text-generation (flan-t5), adicione `task="text2text-generation"`:

```python
llm = HuggingFaceEndpoint(
    repo_id=REPO_ID_MODELO,
    huggingfacehub_api_token=HF_TOKEN,
    task="text2text-generation",  # Apenas para modelos flan-t5
    temperature=0.3,
    max_new_tokens=512,
)
```

Para modelos text-generation (phi-2, mistral, llama), não use o parâmetro `task`:

```python
llm = HuggingFaceEndpoint(
    repo_id=REPO_ID_MODELO,
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.3,
    max_new_tokens=512,
)
```

## Troubleshooting

Se um modelo não funcionar:
1. Verifique se o modelo está disponível em: https://huggingface.co/{MODELO}
2. Verifique os logs do servidor para ver o erro completo
3. Tente outro modelo da lista acima
4. Modelos menores geralmente funcionam melhor com contas free

