import os
# Importando as ferramentas necessárias do LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuração ---
NOME_ARQUIVO_PDF = "documento.pdf"
NOME_BANCO_VETORIAL = "faiss_index"

def main():
    # 1. Verificação básica
    if not os.path.exists(NOME_ARQUIVO_PDF):
        print(f"ERRO: Não encontrei o arquivo '{NOME_ARQUIVO_PDF}' na pasta.")
        return

    print(f"--> Começando a ler: {NOME_ARQUIVO_PDF}")

    # 2. Carregar o PDF
    loader = PyPDFLoader(NOME_ARQUIVO_PDF)
    documentos = loader.load()
    print(f"    PDF carregado. Total de páginas lidas: {len(documentos)}")

    # 3. Dividir em pedaços (Chunks)
    # A IA não consegue ler um livro inteiro de uma vez. Precisamos quebrar em pedaços.
    # chunk_size=1000: cada pedaço terá +/- 1000 caracteres.
    # chunk_overlap=100: cada pedaço compartilha um pouquinho com o anterior para não perder contexto.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    textos_divididos = text_splitter.split_documents(documentos)
    print(f"    Documento dividido em {len(textos_divididos)} pedaços menores.")

    # 4. Criar os "Embeddings" (Vetorização)
    # Aqui a mágica acontece. Vamos baixar um modelo PEQUENO e GRATUITO
    # da Hugging Face que roda na sua CPU. Ele transforma texto em números.
    print("--> Baixando modelo de Embeddings (pode demorar um pouquinho na 1ª vez)...")
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("    Modelo carregado!")

    # 5. Criar e Salvar o Banco de Dados (FAISS)
    # O FAISS pega os textos e os números (vetores) e cria um índice super rápido.
    print("--> Criando o banco de dados vetorial (FAISS)...")
    vector_store = FAISS.from_documents(textos_divididos, embeddings_model)
    
    # Salvando no disco para usarmos depois na API
    vector_store.save_local(NOME_BANCO_VETORIAL)
    print(f"--> SUCESSO! Banco de dados salvo na pasta: '{NOME_BANCO_VETORIAL}'")

if __name__ == "__main__":
    main()
