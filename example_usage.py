"""
DocMind API Usage Example
"""
import requests
import json
import os

# API URL (adjust as needed)
API_URL = "http://localhost:8000"

def upload_pdf(pdf_path: str):
    """Upload a PDF file to the API"""
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File {pdf_path} not found")
        return None
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
        response = requests.post(f"{API_URL}/upload", files=files)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n📄 Uploaded: {data['filename']}")
        print(f"✅ Pages: {data['pages']}")
        print(f"✅ Chunks: {data['chunks']}")
        print(f"✅ Status: {data['status']}")
        return data
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

def ask_question(question: str):
    """Ask a question to the API"""
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n❓ Question: {question}")
        print(f"✅ Answer: {data['answer']}")
        print(f"📄 Sources (pages): {data['sources']}")
        return data
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return None

def check_status():
    """Check API status"""
    response = requests.get(f"{API_URL}/")
    data = response.json()
    print(f"Status: {data['status']}")
    print(f"Health: {data['health']}")
    print(f"Index Ready: {data['index_ready']}")
    print(f"Endpoints: {data['endpoints']}")
    return data

if __name__ == "__main__":
    # Check status
    print("Checking API status...")
    check_status()
    
    # Example: Upload a PDF (if you have one)
    # pdf_file = "documento.pdf"
    # if os.path.exists(pdf_file):
    #     print("\n" + "="*50)
    #     print("Uploading PDF...")
    #     upload_pdf(pdf_file)
    #     print("="*50 + "\n")
    
    # Example questions
    questions = [
        "What is the main theme of the document?",
        "Summarize the document in 3 main points.",
        "What are the conclusions presented?",
    ]
    
    for question in questions:
        ask_question(question)
        print("\n" + "="*50 + "\n")
