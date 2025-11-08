"""
DocMind API Usage Example
"""
import requests
import json

# API URL (adjust as needed)
API_URL = "http://localhost:8000"

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
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.json()}")
    return response.json()

if __name__ == "__main__":
    # Check status
    print("Checking API status...")
    check_status()
    
    # Example questions
    questions = [
        "What is the main theme of the document?",
        "Summarize the document in 3 main points.",
        "What are the conclusions presented?",
    ]
    
    for question in questions:
        ask_question(question)
        print("\n" + "="*50 + "\n")
