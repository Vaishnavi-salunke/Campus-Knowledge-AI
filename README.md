# 🎓 Campus Knowledge AI

An AI-powered Retrieval-Augmented Generation (RAG) application that answers questions from college documents using semantic search and Large Language Models.

The system processes PDF and TXT documents, retrieves the most relevant information using Sentence Transformers and cosine similarity, and generates context-aware responses using Groq's Llama 3.3 70B model.

## 🚀 Live Demo

**Application:** https://campus-knowledge-ai.streamlit.app/

**GitHub Repository:** https://github.com/Vaishnavi-salunke/Campus-Knowledge-AI/edit/main/README.md

---

## ✨ Features

* Ask questions about college documents in natural language
* PDF and TXT document support
* Semantic search using Sentence Transformers
* Context-aware answer generation using Groq Llama 3.3 70B
* Similarity score display
* Source document tracking
* Streamlit-based interactive UI
* Retrieval details for transparency

---

## 🛠️ Tech Stack

* Python
* Streamlit
* Groq API
* Sentence Transformers
* Scikit-Learn
* Hugging Face Transformers
* NumPy
* PyPDF

---

## 📂 Project Structure

```text
Campus-Knowledge-AI
│
├── data/
├── src/
│   └── rag/
│       ├── document_loader.py
│       ├── text_chunker.py
│       ├── rag_engine.py
│       └── ...
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ How It Works

1. Load PDF/TXT documents
2. Extract and chunk text
3. Generate embeddings using all-MiniLM-L6-v2
4. Perform semantic retrieval using cosine similarity
5. Send retrieved context to Groq Llama 3.3 70B
6. Generate accurate answers grounded in document content

---

## 🖼️ Screenshots

Add screenshots of:

* Home Page
  <img width="1904" height="879" alt="image" src="https://github.com/user-attachments/assets/0cdf5e5c-1a30-4a23-a630-bb798ac21907" />

 

* Question Answering Interface
  <img width="1876" height="866" alt="image" src="https://github.com/user-attachments/assets/0a4e0365-ec6d-4cdc-b8f5-0ee3823bc345" />

  
  
 <img width="1895" height="877" alt="image" src="https://github.com/user-attachments/assets/713c00b3-56db-49c5-b3c8-aa0b9932c9c5" />

  

* Retrieval Details Panel
  <img width="1898" height="871" alt="image" src="https://github.com/user-attachments/assets/51e77ad2-cb1d-4fb4-a599-e1f3124f61ea" />


---

## 🔮 Future Improvements

* OCR support for scanned PDFs
* Multi-document filtering
* Persistent vector database
* Hybrid search
* Multi-file upload indexing

---

## 👩‍💻 Author

**Vaishnavi Salunke**

GitHub: https://github.com/vaishnavi-salunke

LinkedIn: https://www.linkedin.com/in/vaishnavi-salunke19/

---

⭐ If you found this project useful, consider giving it a star.
