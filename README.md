# 🎓 Campus Knowledge AI

An AI-powered Retrieval-Augmented Generation (RAG) application that answers questions from college documents using semantic search and Large Language Models.

The system processes PDF and TXT documents, retrieves the most relevant information using Sentence Transformers and cosine similarity, and generates context-aware responses using Groq's Llama 3.3 70B model.

## 🚀 Live Demo

**Application:** https://campus-knowledge-ai.streamlit.app/

**GitHub Repository:** https://github.com/Vaishnavi-salunke/Campus-Knowledge-AI/edit/main/README.md

---
## ▶️ Run Locally

```bash
git clone https://github.com/vaishnavi-salunke/Campus-Knowledge-AI.git

cd Campus-Knowledge-AI

pip install -r requirements.txt

streamlit run streamlit_app.py
```

Create a `.env` file and add:

```env
GROQ_API_KEY=your_groq_api_key
```


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

## 📖 How to Use

1. Open the application.
2. Select **All Documents** or choose the available brochure from the document selector.
3. Enter a question related to admissions, seat allocation, eligibility criteria, quotas, or other information available in the brochure.
4. The system retrieves the most relevant content using semantic search.
5. Groq Llama 3.3 70B generates an answer based only on the retrieved document context.
6. View the similarity score, source document, and retrieved context in the **Retrieval Details** section.

### Example Questions

* What is the allocation of seats for engineering and technology courses?
* What is the Institutional Quota?
* What are CAP seats?
* How are seats distributed among different categories?
* What is the eligibility criteria for undergraduate technical courses?
* Explain the Centralized Admission Process (CAP).
  
---

## ⚙️ How It Works

1. Documents are loaded from the knowledge base.
2. Text is split into smaller chunks.
3. Sentence Transformers generate embeddings for each chunk.
4. Cosine similarity identifies the most relevant content.
5. Retrieved context is sent to Groq Llama 3.3 70B.
6. The model generates an answer grounded in the retrieved information.

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

* OCR support for scanned PDF documents
* Dynamic indexing of newly uploaded documents
* Persistent vector database
* Multi-document search and filtering
* Hybrid keyword + semantic retrieval
* Citation-based responses


---

## 👩‍💻 Author

**Vaishnavi Salunke**

GitHub: https://github.com/vaishnavi-salunke

LinkedIn: https://www.linkedin.com/in/vaishnavi-salunke19/

---

⭐ If you found this project useful, consider giving it a star.
