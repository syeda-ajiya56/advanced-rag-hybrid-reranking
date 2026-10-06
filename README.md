# Advanced RAG with Hybrid Search and Reranking

An advanced Retrieval-Augmented Generation (RAG) document question-answering system that combines semantic search, BM25 keyword search, hybrid retrieval, and reranking to provide accurate, document-grounded answers.

## Overview

This project upgrades a basic RAG system by combining two different retrieval approaches:

- **Semantic Search** finds documents based on meaning and context.
- **BM25 Keyword Search** finds documents based on matching keywords.
- **Hybrid Search** combines both retrieval methods.
- **Reranking** reorders the retrieved results based on their relevance to the user's question.
- **Grounded Generation** generates answers using only the retrieved document context.

The system also includes a safe fallback for questions whose answers are not available in the document knowledge base.

## Architecture

```text
User Question
      ↓
Query Embedding
      ↓
Semantic / Vector Search
      ↓
BM25 Keyword Search
      ↓
Hybrid Search
      ↓
Reranking
      ↓
Top Relevant Documents
      ↓
Gemini Grounded Generation
      ↓
Final Answer
```

## Technologies Used

- Python
- Streamlit
- Google Gemini API
- Gemini Embeddings
- ChromaDB
- BM25
- rank-bm25
- python-dotenv

## Project Structure

```text
advanced-rag-hybrid-reranking/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── documents/
    ├── academic_programs.txt
    ├── library_services.txt
    └── internships.txt
```

## Document Knowledge Base

The application uses a small university knowledge base containing information about:

### Academic Programs

Information about:

- Computer Science
- Software Engineering
- Information Technology
- Program areas and curriculum focus

### Library Services

Information about:

- Books
- Academic journals
- Digital resources
- Study spaces
- Computer facilities
- Library catalogue

### Internship Program

Information about:

- Internship opportunities
- Practical experience
- Career and internship services
- Internship eligibility and application procedures

## Retrieval Pipeline

### 1. Semantic Search

The application converts documents and user queries into embeddings using the Gemini embedding model.

ChromaDB is used to store and retrieve the most semantically relevant document chunks.

### 2. BM25 Keyword Search

BM25 provides keyword-based retrieval using the `rank-bm25` library.

This helps identify documents containing important words from the user's query.

### 3. Hybrid Search

The system combines semantic and BM25 retrieval scores.

The hybrid score uses:

```text
Hybrid Score =
0.6 × Semantic Score
+
0.4 × Normalized BM25 Score
```

This combines semantic relevance with keyword relevance.

### 4. Reranking

The hybrid results are reranked using:

- Hybrid relevance score
- Query-term overlap

The reranking score is calculated using:

```text
Rerank Score =
0.7 × Hybrid Score
+
0.3 × Query Match Score
```

The top-ranked documents are then used as context for answer generation.

## Grounded Answer Generation

The final answer is generated using Gemini.

The model is instructed to:

- Use only the retrieved document context
- Avoid outside knowledge
- Keep answers concise
- Refuse to provide unsupported information

For unsupported questions, the application returns:

```text
I don't have enough reliable information to answer that from the provided documents.
```

## Example Query

### Question

```text
What services does the university library provide?
```

### Retrieval

The system retrieves `library_services.txt` as the most relevant document through semantic search, BM25, hybrid search, and reranking.

### Final Answer

```text
Based on the provided documents, the university library provides students
with access to books, academic journals, digital resources, study spaces,
and computer facilities, as well as quiet study areas for individual
academic work.
```

## Unsupported Question Test

### Question

```text
How much does university accommodation cost?
```

The information is not available in the document knowledge base.

### System Response

```text
I don't have enough reliable information to answer that from the provided documents.
```

This prevents the system from inventing information that is not supported by the documents.

## Installation

Clone the repository:

```bash
git clone https://github.com/syeda-ajiya56/advanced-rag-hybrid-reranking.git
```

Enter the project directory:

```bash
cd advanced-rag-hybrid-reranking
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

Do not commit the `.env` file to GitHub.

## Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in the browser.

## Testing

The application was tested with:

### Test 1 — Successful Q&A

```text
What services does the university library provide?
```

The system successfully retrieved the relevant library document and generated a grounded answer.

### Test 2 — Unsupported Question

```text
How much does university accommodation cost?
```

The system correctly reported that the information was not available in the provided documents.

## Key Features Demonstrated

- Document knowledge base
- Text chunking
- Vector embeddings
- ChromaDB vector retrieval
- BM25 keyword retrieval
- Hybrid search
- Retrieval score combination
- Reranking
- Grounded generation
- Unsupported-question handling
- Streamlit interface

## Limitations

This project uses a small demonstration knowledge base. The reranking approach is lightweight and designed for demonstrating the advanced RAG pipeline.

For a larger production system, the pipeline could be extended with:

- More documents
- Smaller and overlapping chunks
- Dedicated cross-encoder reranking
- Metadata filtering
- Larger evaluation datasets
- Retrieval quality metrics
- Citation generation
- Document upload functionality

## Repository

GitHub:

https://github.com/syeda-ajiya56/advanced-rag-hybrid-reranking