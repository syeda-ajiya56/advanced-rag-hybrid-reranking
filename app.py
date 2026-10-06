import os
import re
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY was not found in the .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# Embedding model
EMBEDDING_MODEL = "gemini-embedding-001"

# Primary generation model
GENERATION_MODEL = "gemini-3.5-flash"

# Backup models in case the primary model is temporarily busy
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash-lite",
]

DOCUMENT_DIR = Path("documents")
CHROMA_DIR = "./chroma_db"

TOP_K_VECTOR = 5
TOP_K_BM25 = 5
TOP_K_HYBRID = 5
TOP_K_RERANK = 3


# =========================================================
# TEXT PROCESSING
# =========================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


# =========================================================
# DOCUMENT LOADING
# =========================================================

def load_documents():

    documents = []

    if not DOCUMENT_DIR.exists():
        st.error("The documents folder was not found.")
        st.stop()

    for file_path in sorted(DOCUMENT_DIR.glob("*.txt")):

        text = file_path.read_text(
            encoding="utf-8"
        )

        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):

            documents.append(
                {
                    "id": f"{file_path.stem}-{index}",
                    "source": file_path.name,
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    if not documents:
        st.error(
            "No .txt documents were found inside the documents folder."
        )
        st.stop()

    return documents


# =========================================================
# EMBEDDINGS
# =========================================================

def create_embeddings(texts):

    embeddings = []

    for text in texts:

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )

        embeddings.append(
            response.embeddings[0].values
        )

    return embeddings


# =========================================================
# CHROMA VECTOR DATABASE
# =========================================================

@st.cache_resource
def build_vector_database():

    documents = load_documents()

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = chroma_client.get_or_create_collection(
        name="advanced_rag_documents"
    )

    existing_ids = set(
        collection.get()["ids"]
    )

    new_documents = [
        document
        for document in documents
        if document["id"] not in existing_ids
    ]

    if new_documents:

        texts = [
            document["text"]
            for document in new_documents
        ]

        embeddings = create_embeddings(texts)

        collection.add(
            ids=[
                document["id"]
                for document in new_documents
            ],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": document["source"],
                    "chunk_index": document["chunk_index"],
                }
                for document in new_documents
            ],
        )

    return collection, documents


# =========================================================
# BM25 INDEX
# =========================================================

@st.cache_resource
def build_bm25_index():

    documents = load_documents()

    corpus = [
        tokenize(document["text"])
        for document in documents
    ]

    bm25 = BM25Okapi(corpus)

    return bm25, documents


# =========================================================
# SEMANTIC / VECTOR SEARCH
# =========================================================

def semantic_search(
    query,
    collection,
    documents
):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
    )

    query_embedding = (
        response.embeddings[0].values
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            TOP_K_VECTOR,
            len(documents)
        ),
    )

    output = []

    result_ids = results["ids"][0]
    result_documents = results["documents"][0]
    result_distances = results["distances"][0]
    result_metadatas = results["metadatas"][0]

    for rank, (
        result_id,
        text,
        distance,
        metadata,
    ) in enumerate(
        zip(
            result_ids,
            result_documents,
            result_distances,
            result_metadatas,
        ),
        start=1,
    ):

        semantic_score = 1 / (
            1 + distance
        )

        output.append(
            {
                "id": result_id,
                "text": text,
                "source": metadata["source"],
                "semantic_score": semantic_score,
                "semantic_rank": rank,
            }
        )

    return output


# =========================================================
# BM25 KEYWORD SEARCH
# =========================================================

def bm25_search(
    query,
    bm25,
    documents
):

    scores = bm25.get_scores(
        tokenize(query)
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for rank, index in enumerate(
        ranked_indices[:TOP_K_BM25],
        start=1,
    ):

        document = documents[index]

        results.append(
            {
                "id": document["id"],
                "text": document["text"],
                "source": document["source"],
                "bm25_score": float(
                    scores[index]
                ),
                "bm25_rank": rank,
            }
        )

    return results


# =========================================================
# HYBRID SEARCH
# =========================================================

def hybrid_search(
    query,
    vector_results,
    keyword_results
):

    combined = {}

    max_bm25 = max(
        [
            item["bm25_score"]
            for item in keyword_results
        ],
        default=1,
    )

    if max_bm25 == 0:
        max_bm25 = 1

    # Add semantic results
    for item in vector_results:

        combined[item["id"]] = {
            "id": item["id"],
            "text": item["text"],
            "source": item["source"],
            "semantic_score": item[
                "semantic_score"
            ],
            "bm25_score": 0,
        }

    # Add BM25 results
    for item in keyword_results:

        if item["id"] not in combined:

            combined[item["id"]] = {
                "id": item["id"],
                "text": item["text"],
                "source": item["source"],
                "semantic_score": 0,
                "bm25_score": item[
                    "bm25_score"
                ],
            }

        else:

            combined[item["id"]][
                "bm25_score"
            ] = item["bm25_score"]

    # Calculate hybrid score
    for item in combined.values():

        normalized_bm25 = (
            item["bm25_score"]
            / max_bm25
        )

        item["hybrid_score"] = (
            0.6 * item["semantic_score"]
            + 0.4 * normalized_bm25
        )

    return sorted(
        combined.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )[:TOP_K_HYBRID]


# =========================================================
# RERANKING
# =========================================================

def rerank_results(
    query,
    hybrid_results
):

    query_tokens = set(
        tokenize(query)
    )

    reranked = []

    for item in hybrid_results:

        text_tokens = set(
            tokenize(item["text"])
        )

        overlap = len(
            query_tokens.intersection(
                text_tokens
            )
        )

        query_match_score = (
            overlap
            / max(len(query_tokens), 1)
        )

        final_score = (
            0.7 * item["hybrid_score"]
            + 0.3 * query_match_score
        )

        reranked_item = item.copy()

        reranked_item[
            "query_match_score"
        ] = query_match_score

        reranked_item[
            "rerank_score"
        ] = final_score

        reranked.append(
            reranked_item
        )

    return sorted(
        reranked,
        key=lambda x: x["rerank_score"],
        reverse=True,
    )[:TOP_K_RERANK]


# =========================================================
# GROUNDED ANSWER GENERATION
# =========================================================

def generate_answer(
    query,
    reranked_results
):

    context_parts = []

    for item in reranked_results:

        context_parts.append(
            f"Source: {item['source']}\n"
            f"{item['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question ONLY using
the provided document context.

Do not use outside knowledge.

If the answer is not supported by
the documents, respond exactly with:

I don't have enough reliable information to answer that from the provided documents.

Keep the answer concise and clear.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}
"""

    # Try primary model first
    models_to_try = [
        GENERATION_MODEL
    ] + FALLBACK_MODELS

    last_error = None

    for model_name in models_to_try:

        try:

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            if response.text:
                return response.text

        except Exception as error:

            last_error = error

            continue

    # If every model is temporarily unavailable
    return (
        "The document retrieval and reranking "
        "completed successfully, but the AI "
        "answer service is temporarily unavailable. "
        "Please try again."
    )


# =========================================================
# STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Advanced RAG - Hybrid Search & Reranking",
    page_icon="🔎",
    layout="wide",
)


# =========================================================
# TITLE
# =========================================================

st.title(
    "🔎 Advanced RAG with Hybrid Search & Reranking"
)

st.write(
    "A document question-answering system using "
    "semantic search, BM25 keyword search, "
    "hybrid retrieval, reranking, and Gemini."
)


# =========================================================
# BUILD KNOWLEDGE BASE
# =========================================================

with st.spinner(
    "Loading document knowledge base..."
):

    collection, documents = (
        build_vector_database()
    )

    bm25, documents = (
        build_bm25_index()
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "📚 Knowledge Base"
)

st.sidebar.write(
    f"Documents: "
    f"{len(set(d['source'] for d in documents))}"
)

st.sidebar.write(
    f"Chunks: {len(documents)}"
)

st.sidebar.write(
    "Search methods:"
)

st.sidebar.write(
    "• Semantic / Vector Search"
)

st.sidebar.write(
    "• BM25 Keyword Search"
)

st.sidebar.write(
    "• Hybrid Search"
)

st.sidebar.write(
    "• Reranking"
)


# =========================================================
# DOCUMENT KNOWLEDGE BASE
# =========================================================

with st.expander(
    "📚 View Document Knowledge Base"
):

    for document in documents:

        st.markdown(
            f"**{document['source']} — "
            f"Chunk {document['chunk_index']}**"
        )

        st.write(
            document["text"]
        )

        st.divider()


# =========================================================
# QUESTION INPUT
# =========================================================

st.subheader(
    "Ask a Question"
)

query = st.text_input(
    "Enter your question:",
    placeholder=(
        "Example: What services does "
        "the university library provide?"
    ),
)


# =========================================================
# SEARCH BUTTON
# =========================================================

if st.button(
    "🔍 Search & Answer"
):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # -------------------------------------------------
        # RETRIEVAL PIPELINE
        # -------------------------------------------------

        with st.spinner(
            "Running hybrid retrieval and reranking..."
        ):

            # Step 1
            semantic_results = (
                semantic_search(
                    query,
                    collection,
                    documents,
                )
            )

            # Step 2
            keyword_results = (
                bm25_search(
                    query,
                    bm25,
                    documents,
                )
            )

            # Step 3
            hybrid_results = (
                hybrid_search(
                    query,
                    semantic_results,
                    keyword_results,
                )
            )

            # Step 4
            reranked_results = (
                rerank_results(
                    query,
                    hybrid_results,
                )
            )

        # -------------------------------------------------
        # STEP 1: SEMANTIC SEARCH
        # -------------------------------------------------

        st.subheader(
            "1️⃣ Semantic Search Results"
        )

        for result in semantic_results:

            st.write(
                f"**{result['source']}** | "
                f"Semantic Score: "
                f"{result['semantic_score']:.4f}"
            )

            st.caption(
                result["text"]
            )

        # -------------------------------------------------
        # STEP 2: BM25 SEARCH
        # -------------------------------------------------

        st.subheader(
            "2️⃣ BM25 Keyword Search Results"
        )

        for result in keyword_results:

            st.write(
                f"**{result['source']}** | "
                f"BM25 Score: "
                f"{result['bm25_score']:.4f}"
            )

            st.caption(
                result["text"]
            )

        # -------------------------------------------------
        # STEP 3: HYBRID SEARCH
        # -------------------------------------------------

        st.subheader(
            "3️⃣ Hybrid Search Results"
        )

        for index, result in enumerate(
            hybrid_results,
            start=1,
        ):

            st.write(
                f"**Rank {index}: "
                f"{result['source']}** | "
                f"Hybrid Score: "
                f"{result['hybrid_score']:.4f}"
            )

            st.caption(
                result["text"]
            )

        # -------------------------------------------------
        # STEP 4: RERANKING
        # -------------------------------------------------

        st.subheader(
            "4️⃣ Reranked Results"
        )

        for index, result in enumerate(
            reranked_results,
            start=1,
        ):

            st.write(
                f"**Rank {index}: "
                f"{result['source']}** | "
                f"Rerank Score: "
                f"{result['rerank_score']:.4f}"
            )

            st.caption(
                result["text"]
            )

        # -------------------------------------------------
        # STEP 5: FINAL ANSWER
        # -------------------------------------------------

        st.subheader(
            "5️⃣ Final Answer"
        )

        with st.spinner(
            "Generating grounded answer..."
        ):

            answer = generate_answer(
                query,
                reranked_results,
            )

        if answer.strip():

            st.success(
                answer
            )

        else:

            st.warning(
                "I don't have enough reliable "
                "information to answer that from "
                "the provided documents."
            )
