import os
import logging
import streamlit as st
from dotenv import load_dotenv
import pickle

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, RecursiveRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.agent import ReActAgent
from chromadb import PersistentClient

logger = logging.getLogger(__name__)

@st.cache_resource
def setup_rag_system(debug=False):
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("groq", {}).get("api_key")
    if not groq_api_key:
        st.error("GROQ API key not found. Please check your environment variables or secrets.")
        st.stop()

    # LLM
    llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        max_input_tokens=1200,
        max_output_tokens=1200
    )

    # Embeddings
    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Persisted vector DBs
    persist_dirs = [
        "vectordb/case_2021",
        "vectordb/case_2022",
        "vectordb/case_2023",
        "vectordb/case_2024",
        "vectordb/case_2025"
    ]
    for persist_dir in persist_dirs:
        if not os.path.exists(persist_dir):
            st.error(f"Vector database directory {persist_dir} not found.")
            st.stop()

    # Build hybrid retrievers
    hybrid_retrievers = []
    for persist_dir in persist_dirs:
        # Load pickled nodes
        nodes_path = os.path.join(persist_dir, "nodes.pkl")
        if not os.path.exists(nodes_path):
            st.error(f"Pickle file {nodes_path} not found.")
            st.stop()

        with open(nodes_path, "rb") as f:
            nodes = pickle.load(f)

        # Vector store
        client = PersistentClient(path=persist_dir)
        collection = client.get_collection("case_collection")
        vector_store = ChromaVectorStore(chroma_collection=collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embedding_model)

        # Retrievers
        vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=2, retriever_mode="mmr")
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

        hybrid_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever, "bm25": bm25_retriever},
            verbose=True
        )
        hybrid_retrievers.append(hybrid_retriever)

    # Case metadata
    documents_info = [
        {
            "name": "Quezada2021_Retriever",
            "description": "Retrieves information from the United States Court of Appeals for the Armed Forces decision in United States v. Quezada (21-0089-MC), issued on December 20, 2021."
        },
        {
            "name": "Thompson2022_Retriever",
            "description": "Retrieves information from the United States Court of Appeals for the Armed Forces decision in United States v. Thompson (22-0098-AF), issued on November 21, 2022."
        },
        {
            "name": "Brown2023_Retriever",
            "description": "Retrieves information from the United States Court of Appeals for the Armed Forces decision in United States v. Brown (22-0249-CG), issued on October 23, 2023."
        },
        {
            "name": "Smith2024_Retriever",
            "description": "Retrieves information from the United States Court of Appeals for the Armed Forces decision in United States v. Smith (23-0207-AF), issued on November 26, 2024."
        },
        {
            "name": "Lopez2025_Retriever",
            "description": "Retrieves information from the United States Court of Appeals for the Armed Forces decision in United States v. Lopez (24-0226-CG), issued on September 2, 2025."
        },
    ]


    # Create retriever → tool
    def create_retriever_tool(retriever, llm, name, description):
        response_synthesizer = get_response_synthesizer(
            llm=llm, response_mode="compact", use_async=False
        )
        query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)
        return QueryEngineTool.from_defaults(query_engine=query_engine, name=name, description=description)

    retriever_tools = [
        create_retriever_tool(hybrid_retrievers[i], llm, info["name"], info["description"])
        for i, info in enumerate(documents_info)
    ]

    # System prompt
    system_prompt = """
        You are a highly specialized legal research assistant. 
        You may ONLY answer questions that are legal in nature. 
        This includes both:
        - Specific case law queries from the provided case documents (2021–2025).
        - General legal concepts, doctrines, or terminology.

        Before answering, always perform this intermediate reasoning step:

        1. Classify the user query:
        - If the query relates to law, legal concepts, legal systems, court rulings, rights, duties, contracts, procedures, or legal doctrines → classify as: LEGAL_QUERY.
        - If the query is casual conversation, mathematics, trivia, technical programming, or anything outside the legal domain → classify as: NON_LEGAL_QUERY.

        2. Response rules:
        - If LEGAL_QUERY:
            a) If the query references specific cases between 2021–2025, use the provided case documents to retrieve and answer. Cite the case name and year.
            b) If the query is a general legal question, answer concisely and professionally, using legal reasoning. Do NOT speculate beyond standard legal knowledge.
        - If NON_LEGAL_QUERY:
            Respond ONLY with: "I can only answer questions about legal cases (2021–2025) or general law queries."

        3. Examples:
        - LEGAL_QUERY (answer these):
            • "What is the difference between civil and criminal law?"
            • "Explain the principle of judicial review."
            • "Summarize the ruling in United States v. Lopez (2025)."
            • "What is mens rea in criminal law?"
        - NON_LEGAL_QUERY (reject these):
            • "What is 2+2?"
            • "Who won the FIFA World Cup in 2022?"
            • "Write me a Python script."
            • "Tell me a joke."

        4. Style & tone:
        - Be concise, professional, and clear.
        - Use citations ONLY when referring to case documents (case name + year).
        - Never provide speculative or non-legal answers.
        """


    # ReActAgent
    agent = ReActAgent(
        tools=retriever_tools,
        llm=llm,
        verbose=True,
        max_iterations=20,
        system_prompt=system_prompt
    )

    logger.info("RAG system setup complete.")

    if debug:
        return agent, llm, hybrid_retrievers

    return agent, llm
