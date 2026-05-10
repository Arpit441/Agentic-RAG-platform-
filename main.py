import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment variables (like GROQ_API_KEY)
load_dotenv()

# Set page config before any other Streamlit commands
st.set_page_config(page_title="Enterprise RAG Copilot", layout="wide")

from rag.orchestrator import RAGOrchestrator
from rag.ingest import IngestionPipeline

# Initialize the orchestrator in session state so we don't rebuild models on every render
if "orchestrator" not in st.session_state:
    with st.spinner("Loading AI Models into Memory..."):
        try:
            st.session_state.orchestrator = RAGOrchestrator()
        except Exception as e:
            st.error(f"Failed to initialize models. Please check API keys. Error: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Data Ingestion & Settings ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("Place `.txt` files in the `/data` folder, then index them below.")
    
    if st.button("🔄 Run Data Ingestion"):
        with st.spinner("Building FAISS & BM25 Indexes..."):
            pipeline = IngestionPipeline()
            try:
                pipeline.ingest_documents("data")
                st.success("✅ Indexes built successfully!")
                # Force orchestrator to reload indexes
                st.session_state.orchestrator = RAGOrchestrator()
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
                
    st.markdown("---")
    st.markdown("**Architecture Stack:**\n- UI: Streamlit (Streaming)\n- Memory: Query Decontextualization\n- LLM: Groq (Llama 3.1)\n- Embeddings: HuggingFace\n- Store: FAISS + BM25\n- Reranker: Cross-Encoder")

# --- Tabs ---
tab1, tab2 = st.tabs(["💬 Copilot Chat", "📊 Admin Dashboard"])

# --- Tab 1: Chat Interface ---
with tab1:
    st.title("🏢 Enterprise AI Copilot")
    st.markdown("Ask questions about company policies. The AI is strictly guarded against hallucinations.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "metrics" in msg:
                with st.expander("🔍 View Telemetry & Guardrails"):
                    st.json({
                        "Debug": msg.get("debug", {}),
                        "Verification": msg.get("verification", {}),
                        "Metrics": msg.get("metrics", {})
                    })

    # Chat Input
    if prompt := st.chat_input("E.g., What is the equipment stipend for remote workers?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if "orchestrator" in st.session_state:
                try:
                    # Execute streaming pipeline
                    stream = st.session_state.orchestrator.run_stream(
                        query=prompt, 
                        chat_history=st.session_state.messages
                    )
                    
                    # Streamlit natively handles generators via write_stream
                    final_answer = st.write_stream(stream)
                    
                    # Grab the telemetry that was saved to the orchestrator after the stream
                    run_data = st.session_state.orchestrator.last_run_data
                    
                    if run_data:
                        verification = run_data["verification_result"]
                        metrics = run_data["metrics"]
                        debug_info = {
                            "standalone_query": run_data["standalone_query"],
                            "retrieved_chunks_count": len(run_data["retrieved_docs"]),
                            "reranked_chunks_count": len(run_data["reranked_docs"]),
                            "compressed_context_length": len(run_data["compressed_context"])
                        }
                        
                        # Highlight if hallucination guardrail was tripped
                        if not verification.get("faithful", True):
                            st.warning("⚠️ **Guardrail Alert**: The verifier detected that parts of this response might be unsupported by the retrieved context.")
                        
                        with st.expander("🔍 View Telemetry & Guardrails"):
                            st.json({
                                "Debug": debug_info,
                                "Verification": verification,
                                "Metrics": metrics
                            })
                            
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": final_answer,
                            "debug": debug_info,
                            "verification": verification,
                            "metrics": metrics
                        })
                    
                except Exception as e:
                    st.error(f"Error during execution: {e}")
            else:
                st.error("Orchestrator not initialized.")

# --- Tab 2: Admin Dashboard ---
with tab2:
    st.title("📊 Pipeline Evaluation & Telemetry")
    
    if "orchestrator" in st.session_state:
        # Fetch stats from SQLite
        stats = st.session_state.orchestrator.feedback_manager.get_dashboard_stats()
        recommendation = st.session_state.orchestrator.feedback_manager.adjust_chunk_size_recommendation()
        
        # Display high-level metrics in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries Processed", stats["total_queries"])
        with col2:
            st.metric("Hallucination Rate", f"{stats['hallucination_percentage']}%")
        with col3:
            st.metric("Avg Retrieval Score", stats["avg_retrieval_score"])
            
        st.markdown("---")
        
        # Display Auto-Tuning Recommendations
        st.subheader("🤖 AI Auto-Tuning Recommendation")
        st.info(recommendation)
        
        st.markdown("---")
        
        # Display the failed query log
        st.subheader("⚠️ Hallucinated / Poor Performing Queries Log")
        bad_queries = st.session_state.orchestrator.feedback_manager.get_low_performing_queries()
        if bad_queries:
            df = pd.DataFrame(bad_queries)
            # Hide the id and user_rating for cleaner view
            df = df.drop(columns=["id", "user_rating"], errors="ignore")
            st.dataframe(df, use_container_width=True)
        else:
            st.success("No hallucinations or poor performing queries detected yet!")
            
        col_refresh, col_clear = st.columns([1, 1])
        with col_refresh:
            if st.button("🔄 Refresh Dashboard"):
                st.rerun()
        with col_clear:
            if st.button("🗑️ Clear Telemetry Database", type="secondary"):
                st.session_state.orchestrator.feedback_manager.clear_all_feedback()
                st.success("Database cleared! Ask new questions to rebuild clean stats.")
                st.rerun()
    else:
        st.error("Orchestrator not initialized. Please run ingestion first.")
