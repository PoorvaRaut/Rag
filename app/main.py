# main.py
import sys
import os

# Add the project's root directory to the Python path
# This ensures modules like 'core' and 'tools' can be imported
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import streamlit as st
import streamlit.components.v1 as components
import asyncio
import json
import logging
import yfinance as yf

from core.assistant import FinancialAssistant
from tools.ingestion import DataIngestionTool

def display_portfolio_chart(holdings, values):
    """Generates a bar chart for portfolio holdings using Chart.js."""
    chart_config = {
        "type": "bar",
        "data": {
            "labels": list(holdings.keys()),
            "datasets": [{
                "label": "Portfolio Holdings Value (₹)",
                "data": values,
                "backgroundColor": ["#4CAF50", "#2196F3", "#FFC107", "#FF5722"],
                "borderColor": ["#388E3C", "#1976D2", "#FFA000", "#D81B60"],
                "borderWidth": 1
            }]
        },
        "options": {
            "scales": {
                "y": {"beginAtZero": True, "title": {"display": True, "text": "Value (₹)"}},
                "x": {"title": {"display": True, "text": "Stocks"}}
            },
            "plugins": {"title": {"display": True, "text": "Portfolio Holdings"}}
        }
    }
    components.html(f"""
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <canvas id="portfolioChart" width="400" height="200"></canvas>
        <script>
            const ctx = document.getElementById('portfolioChart').getContext('2d');
            new Chart(ctx, {json.dumps(chart_config)});
        </script>
    """, height=300)

def main():
    """Main function to run the Streamlit application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('financial_assistant.log')
        ]
    )
    logger = logging.getLogger(__name__)

    st.set_page_config(
        page_title="Financial Assistant Demo - FAISS Edition",
        page_icon="💰",
        layout="wide"
    )

    st.markdown("""
        <style>
            .main-title { animation: fadeIn 1.5s ease-in-out; }
            @keyframes fadeIn { 0% { opacity: 0; transform: translateY(-20px); } 100% { opacity: 1; transform: translateY(0); } }
            .logo-text-container { display: flex; align-items: center; margin-bottom: 20px; }
            .logo-text { font-size: 24px; font-weight: bold; color: #4CAF50; margin-left: 10px; animation: textPop 1s ease-in-out; }
            @keyframes textPop { 0% { opacity: 0; transform: scale(0.8); } 100% { opacity: 1; transform: scale(1); } }
            .sidebar .sidebar-content > div { animation: slideIn 0.5s ease-in-out; }
            @keyframes slideIn { 0% { opacity: 0; transform: translateX(-20px); } 100% { transform: translateX(0); } }
            .stChatMessage { animation: messagePop 0.5s ease-in-out; }
            @keyframes messagePop { 0% { opacity: 0; transform: scale(0.9); } 100% { opacity: 1; transform: scale(1); } }
            .large-toggle-container {
                margin-top: -20px;
                padding: 10px;
                background-color: #333;
                border-radius: 8px;
            }
            .large-toggle-container label {
                font-size: 1.25rem;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="large-toggle-container">', unsafe_allow_html=True)
        enable_llm_fallback = st.toggle(
            "Enable LLM Fallback",
            value=True,
            help="If enabled, the assistant will use the LLM (e.g., GPT-3.5) if the answer is not found in the ingested data."
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.enable_llm_fallback = enable_llm_fallback

        st.markdown('<div class="logo-text-container">', unsafe_allow_html=True)
        st.markdown('<span class="logo-text">Infinity Pool</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.header("⚙️ Configuration")
        user_id = st.text_input(
            "User ID",
            value="demo_user_123",
            help="Unique identifier for user sessions"
        )
        
        if 'assistant' in st.session_state:
            st.markdown("### 📊 FAISS Memory Stats")
            try:
                stats = st.session_state.assistant.get_memory_stats()
                st.metric("Total Vectors", stats["total_vectors"])
                st.metric("Metadata Entries", stats["total_metadata"])
                st.metric("Vector Dimension", stats["dimension"])
                st.text(f"Index Type: {stats['index_type']}")
            except Exception as e:
                st.error(f"Error loading stats: {str(e)}")
        
        st.markdown("---")
        st.markdown("### 📄 Data Ingestion")
        st.info("Ingest data from various sources to enable RAG.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Google Drive")
            if st.button("Google Docs"):
                st.info(st.session_state.ingestion_tool.ingest_from_gdrive("docs"))
            if st.button("Google Slides"):
                st.info(st.session_state.ingestion_tool.ingest_from_gdrive("slides"))
        with col2:
            st.subheader("Link")
            website_url = st.text_input("Website URL", key="website_url")
            if st.button("Process Website"):
                with st.spinner("Ingesting website..."):
                    message = st.session_state.ingestion_tool.ingest_from_website(website_url)
                    st.success(message)
            youtube_url = st.text_input("YouTube URL", key="youtube_url")
            if st.button("Process YouTube"):
                with st.spinner("Ingesting YouTube video..."):
                    message = st.session_state.ingestion_tool.ingest_from_youtube(youtube_url)
                    st.success(message)

        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Upload a PDF for RAG", type=["pdf"])
        if uploaded_file:
            message = st.session_state.ingestion_tool.ingest_from_pdf(uploaded_file)
            st.success(message)
        
        st.subheader("Paste Text")
        pasted_text = st.text_area("Paste your text here", height=150, key="pasted_text_area")
        if st.button("Process Copied Text"):
            if pasted_text:
                message = st.session_state.ingestion_tool.ingest_from_text(pasted_text)
                st.success(message)
            else:
                st.warning("Please paste some text to process.")


    st.markdown('<h1 class="main-title">🤖 Financial Assistant Demo - FAISS Edition</h1>', unsafe_allow_html=True)
    st.subheader("Powered by LangChain + LangGraph + LlamaIndex + FAISS")

    if 'assistant' not in st.session_state:
        try:
            logger.info("Initializing Financial Assistant with FAISS")
            if 'ingestion_tool' not in st.session_state:
                st.session_state.ingestion_tool = DataIngestionTool()
            with st.spinner("Initializing FAISS vector database..."):
                st.session_state.assistant = FinancialAssistant(st.session_state.ingestion_tool)
            st.success("✅ Financial Assistant with FAISS ready!")
        except Exception as e:
            logger.error(f"Error initializing assistant: {str(e)}", exc_info=True)
            st.error(f"❌ Error initializing assistant: {str(e)}")
            return

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Ask me about stocks, portfolio, SIPs, or document content..."):
        logger.info(f"Processing user prompt: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Processing your request with FAISS memory..."):
                try:
                    initial_state = {
                        "messages": [prompt],
                        "user_id": user_id,
                        "context": {"enable_llm_fallback": st.session_state.enable_llm_fallback},
                        "tool_results": {},
                        "final_response": ""
                    }
                    response_state = asyncio.run(
                        st.session_state.assistant.workflow_graph.app.ainvoke(initial_state)
                    )
                    response = response_state.get("final_response", "No response generated.")

                    logger.debug(f"Assistant response: {response}")
                    st.write(response)

                    if "portfolio" in response.lower() or "added" in response.lower():
                        portfolio = st.session_state.assistant.db_manager.get_portfolio(user_id)
                        if portfolio:
                            logger.debug(f"Portfolio for chart: {portfolio.holdings}")
                            values = []
                            valid_holdings = {}
                            for symbol, quantity in portfolio.holdings.items():
                                try:
                                    stock = yf.Ticker(symbol)
                                    hist = stock.history(period="1d")
                                    if not hist.empty:
                                        value = hist['Close'].iloc[-1] * quantity
                                        values.append(value)
                                        valid_holdings[symbol] = quantity
                                        logger.debug(f"Chart data for {symbol}: value={value}")
                                    else:
                                        values.append(0.0)
                                        logger.warning(f"No data for {symbol}")
                                        st.warning(f"No data for {symbol}")
                                except Exception as e:
                                    values.append(0.0)
                                    logger.error(f"Error fetching data for {symbol}: {str(e)}")
                                    st.warning(f"Error fetching data for {symbol}: {str(e)}")
                            if any(v > 0 for v in values):
                                display_portfolio_chart(valid_holdings, values)
                            else:
                                logger.error("No valid stock data available for chart")
                                st.error("No valid stock data available for chart.")
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"I encountered an error: {str(e)}. Please try again."
                    logger.error(f"Error processing prompt: {error_msg}", exc_info=True)
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()