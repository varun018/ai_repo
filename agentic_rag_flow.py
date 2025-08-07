#!/usr/bin/env python
# coding: utf-8

import warnings
import streamlit as st
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import os
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic
from IPython.display import Markdown
import pandas as pd
from io import BytesIO, StringIO
from fpdf import FPDF

# RAG Dependencies
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
import numpy as np
from pathlib import Path
from crewai_tools import tools


# CrewAI imports
import crewai
from crewai import Agent, Task, Crew, Process

load_dotenv()

# Initialize LLM and embeddings
api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0.1
)

embeddings = OpenAIEmbeddings(api_key=api_key)

# Global vector store variable
vector_store = None

class RAGKnowledgeBase:
    """Enhanced RAG Knowledge Base for incident resolution"""
    
    def __init__(self, embeddings_model):
        self.embeddings = embeddings_model
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        # CALL initialize_embeddings_safely HERE
        if embeddings is None:
            success = self.initialize_embeddings_safely()  # ← Called here
            if not success:
                raise Exception("Failed to initialize any embedding model")
        else:
            self.embeddings = embeddings
            
        # Initialize other attributes
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
def load_knowledge_base(self, kb_path="knowledge_base/"):
    
    """Load and process knowledge base documents"""
    try:
        if not os.path.exists(kb_path):
            os.makedirs(kb_path)
            return self.create_default_knowledge_base(kb_path)
                    
        # Load documents from directory
        loader = DirectoryLoader(
            kb_path, 
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
                    
        if not documents:
            return self.create_default_knowledge_base(kb_path)
                    
        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Fix for ONNX runtime error - Initialize embeddings properly
        try:
            # Option 1: Use CPU-only embeddings to avoid ONNX GPU issues
            if hasattr(self, 'embeddings') and self.embeddings is None:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},  # Force CPU usage
                    encode_kwargs={'normalize_embeddings': True}
                )
            
            # Option 2: Alternative - use a different embedding model
            # self.embeddings = HuggingFaceEmbeddings(
            #     model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
            #     model_kwargs={'device': 'cpu'}
            # )
            
            # Create vector store with error handling
            print(f"Creating vector store with {len(splits)} document chunks...")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
            print("Vector store created successfully")
            
        except Exception as embedding_error:
            print(f"ONNX/Embedding error: {embedding_error}")
            
            # Fallback: Try with a different embedding model
            #try:
                #print("Trying fallback embedding model...")
                #self.embeddings = HuggingFaceEmbeddings(
                    #model_name="sentence-transformers/paraphrase-MiniLM-L6-v2",
                    #model_kwargs={'device': 'cpu'}
                #)
                #self.vector_store = FAISS.from_documents(splits, self.embeddings)
                #print("Fallback embedding successful")
            #except Exception as fallback_error:
                #print(f"Fallback also failed: {fallback_error}")
                # Last resort: Use OpenAI embeddings if available
                #try:
                    #from langchain.embeddings import OpenAIEmbeddings
                    #self.embeddings = OpenAIEmbeddings()
                    #self.vector_store = FAISS.from_documents(splits, self.embeddings)
                    #print("Using OpenAI embeddings as final fallback")
                #except:
                    #return f"Critical error: Could not initialize any embedding model. ONNX runtime error: {embedding_error}"
                    
        # Create BM25 retriever for keyword search
        try:
            self.bm25_retriever = BM25Retriever.from_documents(splits)
                    
            # Create ensemble retriever combining semantic and keyword search
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[
                    self.vector_store.as_retriever(search_kwargs={"k": 3}),
                    self.bm25_retriever
                ],
                weights=[0.7, 0.3]
            )
        except Exception as retriever_error:
            print(f"Warning: Could not create BM25/Ensemble retriever: {retriever_error}")
            # Fall back to just FAISS retriever
            self.ensemble_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
                    
        return f"Loaded {len(documents)} documents with {len(splits)} chunks"
                
    except Exception as e:
        return f"Error loading knowledge base: {str(e)}"
    
# Alternative initialization method to prevent ONNX errors

    
def create_default_knowledge_base(self, kb_path):
    """Create a default knowledge base with common incident resolution patterns"""
    default_docs = [
        {
            "filename": "database_issues.txt",
            "content": """
DATABASE PERFORMANCE ISSUES - TROUBLESHOOTING GUIDE

Common Symptoms:
- Slow query response times
- Connection timeouts
- High CPU usage on database server
- Memory exhaustion
- Lock contention

Root Cause Analysis Steps:
1. Check active connections and connection pool status
2. Identify slow queries using performance monitoring
3. Analyze query execution plans
4. Review recent schema changes or deployments
5. Check database server resources (CPU, memory, disk I/O)

Resolution Strategies:
- Restart database connection pools
- Kill long-running queries
- Optimize problematic queries
- Scale database resources
- Implement query caching
- Review and optimize indexes

Historical Incidents:
- 2024-01-15: Connection pool exhaustion resolved by increasing max connections
- 2024-02-10: Slow queries fixed by adding composite index on user_orders table
- 2024-03-05: Memory leak in ORM resolved by upgrading to latest version
            """
        },
        {
            "filename": "network_security.txt", 
            "content": """
NETWORK SECURITY INCIDENTS - RESPONSE PROCEDURES

Common Attack Vectors:
- DDoS attacks
- Network intrusion attempts
- Data exfiltration
- Lateral movement
- Privilege escalation

Detection Indicators:
- Unusual network traffic patterns
- Failed authentication attempts
- Suspicious file access
- Anomalous user behavior
- Network segmentation violations

Immediate Response Actions:
1. Isolate affected systems
2. Preserve evidence and logs
3. Block suspicious IP addresses
4. Reset compromised credentials
5. Notify security team and stakeholders

Resolution Steps:
- Implement network segmentation rules
- Update firewall configurations
- Deploy additional monitoring
- Conduct security audit
- Update incident response procedures

Case Studies:
- 2024-01-20: Blocked APT group using firewall rules and threat intelligence
- 2024-02-25: Contained ransomware by network isolation and backup restoration
            """
        },
        {
            "filename": "application_errors.txt",
            "content": """
APPLICATION ERROR RESOLUTION GUIDE

Common Error Types:
- 500 Internal Server Error
- Memory leaks
- Resource exhaustion
- API timeout errors
- Configuration issues

Diagnostic Steps:
1. Check application logs for error patterns
2. Monitor resource usage (CPU, memory, disk)
3. Verify external service dependencies
4. Review recent code deployments
5. Test application functionality

Resolution Approaches:
- Restart application services
- Rollback recent deployments
- Scale application resources
- Fix configuration issues
- Apply hotfixes for critical bugs
- Implement circuit breakers for external calls

Resolution Examples:
- Memory leak: Restart services and schedule memory profiling
- API timeouts: Increase timeout values and implement retries
- Config errors: Revert to last known good configuration
- Dependency failures: Enable fallback mechanisms
            """
        }
    ]
    
    try:
        for doc in default_docs:
            file_path = os.path.join(kb_path, doc["filename"])
            with open(file_path, 'w') as f:
                f.write(doc["content"])
        
        # Now load the created documents
        return self.load_knowledge_base(kb_path)
        
    except Exception as e:
        return f"Error creating default knowledge base: {str(e)}"

def search(self, query, k=5):
    """Search the knowledge base for relevant information"""
    if not self.ensemble_retriever:
        return "Knowledge base not loaded. Please load the knowledge base first."
    
    try:
        docs = self.ensemble_retriever.get_relevant_documents(query)
        return docs[:k]
    except Exception as e:
        return f"Search error: {str(e)}"

# Initialize RAG system
rag_kb = RAGKnowledgeBase(embeddings)

from langchain.tools import tool  # Add this import at the top of your file

#def create_rag_tools(rag_knowledge_base):
    #"""Create RAG tools for the agents"""
@tool("Search Knowledge Base")
def search_knowledge_base(query: str) -> str:
    """Search the incident resolution knowledge base for relevant information"""
    docs = rag_kb.search(query, k=3)
    if isinstance(docs, str):  # Error case
        return docs
    
    if not docs:
        return "No relevant information found in knowledge base."
    
    context = "\n\n".join([f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs])
    return f"Relevant knowledge base information:\n{context}"

@tool("Search Similar Incidents")
def search_similar_incidents(incident_description: str) -> str:
    """Search for similar historical incidents and their resolutions"""
    docs = rag_kb.search(f"similar incident resolution: {incident_description}", k=2)
    if isinstance(docs, str):
        return docs
    
    if not docs:
        return "No similar incidents found in knowledge base."
    
    context = "\n\n".join([f"Historical Incident: {doc.page_content}" for doc in docs])
    return f"Similar incidents and resolutions:\n{context}"

Tool=[search_knowledge_base, search_similar_incidents]
    
    #return [
        #Tool(
            #name="search_knowledge_base",
            #description="Search the incident resolution knowledge base for troubleshooting guides, procedures, and technical documentation",
            #func=search_knowledge_base
        #),
        #Tool(
            #name="search_similar_incidents", 
            #description="Search for similar historical incidents and their resolution strategies",
            #func=search_similar_incidents
        #)
    #]

def load_input_section():
    st.title("🤖 Agentic RAG Incident Resolution System")
    
    # Knowledge base status
    st.sidebar.header("📚 Knowledge Base")
    if st.sidebar.button("Load/Refresh Knowledge Base"):
        with st.spinner("Loading knowledge base..."):
            result = rag_kb.load_knowledge_base()
            st.sidebar.success(result)
    
    # File uploader for adding documents
    uploaded_kb_file = st.sidebar.file_uploader(
        "Upload Knowledge Base Document:", 
        type=["txt", "pdf"], 
        help="Upload additional documents to enhance the knowledge base"
    )
    
    if uploaded_kb_file is not None:
        try:
            kb_path = "knowledge_base/"
            os.makedirs(kb_path, exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(kb_path, uploaded_kb_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_kb_file.read())
            
            # Reload knowledge base
            result = rag_kb.load_knowledge_base()
            st.sidebar.success(f"Added {uploaded_kb_file.name} to knowledge base!")
            
        except Exception as e:
            st.sidebar.error(f"Error adding file: {str(e)}")
    
    # Main input section
    uploaded_file = st.file_uploader("Upload a text file containing the Error Log:", type=["txt"], key="error_log_uploader")
    
    if uploaded_file is not None:
        topic = uploaded_file.read().decode("utf-8").strip()
        st.session_state["topic"] = topic
        with st.expander("Preview uploaded content"):
            st.text(topic[:500] + "..." if len(topic) > 500 else topic)
    else:
        topic = st.text_input("Enter the incident description:", value=st.session_state.get("topic", ""), key="topic_input")
        if topic:
            st.session_state["topic"] = topic
    
    return topic

def agent_task_creation(topic, api_key, llm, Tool):
    """Create agents with RAG capabilities"""
    
    # Enhanced Alert Detection Agent with RAG
    Alert_Detection_Agent = Agent(
        role=(
            f"RAG-Enhanced Alert Detection Specialist for {topic}. You have access to a comprehensive knowledge base "
            "of historical incidents, troubleshooting procedures, and resolution strategies. Use the search tools "
            "to find relevant information before making decisions."
        ),
        goal=(
            f"To provide intelligent alert processing with knowledge-base backed context for {topic}, "
            "minimizing MTTD through historical pattern recognition and documented procedures."
        ),
        backstory=(
            "You are an experienced incident response specialist with access to years of accumulated knowledge "
            "from previous incidents. You can search through documented procedures, similar cases, and "
            "proven resolution strategies to provide context-aware alert processing."
        ),
        allow_delegation=False,
        api_key=api_key,
        llm=llm,
        tools=Tool,
        verbose=True
    )

    # Enhanced Data Collector with RAG
    Data_Collector = Agent(
        role=(
            f"RAG-Powered Data Investigation Agent for {topic}. You leverage historical incident data "
            "and documented collection procedures to gather the most relevant operational intelligence."
        ),
        goal="To provide complete situational awareness by combining real-time data collection with historical context from the knowledge base",
        backstory=(
            "You are a forensic data analyst with access to a vast repository of incident investigation procedures. "
            "You know what data patterns to look for based on similar historical incidents and can prioritize "
            "data collection efforts using proven methodologies."
        ),
        allow_delegation=False,
        api_key=api_key,
        llm=llm,
        tools=Tool,
        verbose=True
    )

    # Enhanced Root Cause Analysis with RAG
    Root_Cause_Analysis = Agent(
        role=(
            "RAG-Enhanced Root Cause Analysis Expert with access to historical incident patterns, "
            "diagnostic procedures, and proven analysis methodologies from the knowledge base."
        ),
        goal="To accelerate root cause identification by combining current incident data with historical patterns and documented diagnostic approaches",
        backstory=(
            "You are a senior systems analyst with access to a comprehensive database of previous incidents, "
            "their root causes, and the analysis methods that led to successful resolution. You can pattern-match "
            "current symptoms with historical cases and apply proven diagnostic procedures."
        ),
        allow_delegation=False,
        llm=llm,
        tools=Tool,
        verbose=True
    )

    # Enhanced Action Execution with RAG
    Action_Execution = Agent(
        role=(
            "RAG-Powered Resolution Orchestrator with access to documented resolution procedures, "
            "success rates of different approaches, and risk assessments from historical incidents."
        ),
        goal="To execute optimal resolution strategies based on historical success patterns and documented procedures",
        backstory=(
            "You are a senior site reliability engineer with access to a comprehensive playbook of resolution "
            "strategies, their success rates, and risk assessments. You can choose the most appropriate "
            "resolution approach based on historical effectiveness and current system context."
        ),
        allow_delegation=False,
        llm=llm,
        tools=Tool,
        verbose=True
    )

    # Enhanced tasks with RAG instructions
    alert = Task(
        description=(
            "1. Search the knowledge base for similar alert patterns and documented procedures\n"
            "2. Process raw alerts using historical context and classification methods\n"
            "3. Group related alerts based on known patterns from the knowledge base\n"
            "4. Add business and technical context using documented incident impacts\n"
            "5. Assign priority based on historical severity and business impact data\n"
            "6. Determine response actions using proven procedures from the knowledge base\n"
        ),
        expected_output=(
            "A comprehensive alert analysis report containing: normalized alert data with RAG-enhanced context, "
            "correlation results based on historical patterns, business impact assessment using documented cases, "
            "priority assignment with knowledge-base justification, and recommended actions based on proven procedures."
        ),
        agent=Alert_Detection_Agent,
    )

    data_collector = Task(
        description=(
            "1. Search knowledge base for relevant data collection procedures for this incident type\n"
            "2. Gather logs and error patterns guided by historical incident analysis\n"
            "3. Collect metrics identified as critical in similar past incidents\n"
            "4. Capture configuration data based on documented investigation procedures\n"
            "5. Retrieve deployment information using proven forensic methods\n"
            "6. Structure data according to analysis templates from the knowledge base\n"
        ),
        expected_output=(
            "A structured data package enhanced with RAG insights containing: filtered logs with pattern analysis "
            "based on historical incidents, performance metrics aligned with documented investigation procedures, "
            "configuration data organized according to proven forensic templates, and deployment correlation "
            "using historical incident timelines."
        ),
        agent=Data_Collector,
    )

    Root_Cause_Analysis_Task = Task(
        description=(
            "1. Search knowledge base for similar incident patterns and their root causes\n"
            "2. Apply documented diagnostic procedures from historical successful analyses\n"
            "3. Compare current symptoms with known failure signatures from the knowledge base\n"
            "4. Use proven analysis methodologies to identify the most likely root cause\n"
            "5. Validate findings against historical incident resolutions and outcomes\n"
            "6. Provide confidence scoring based on similarity to documented cases\n"
        ),
        expected_output=(
            "A comprehensive RCA report in markdown format containing: identified root cause with confidence "
            "score based on historical pattern matching, supporting evidence using documented analysis methods, "
            "comparison with similar historical incidents and their resolutions, proven diagnostic reasoning chain, "
            "impact assessment based on documented similar incidents, and resolution recommendations with "
            "success probability based on historical outcomes."
        ),
        agent=Root_Cause_Analysis,
    )

    Action_Executor = Task(
        description=(
            "1. Search knowledge base for proven resolution strategies for identified root causes\n"
            "2. Select optimal resolution approach based on historical success rates\n"
            "3. Apply documented safety procedures and validation steps\n"
            "4. Execute resolution using proven methodologies from the knowledge base\n"
            "5. Validate success using documented verification procedures\n"
            "6. Document results for future knowledge base enhancement\n"
        ),
        expected_output=(
            "A detailed resolution report in markdown format containing: selected resolution strategy with "
            "RAG-based justification, execution steps based on documented procedures, success validation using "
            "proven verification methods, system health assessment according to documented metrics, "
            "comparison with similar historical resolution outcomes, lessons learned for knowledge base update, "
            "and incident closure confirmation with compliance audit trail."
        ),
        agent=Action_Execution,
    )

    crew = Crew(
        agents=[Alert_Detection_Agent, Data_Collector, Root_Cause_Analysis, Action_Execution],
        tasks=[alert, data_collector, Root_Cause_Analysis_Task, Action_Executor],
        verbose=True,
        process=Process.sequential
    )

    return crew

def topic_and_crew_creation():
    topic = load_input_section()
    
    # Ensure knowledge base is loaded
    if rag_kb.vector_store is None:
        with st.spinner("Initializing knowledge base..."):
            result = rag_kb.load_knowledge_base()
            st.success(result)
    
    # Create RAG tools
    #rag_tools = create_rag_tools(rag_kb)
    
    # Create crew with RAG capabilities
    crew = agent_task_creation(topic, api_key, llm, Tool)
    return topic, crew

def create_download_buttons(result):
    """Create download buttons for text and PDF formats."""
    result_str = result.serialize() if hasattr(result, 'serialize') else str(result)
    
    # Create text download
    text_buffer = StringIO()
    text_buffer.write(result_str)
    text_bytes = text_buffer.getvalue().encode('utf-8')
    
    st.download_button(
        label="📄 Download Report as Text",
        data=text_bytes,
        file_name="rag_incident_report.txt",
        mime="text/plain"
    )
    
    # Create PDF download
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        try:
            pdf.multi_cell(0, 10, result_str)
        except UnicodeEncodeError:
            result_str = result_str.encode('latin-1', errors='ignore').decode('latin-1')
            pdf.multi_cell(0, 10, result_str)
        
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        
        st.download_button(
            label="📑 Download Report as PDF",
            data=pdf_bytes,
            file_name="rag_incident_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")

def generate_response(topic, crew):
    """Generate RAG-enhanced incident resolution report"""
    try:
        with st.spinner("🔍 Analyzing incident with RAG intelligence..."):
            max_retries = 5
            result = None
            
            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for attempt in range(max_retries):
                try:
                    status_text.text(f"Processing with agents... (Attempt {attempt + 1}/{max_retries})")
                    progress_bar.progress((attempt + 1) * 20)
                    
                    result = crew.kickoff(inputs={"topic": topic})
                    progress_bar.progress(100)
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        status_text.text(f"Retrying... (Attempt {attempt + 2}/{max_retries})")
                        time.sleep(2 ** attempt)
                    else:
                        raise
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results with enhanced formatting
            st.markdown("## 🎯 RAG-Enhanced Incident Resolution Report")
            st.markdown("*Powered by intelligent knowledge retrieval and historical incident analysis*")
            st.markdown("---")
            
            # Format and display the result
            result_str = str(result)
            
            # Try to parse and format sections if possible
            if "Alert Detection" in result_str or "Root Cause" in result_str:
                sections = result_str.split("Agent:")
                for i, section in enumerate(sections[1:], 1):  # Skip first empty split
                    with st.expander(f"📋 Agent {i} Analysis", expanded=True):
                        st.markdown(section.strip())
            else:
                st.markdown(result_str)
            
            st.markdown("---")
            st.success("✅ RAG-enhanced analysis complete! The report includes insights from historical incidents and documented procedures.")
            
            # Store result
            st.session_state["latest_result"] = result
            st.session_state["report_generated"] = True
            
            # Create download buttons
            if result:
                st.markdown("### 💾 Download Options")
                create_download_buttons(result)
                
        return True
            
    except Exception as e:
        st.error(f"❌ Analysis failed: {str(e)}")
        st.info("💡 Try refreshing the knowledge base or simplifying the incident description.")
        return False

# Initialize session state
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

# Create the crew
topic, crew = topic_and_crew_creation()

# Main execution
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("🚀 Generate RAG-Enhanced Analysis", type="primary"):
        st.session_state.feedback = None
        generate_response(topic, crew)

with col2:
    if st.button("🔄 Reset System"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Feedback section
if st.session_state.get("report_generated"):
    st.markdown("---")
    st.markdown("### 💭 Feedback on RAG Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("👍 Excellent Analysis", key="like_button"):
            st.session_state.feedback = "Like"
            st.rerun()
            
    with col2:
        if st.button("👎 Needs Improvement", key="dislike_button"):
            st.session_state.feedback = "Dislike"
            st.rerun()
    
    with col3:
        if st.button("🔄 Regenerate with Different Context", key="regenerate_button"):
            st.session_state.feedback = None
            st.session_state.report_generated = False
            generate_response(topic, crew)
    
    # Show feedback messages
    if st.session_state.feedback == "Like":
        st.success("🎉 Thank you! Your feedback helps improve our RAG system.")
    elif st.session_state.feedback == "Dislike":
        st.warning("📝 Thank you for the feedback! Consider uploading additional knowledge base documents to improve future analyses.")

# Knowledge base statistics in sidebar
if rag_kb.vector_store:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Knowledge Base Stats")
    try:
        # Get vector store stats if available
        if hasattr(rag_kb.vector_store, 'index'):
            vector_count = rag_kb.vector_store.index.ntotal
            st.sidebar.metric("Document Chunks", vector_count)
        st.sidebar.info("💡 Upload more documents to improve analysis quality!")
    except:
        st.sidebar.info("Knowledge base loaded and ready!")