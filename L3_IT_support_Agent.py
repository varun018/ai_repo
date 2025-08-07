#!/usr/bin/env python
# coding: utf-8

# In[1]:


import onnxruntime as ort
print(ort.get_device())


# In[2]:


import warnings
import streamlit as st
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatAnthropic
from langchain_openai import ChatOpenAI
from IPython.display import Markdown
import pandas as pd
from io import BytesIO, StringIO
from fpdf import FPDF


# In[3]:


import crewai
from crewai import Agent, Task, Crew
from crewai_tools import RagTool ,SerperDevTool

# In[4]:

api_key=os.getenv("OPENAI_API_KEY")
#print(api_key)


# In[5]:

serper_api_key=os.getenv("SERPER_API_KEY")

load_dotenv()

# In[6]:


#claude_api_key=os.getenv("ANTHROPIC_API_KEY")

# In[7]:

#import requests
#response = requests.get("https://api.anthropic.com/v1/messages", 
                       #headers={"x-api-key": "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"})

config = {
            "chunker": {
                "chunk_size": 200,  # Smaller chunks
                "chunk_overlap": 50,
                "length_function": "len"
            },
            "vectordb": {
                "provider": "chroma",
                "config": {
                    "collection_name": "my_docs",
                    "dir": "./chroma_db",
                    "allow_reset": True
                }
            }
        }


# In[13]:


rag_tool = RagTool(config=config)


# In[14]:


#creating the Knowledge Base

rag_tool.add("D:\\Agentic_AI\\knowledge_base\\application_errors.txt")

rag_tool.add("D:\\Agentic_AI\\knowledge_base\\network_security.txt")

rag_tool.add("D:\\Agentic_AI\\knowledge_base\\database_security.txt")


# In[15]:


result = rag_tool._run("test query")
print(result)


# In[16]:


web_search_tool = SerperDevTool(api_key=serper_api_key)


# In[17]:


llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key,
    temperature=0.1
)

def load_input_section():
    st.title("Agentic AI based resolution system")
    
    uploaded_file = st.file_uploader("Upload a text file containing the Error Log:", type=["txt"], key="error_log_uploader")
    
    if uploaded_file is not None:
        topic = uploaded_file.read().decode("utf-8").strip()
        st.session_state["topic"] = topic
        # Show preview
        with st.expander("Preview uploaded content"):
            st.text(topic[:500] + "..." if len(topic) > 500 else topic)
    else:
        topic = st.text_input("Enter the topic:", value=st.session_state.get("topic", ""), key="topic_input")
        if topic:
            st.session_state["topic"] = topic
    
    return topic

def agent_task_creation(topic, api_key, llm):

    Alert_Detection_Agent = Agent(
        role=(
            f"The Alert Detection Agent serves as the critical first line of defense in the incident response pipeline for the issue {topic}, "
            "acting as an intelligent triage system that receives, processes, and categorizes incoming alerts from various monitoring platforms. "
            "This agent transforms raw alert data into actionable intelligence, ensuring that the right information reaches the right teams at the right time."
        ),
        goal=(
            f"To minimize Mean Time to Detection (MTTD) and reduce alert fatigue by providing intelligent, "
            f"context-aware alert processing that enables rapid and accurate incident response for {topic}."
        ),
        backstory=(
            f"The Alert Detection Agent emerged from a critical Black Friday 2023 outage where thousands of alerts from multiple monitoring systems buried the engineering team in noise. "
            f"What should have been a quick fix for {topic} took four hours to identify due to alert chaos. "
            "The post-mortem revealed the need for an intelligent system that could distinguish symptoms from root causes, understand system relationships, learn from incidents, and provide consistent alert handling regardless of engineer experience or time of day."
        ),
        allow_delegation=False,
        api_key=api_key,  # Ensure api_key is a valid string
        llm = llm,  # Use a valid LLM model name
        verbose=True
    )
    
    Data_Collector = Agent(
    role=f"The Data Collector Agent operates as the investigative backbone of the incident response system, functioning as an intelligent data harvester that rapidly assembles comprehensive context around {topic}."
    "This agent transforms scattered information across multiple systems into a unified intelligence picture, enabling informed decision-making during critical incidents.",
    goal="To provide complete situational awareness by autonomously gathering, correlating, and delivering relevant operational data within minutes of an incident trigger",
    backstory=f"The Data Collector Agent serves as the investigative arm of incident response, automatically gathering critical evidence when {topic} arise." 
    "It rapidly pulls system logs, performance metrics, configuration snapshots, and recent deployment data to build a comprehensive picture of system state." 
    "This enables faster root cause analysis by providing engineers with all relevant context upfront, eliminating time-consuming manual data hunting during critical incidents."
    "The post-mortem revealed the need for an intelligent system that could distinguish symptoms from root causes, understand system relationships, learn from incidents, and provide consistent alert handling regardless of engineer experience or time of day.",
    allow_delegation=False,
    api_key=api_key,
    llm= llm,
    verbose=True
    )
    
    Root_Cause_Analysis=Agent(
        role="The Root Cause Analysis Agent serves as the analytical core of incident response, transforming operational data into actionable insights." 
            "Acting as a digital detective, it employs sophisticated reasoning to identify root causes and distinguish correlation from causation in complex systems."
            "The agent uses the local knowledge base as a context to find the RCA using LLM.",
        goal="To accelerate incident resolution by accurately identifying root causes through intelligent analysis, reducing Mean Time to Resolution (MTTR) from hours to minutes.",
        backstory="The Root Cause Analysis Agent serves as the diagnostic brain that analyzes collected data to identify true underlying causes."
                "It performs log pattern analysis, dependency graph traversal, anomaly detection, and historical incident comparison. Using reasoning and memory, it correlates symptoms with known issues, distinguishes causes from symptoms, and prevents recurring incidents through systematic analysis.",
        allow_delegation=False,
        tools=[rag_tool],
        llm = llm,
        verbose=True          

    )
    
    Root_Cause_Analysis_checker = Agent(
        role="RCA Confidence Evaluator",
        goal="Analyze the confidence score of Root Cause Analysis results and trigger web search if confidence is below 0.8 threshold",
        backstory="""You are a quality assurance specialist who meticulously evaluates the 
        confidence level of root cause analysis findings. Your primary responsibility is to 
        assess whether the RCA agent's analysis meets the minimum confidence threshold of 0.8. 
        When the confidence score falls below this benchmark, you immediately escalate the case 
        to the web search agent for additional research and validation. You serve as the critical 
        checkpoint ensuring only high-confidence root cause analyses proceed without additional 
        verification.""",
        tools=[],
        llm=llm,
        verbose=True
    )
    
    Web_Search_Agent =Agent(
        role="Web Search Agent.",
        goal="Search the web for additional information if the Root Cause Analysis Agent has not provided a relevant RCA.",
        backstory="The Web Search Agent acts as a digital librarian, scouring the internet for relevant information to supplement the Root Cause Analysis Agent's findings.",
        tools=[web_search_tool],
        llm = llm,
        verbose=True  
    )
    
    Action_Execution = Agent(
        role="Final Output Synthesizer and Resolution Orchestrator",

        goal="To consolidate findings from Root Cause Analysis or Web Search agents into a comprehensive final output, then execute optimal resolution strategies with appropriate risk management and clear action plans.",

        backstory="""You are the final decision-making authority in the incident response pipeline, 
        functioning as a senior site reliability engineer with expertise in synthesizing complex 
        technical information into actionable insights. Your primary responsibility is to receive 
        outputs from either the Root Cause Analysis agent (when confidence >= 0.8) or the Web 
        Search agent (when additional research was required), then transform these findings into 
        a unified, comprehensive final report.

        You excel at consolidating multiple data sources, identifying the most critical information, 
        and presenting it in a structured format that includes root causes, recommended actions, 
        risk assessments, and implementation timelines. Your output serves as the definitive 
        incident resolution document that stakeholders rely on for decision-making.

        Beyond synthesis, you operate with configurable autonomy levels - immediately flagging 
        safe remediation steps while highlighting high-risk actions that require human approval. 
        You eliminate analysis paralysis by providing clear, prioritized action items with 
        associated risk levels and expected outcomes.""",

        tools=[],
        allow_delegation=False,
        llm=llm,
        verbose=True
    )
    
    alert= Task(
        description=(
            "1.Process raw alerts from multiple sources\n"
            "2.Group related alerts and reduce noise\n"
            "3.Add business and Technical Context\n"
            "4.Assign priority intelligently\n"
            "5.Determine appropriate response action\n"
        ),
        expected_output="The Alert Detection Agent processes incoming alerts and outputs a classified, enriched alert package containing: normalized alert data with standardized severity (P0-P4), correlation results linking related alerts, business context including impact assessment, routing decisions with assigned teams, and trigger signals for the Data Collector Agent to begin investigation.",
        agent=Alert_Detection_Agent,
    )
    
    data_collector=Task(
        description=(
            "1.Gather Filtered Logs and Error Patterns\n"
            "2.Collect System and application metrics\n"
            "3.Capture recent configs and changes\n"
            "4.Retrieve recent deployment activities\n"
            "5.Unify all data for analysis\n"
        ),
        expected_output="The Data Collector Agent outputs a unified package containing structured logs with error patterns, time-series performance metrics, configuration snapshots with change differentials, deployment timeline with correlation markers, and metadata quality indicators."
                        "All data is temporally aligned and cross-referenced for seamless consumption by the Root Cause Analysis Agent.",
        agent=Data_Collector,
    )
    
    Root_Cause_Analysis_Task=Task(
        description=(
            "1.Identify critical error signatures and Anamoly sequences\n"
            "2.Map failure propogation through service relationship\n"
            "3.Perform Statistical Analysis of Performance Deviation\n"
            "4.Match current symptoms with past incidents\n"
            "5.Integrate all findings into confident root cause determination\n"
        ),
        expected_output="The RCA report output should be a structured report written in markdown format containing: identified root cause with confidence score (0-1), supporting evidence from multiple analysis techniques, ranked alternative hypotheses, causal reasoning chain linking symptoms to causes, impacted service dependencies, similar historical incidents with resolution outcomes, and actionable recommendations for resolution strategy with risk assessments. ",
        agent=Root_Cause_Analysis,
    )
    
    Root_Cause_Analysis_Task_checker=Task(
        description=(
            "1.Check if the Root Cause Analysis Agent has a confidence score of greater than 0.8\n"
            "2.If not, request WEB SEARCH NEEDED\n"
        ),
        expected_output="The Root Cause Analysis Checker outputs a confirmation of whether the Root Cause Analysis Agent has provided a relevant RCA. If not, it triggers the Web Search Agent to gather additional information.",
        agent=Root_Cause_Analysis_checker
    )
    
    Web_Search_Task=Task(
        description=(
            "1.Search the web for additional information if the Root Cause Analysis Agent has a confidence score of less than 0.8\n"
        ),
        expected_output="The Web Search Agent outputs relevant web search results that supplement the Root Cause Analysis Agent's findings, providing additional context or information that may lead to a more accurate root cause analysis.",
        agent=Web_Search_Agent
    )
    
    Action_Executor=Task(
        description="""Analyze the root cause findings provided by either the Root Cause Analysis agent 
        or Web Search agent and create a comprehensive incident resolution report""",
        expected_output="## Root Cause Analysis Summary"
                        "- **Primary Root Cause**: Detailed explanation of the underlying issue that triggered the incident"
                        "- **Contributing Factors**: Secondary factors that amplified or enabled the primary cause"
                        "- **Root Cause Category**: Classification (human error, system failure, process gap, external dependency, etc.)"
                        "- **Impact Chain Analysis**: How the root cause propagated through the system to create the observed symptoms "
                        "## Resolution Implementation"
                        "- **Fix Applied**: Detailed description of the specific solution implemented to address the root cause"
                        "- **Implementation Approach**: Step-by-step methodology used to apply the fix"
                        "- **Validation Results**: Evidence that the fix successfully resolved the root cause"
                        "- **Rollback Plan**: Contingency measures available if the fix proves inadequate"
                        "## Future Prevention Strategy"
                        "- **Immediate Preventive Measures**: Quick wins to prevent recurrence in the short term"
                        "- **Long-term Systemic Improvements**: Architectural or process changes to eliminate the root cause class"
                        "- **Monitoring Enhancements**: New alerts, dashboards, or observability measures to detect similar issues early"
                        "- **Process Improvements**: Updates to procedures, runbooks, or operational practices"
                        "- **Team Training Requirements**: Knowledge gaps identified and training recommendations"
                        "## Risk Mitigation Framework"
                        "- **Similar Risk Patterns**: Other areas in the system susceptible to the same type of failure"
                        "- **Detection Mechanisms**: Early warning systems to catch similar issues before they become incidents"
                        "- **Response Optimization**: Improvements to incident response based on lessons learned"
                        "- **Compliance Considerations**: Regulatory or security implications and required documentation"
                        "## Lessons Learned Documentation"
                        "- **Knowledge Base Updates**: New information to be added to team knowledge repositories"
                        "- **Runbook Modifications**: Updates needed to operational procedures"
                        "- **Post-Incident Review Items**: Key discussion points for team retrospectives"
                        "- **Success Criteria for Prevention**: Metrics to measure effectiveness of preventive measures",
                        #"## Modified Code"
                        #"- **Modified Code**:Generate the modified code for the issue",
        agent=Action_Execution,
    )
    
    crew = Crew(
    agents=[Alert_Detection_Agent, Data_Collector, Root_Cause_Analysis,Root_Cause_Analysis_checker,Web_Search_Agent,Action_Execution],
    tasks=[alert, data_collector, Root_Cause_Analysis_Task,Root_Cause_Analysis_Task_checker,Web_Search_Task,Action_Executor],
    verbose=True
)
    return crew

# In[51]:

def topic_and_crew_creation():
    topic = load_input_section()
    crew = agent_task_creation(topic, api_key, llm)
    return topic, crew


topic, crew=topic_and_crew_creation()

def create_download_buttons(result):
    """Create download buttons for text and PDF formats."""
    # Convert result to string
    result_str = result.serialize() if hasattr(result, 'serialize') else str(result)
    
    # Create text download
    text_buffer = StringIO()
    text_buffer.write(result_str)
    text_bytes = text_buffer.getvalue().encode('utf-8')
    
    st.download_button(
        label="Download Report as Text",
        data=text_bytes,
        file_name="incident_report.txt",
        mime="text/plain"
    )
    
    # Create PDF download
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Handle potential encoding issues
        try:
            pdf.multi_cell(0, 10, result_str)
        except UnicodeEncodeError:
            # Fallback for characters that can't be encoded
            result_str = result_str.encode('latin-1', errors='ignore').decode('latin-1')
            pdf.multi_cell(0, 10, result_str)
        
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        
        st.download_button(
            label="Download Report as PDF",
            data=pdf_bytes,
            file_name="incident_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        
def generate_response(topic,crew):
    """Generate incident resolution report with retry logic and download options."""
    try:
        with st.spinner("Analyzing incident..."):
            max_retries = 5
            result = None
            
            # Retry logic with exponential backoff
            for attempt in range(max_retries):
                try:
                    result = crew.kickoff(inputs={"topic": topic})
                    break  # Exit loop if successful
                except Exception as e:
                    print(f"Attempt failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                       raise
            
            # Display the result
            st.markdown("### Incident Resolution Report:")
            st.markdown("---")
            st.markdown(f"{result}")
            st.markdown("---")
            st.success("Analysis complete! Review the report above.")
            
            # Store result in session state
            st.session_state["latest_result"] = result
            st.session_state["report_generated"] = True
            
            # Create download buttons
            if result:
                create_download_buttons(result)
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False
    
    return True
        
# Initialize session state
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

# Main execution logic
if st.button("Generate Response"):
    # Reset feedback when generating new response
    st.session_state.feedback = None
    #topic, crew = topic_and_crew_creation()
    generate_response(topic,crew)

# Show feedback section only if report has been generated
if st.session_state.get("report_generated"):
    st.markdown("### Feedback:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👍 Like", key="like_button"):
            st.session_state.feedback = "Like"
            st.rerun()
            
    with col2:
        if st.button("👎 Dislike", key="dislike_button"):
            st.session_state.feedback = "Dislike"
            st.rerun()
    
    # Show feedback messages and regenerate option
    if st.session_state.feedback == "Like":
        st.success("Thank you for your feedback!")
    elif st.session_state.feedback == "Dislike":
        st.warning("Thank you for your feedback!")
        if st.button("Regenerate Response", key="regenerate_button"):
            # Clear feedback and regenerate
            st.session_state.feedback = None
            st.session_state.report_generated = False
            if "latest_result" in st.session_state:
                del st.session_state["latest_result"]
            # Regenerate the response
            st.rerun()
            topic, crew = topic_and_crew_creation()
            generate_response(topic,crew)
            #topic = load_input_section()

