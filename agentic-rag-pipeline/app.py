import streamlit as st
from pipeline.agent import AgenticRAGAgent
from pipeline.langgraph_agent import create_multiagent_graph
from schemas.models import AgentResponse

# Page configuration
st.set_page_config(page_title="Agentic RAG Pipeline", page_icon="🤖", layout="wide")
st.title("🤖 Agentic RAG Pipeline")
st.markdown("Interact with the Agentic RAG Pipeline using Classic ReAct or Multi-Agent LangGraph.")

@st.cache_resource
def get_classic_agent():
    return AgenticRAGAgent()

@st.cache_resource
def get_langgraph_app():
    return create_multiagent_graph()

# Initialize agents
with st.spinner("Initializing agents and models..."):
    classic_agent = get_classic_agent()
    langgraph_app = get_langgraph_app()

query = st.text_input("Enter your query:", placeholder="What tasks are involved in building a REST API?")

tab1, tab2 = st.tabs(["Agentic RAG (Classic)", "Multi-Agent LangGraph"])

def render_response(resp, source="Classic Agent"):
    st.subheader("RAG Answer")
    st.write(resp.answer)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Task Classification")
        st.write(f"**Category:** {resp.task.category.value}")
        st.write(f"**Confidence:** {resp.task.confidence * 100:.0f}%")
        st.write(f"**Rationale:** {resp.task.rationale}")
        
    with col2:
        st.subheader("Sources")
        if resp.sources:
            for s in resp.sources:
                st.markdown(f"- `{s}`")
        else:
            st.write("No sources used.")
            
    st.subheader("Workflow Plan")
    for step in resp.workflow.steps:
        deps = resp.workflow.dependencies.get(str(step.id), [])
        st.markdown(f"**Step {step.id}:** {step.description} (Complexity: {step.estimated_complexity}, Depends on: {deps})")

# Tab 1: Classic Agentic RAG
with tab1:
    if st.button("Run Classic RAG", key="btn_classic") and query:
        with st.spinner("Running Classic Agentic RAG..."):
            resp = classic_agent.run(query)
            render_response(resp)

# Tab 2: Multi-Agent LangGraph
with tab2:
    if st.button("Run Multi-Agent LangGraph", key="btn_langgraph") and query:
        st.markdown("### Multi-Agent Routing Status")
        status_container = st.empty()
        
        # We use graph.stream to capture intermediate steps and visualize routing decisions
        initial_state = {
            "query": query, 
            "messages": [], 
            "task_context": "", 
            "task": None, 
            "workflow_plan": None, 
            "final_answer": "", 
            "next_step": ""
        }
        
        current_state = initial_state.copy()
        routing_logs = []
        for step in langgraph_app.stream(initial_state):
            # step is a dict containing the node name as key and the state updates as value
            for node_name, state_update in step.items():
                current_state.update(state_update)
                if "messages" in state_update and state_update["messages"]:
                    last_msg = state_update["messages"][-1].content
                    routing_logs.append(f"**{node_name.capitalize()}**: {last_msg}")
                    status_container.info("\n\n".join(routing_logs))
            
        st.success("Workflow Complete!")
        
        # Package into AgentResponse for consistent rendering
        resp = AgentResponse(
            answer=current_state.get("final_answer", "No answer generated."),
            task=current_state.get("task"),
            workflow=current_state.get("workflow_plan"),
            sources=[], # LangGraph pipeline can be updated to return sources, skipping for now
            mode="rag"
        )
        
        render_response(resp, source="Multi-Agent LangGraph")
