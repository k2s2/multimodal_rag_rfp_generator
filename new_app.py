__import__("pysqlite3")
import sys
sys.modules["sqlite3"]=sys.modules.pop("pysqlite3")

import streamlit as st
import torch
from pathlib import Path
from new_chatbot_rfp import (
    setup_rag_vectorstore, 
    create_combined_agent_executor,
    extract_all_document_content,
    get_database_summary,
    generate_rfp,
    remove_think_tags,
    setup_sqlalchemy_engine,
    convert_to_docx,
    convert_to_pdf,
    cleanup_temp_files,
    FOLDER
)

torch.classes.__path__ = []

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Chatbot & RFP Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background: white;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .warning-box {
        background: linear-gradient(135deg, #ff9a56 0%, #ffad56 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    div[data-testid="stChatMessage"] {
        background-color:#F0F9FF;
        padding:1rem;
        border-radius:10px;
        margin:0.5rem 0;
        border-left:4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'agent_executor' not in st.session_state:
    st.session_state.agent_executor = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'files_uploaded' not in st.session_state:
    st.session_state.files_uploaded = False
if 'rfp_content' not in st.session_state:
    st.session_state.rfp_content = ""

def save_uploaded_files(uploaded_files):
    """Save uploaded files to the temporary directory"""
    data_dir = Path(FOLDER)
    data_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing files
    for file in data_dir.glob("*"):
        if file.is_file():
            try:
                file.unlink()
            except Exception as e:
                st.warning(f"Could not delete {file.name}: {str(e)}")
    # Save new files
    saved_files = []
    for uploaded_file in uploaded_files:
        try:
            file_path = data_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Error saving {uploaded_file.name}: {str(e)}")
    return saved_files

def setup_system():
    """Setup the RAG vectorstore and agent executor"""
    try:
        with st.spinner("Setting up system... This may take a few moments."):
            st.session_state.vectorstore = setup_rag_vectorstore(FOLDER)
            st.session_state.agent_executor = create_combined_agent_executor(st.session_state.vectorstore)
        return True
    except Exception as e:
        st.error(f"Error setting up system: {str(e)}")
        return False

def main():
    # Header
    st.markdown('<h1 class="main-header">Multimodal RAG Chatbot & RFP Generator</h1>', unsafe_allow_html=True)
    
    # Sidebar for file upload
    with st.sidebar:
        st.markdown('<h2 class="sub-header">📄 File Upload</h2>', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'docx', 'csv', 'txt', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'mp3', 'wav', 'flac', 'm4a', 'ogg', 'mp4', 'mov', 'avi'],
            accept_multiple_files=True,
            help="Upload documents that will be used for RFP generation and chatbot queries"
        )
        
        if uploaded_files and st.button("Process Files", type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                try:
                    with st.spinner("Saving files..."):
                        saved_files = save_uploaded_files(uploaded_files)
                    
                    if saved_files:
                        st.success(f"Saved {len(saved_files)} files")
                        st.session_state.files_uploaded = True
                        
                        with st.spinner("Processing documents and setting up system..."):
                            if setup_system():
                                st.success("✅ System ready!")
                                st.rerun()
                            else:
                                st.error("Failed to set up system. Please check your configuration.")
                    else:
                        st.error("No files were saved successfully.")
                        
                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")
        
        # System status
        st.markdown("---")
        st.markdown('<h3 class="sub-header">System Status</h3>', unsafe_allow_html=True)
        if st.session_state.files_uploaded and st.session_state.vectorstore:
            st.success("System Ready")
            if uploaded_files:
                st.info(f"📁 {len(uploaded_files)} files loaded")
        else:
            st.warning("Upload files to continue")
        
        # Clear button
        if st.session_state.files_uploaded:
            st.markdown("---")
            if st.button("🗑️ Clear All Data", type="secondary"):
                cleanup_temp_files()
                st.session_state.vectorstore = None
                st.session_state.agent_executor = None
                st.session_state.chat_history = []
                st.session_state.files_uploaded = False
                st.session_state.rfp_content = ""
                st.success("Data cleared!")
                st.rerun()
    
    # Main content area
    if not st.session_state.files_uploaded:
        st.markdown("""
        ### Welcome!
        
        **Getting Started:**
        1. Upload your documents using the sidebar
        2. Click "Process Files" to set up the system
        3. Choose your desired mode: Chatbot or RFP Generation
        """)
    else:
        # Mode selection tabs
        tab1, tab2 = st.tabs(["Chatbot Mode", "RFP Generator"])
        
        with tab1:
            st.markdown('<h2 class="sub-header">RAG + DB Chatbot</h2>', unsafe_allow_html=True)
            
            if st.session_state.agent_executor:
                # Chat interface
                st.write("Ask questions about your documents or database.")
                
                # Display chat history
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    with st.chat_message("user"):
                        st.markdown(question)
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                
                # Chat input
                if user_question := st.chat_input("Ask your question"):
                    try:
                        with st.spinner("Thinking..."):
                            result = st.session_state.agent_executor.invoke({"input": user_question})
                            answer = result["output"]
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_question, answer))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                
                # Clear chat button
                if st.session_state.chat_history:
                    if st.button("🗑️ Clear Chat"):
                        st.session_state.chat_history = []
                        st.rerun()
            else:
                st.warning("⚠️ System not initialized. Please process files first.")
        
        with tab2:
            st.markdown('<h2 class="sub-header">RFP Generator</h2>', unsafe_allow_html=True)
            st.write("Generate a comprehensive Request for Proposal (RFP) document based on your uploaded files and database information OR provide your own description.")
            
            # Input method selection
            input_method = st.radio(
                "Choose input method:",
                ["Use uploaded documents", "Provide description"],
                help="Select how you want to generate the RFP"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                project_type = st.text_input("Project Type:", value="general", help="Specify the type of project for the RFP")
            
            with col2:
                if input_method == "Use uploaded documents":
                    use_database = st.selectbox("Include Database Information:", ["No", "Yes"], help="Whether to include database analysis in the RFP")
                else:
                    use_database = "No"
            
            # User prompt input (only shown if "Provide description" is selected)
            user_prompt = ""
            if input_method == "Provide description":
                user_prompt = st.text_area(
                    "Describe your RFP requirements:",
                    height=200,
                    placeholder="Enter a detailed description of your project requirements, scope, deliverables, and any other relevant information...",
                    help="Provide a comprehensive description that will be used to generate the RFP"
                )
            
            if st.button("Generate RFP", type="primary"):
                # Validate input based on method
                if input_method == "Provide description" and not user_prompt.strip():
                    st.error("Please provide a description for the RFP.")
                else:
                    try:
                        with st.spinner("Generating RFP document... This may take a few moments."):
                            # Extract document content
                            documents_content = {}
                            if input_method == "Use uploaded documents":
                                documents_content = extract_all_document_content(FOLDER)
                                
                                if not documents_content:
                                    st.error("No documents found or no content extracted.")
                                    st.stop()
                            
                            # Get database summary if requested
                            database_summary = {}
                            if use_database == "Yes":
                                engine = setup_sqlalchemy_engine()
                                database_summary = get_database_summary(engine)
                            
                            # Generate RFP
                            db_use = "yes" if use_database == "Yes" else "no"
                            rfp_content = generate_rfp(documents_content, database_summary, project_type, db_use, user_prompt)
                            rfp_content = remove_think_tags(rfp_content)
                            st.session_state.rfp_content = rfp_content
                        
                        st.success("✅ RFP generated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating RFP: {str(e)}")
            
            # Show editor only if RFP content exists
            if st.session_state.rfp_content:
                st.markdown('<h3 class="sub-header">📝 RFP Editor</h3>', unsafe_allow_html=True)
                
                # Editor section
                edited_rfp_content = st.text_area(
                    "Edit the RFP content:",
                    value=st.session_state.rfp_content,
                    height=400,
                    key="rfp_editor"
                )
                
                # Save changes button
                if st.button("Save Changes"):
                    st.session_state.rfp_content = edited_rfp_content
                    st.success("Changes saved!")
                
                # Preview section
                st.markdown('<h3 class="sub-header">Preview</h3>', unsafe_allow_html=True)
                with st.expander("Click to view preview", expanded=False):
                    st.markdown(st.session_state.rfp_content)
                
                # Download section
                st.markdown('<h3 class="sub-header">📥 Download Options</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    filename_base = st.text_input("Filename (without extension):", value=f"rfp_{project_type}")
                
                with col2:
                    file_format = st.selectbox("Format:", ["md", "pdf", "docx"])
                
                # Download buttons
                if file_format == "md":
                    st.download_button(
                        label="Download Markdown",
                        data=st.session_state.rfp_content.encode("utf-8"),
                        file_name=f"{filename_base}.md",
                        mime="text/markdown"
                    )
                elif file_format == "pdf":
                    pdf_data = convert_to_pdf(st.session_state.rfp_content)
                    if pdf_data:
                        st.download_button(
                            label="Download PDF",
                            data=pdf_data,
                            file_name=f"{filename_base}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.error("PDF conversion requires reportlab. Install with: pip install reportlab")
                elif file_format == "docx":
                    docx_data = convert_to_docx(st.session_state.rfp_content)
                    if docx_data:
                        st.download_button(
                            label="Download DOCX",
                            data=docx_data,
                            file_name=f"{filename_base}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    else:
                        st.error("DOCX conversion requires python-docx. Install with: pip install python-docx")


if __name__ == "__main__":
    main()