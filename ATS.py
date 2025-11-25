import streamlit as st
import tempfile
import os
import pandas as pd
import docx2txt
import asyncio
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Try to import necessary LlamaIndex components.
try:
    # Ensure llama-index imports are available
    from llama_index.core.llama_pack import download_llama_pack
    from llama_index.llms.azure_inference import AzureAICompletionsModel
except ImportError:
    st.error("LlamaIndex libraries not found. Please ensure 'llama-index' and necessary underlying packages are installed.")
    st.stop()

st.set_page_config(
    page_title="FriskaAi Resume Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONFIGURATION (Ensure these are secured in st.secrets in production) ---
AZURE_ENDPOINT = "https://mistral-small-2503-Pamanji-test.southcentralus.models.ai.azure.com"
AZURE_CREDENTIAL = "5SKKbylMh5ueyeSfvUre68vknfYZMVAr"

st.title("📄 Friska AI Resume Tracker (Async)")
st.markdown("Upload resumes, provide a job description, and list the specific criteria for automatic ATS scoring and decision making.")

@st.cache_resource
def initialize_llm_and_pack():
    """Initializes and caches the LLM and the LlamaPack with temporary status indicators."""
    try:
        # Use st.status to show temporary initialization progress with a spinner
        with st.status("Initializing AI Resources...", expanded=True) as status:
            
            # 1. Initialize LLM
            status.update(label="Initializing Azure AI Completions Model...")
            llm = AzureAICompletionsModel(
                endpoint=AZURE_ENDPOINT,
                credential=AZURE_CREDENTIAL,
            )

            # 2. Download and Initialize Resume Screener Pack
            status.update(label="Downloading and initializing ResumeScreenerPack...")
            # Create a temporary directory for the pack download
            pack_dir = os.path.join(tempfile.gettempdir(), "resume_screener_pack_cache")
            os.makedirs(pack_dir, exist_ok=True)

            ResumeScreenerPack = download_llama_pack(
                "ResumeScreenerPack", pack_dir
            )
            
            # Change status to 'complete' and collapses the status box
            status.update(label="AI Resources Ready!", state="complete")
            return ResumeScreenerPack, llm

    except Exception as e:
        st.error(f"Error initializing LLM or LlamaPack: {e}")
        st.stop()


# Load resources
ResumeScreenerPack, llm = initialize_llm_and_pack()

# --- DOCX ➜ PDF conversion helper (Synchronous part) ---
def _docx_to_pdf_sync(input_path: str, output_path: str = None) -> str:
    """Synchronous function to convert DOCX to PDF (blocking operation)."""
    if not input_path.lower().endswith(".docx"):
        raise ValueError("Input file must be a .docx file")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".pdf"

    text = docx2txt.process(input_path)
    pdf = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 50

    # Draw text content onto the PDF canvas
    for line in text.splitlines():
        if y <= 50:
            pdf.showPage()
            y = height - 50
        pdf.drawString(50, y, line)
        y -= 15

    pdf.save()
    return output_path

async def docx_to_pdf(input_path: str, output_path: str = None) -> str:
    """Async wrapper for the blocking DOCX to PDF conversion."""
    # This runs the synchronous function in a separate thread, making it non-blocking
    return await asyncio.to_thread(_docx_to_pdf_sync, input_path, output_path)

# --- ASYNC RESUME PROCESSING LOGIC ---

# Helper function to wrap the synchronous LLM call
def _run_screener_sync(screener_instance, resume_path):
    """Executes the synchronous LlamaPack run method in a separate thread."""
    return screener_instance.run(resume_path=resume_path)


async def process_single_resume_async(
    uploaded_file, job_description, criteria, ResumeScreenerPack, llm, temp_dir
):
    """
    Handles the end-to-end processing of a single resume, including file I/O 
    and concurrent LLM calling.
    """
    resume_name = uploaded_file.name
    temp_file_path = os.path.join(temp_dir, resume_name)
    
    def save_file():
        """Synchronous file saving for use with asyncio.to_thread."""
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_file_path

    try:
        # 1. Save uploaded file temporarily (Blocking I/O - wrapped with to_thread)
        temp_file_path = await asyncio.to_thread(save_file)
        
        # 2. Convert DOCX to PDF if needed (I/O/CPU-bound - wrapped with async)
        if resume_name.lower().endswith(".docx"):
            pdf_path = await docx_to_pdf(temp_file_path)
            temp_file_path = pdf_path
        elif not resume_name.lower().endswith(".pdf"):
            return {
                "Resume Name": resume_name,
                "ATS Score Raw": 0.0,
                "ATS Overall Decision": "Skipped",
                "Overall Reasoning": "Unsupported file type."
            }
        
        # 3. Initialize screener (This is fast)
        screener = ResumeScreenerPack(
            job_description=job_description,
            criteria=criteria,
            llm=llm,
        )
        
        # 4. FIXED: Use asyncio.to_thread to run the synchronous 'run' method non-blockingly
        response = await asyncio.to_thread(_run_screener_sync, screener, temp_file_path)
        
        # 5. Calculate and return results
        total_criteria = len(response.criteria_decisions)
        passed_criteria = sum(1 for cd in response.criteria_decisions if cd.decision is True)
        ats_score = (passed_criteria / total_criteria) * 100 if total_criteria > 0 else 0
        overall_decision_text = "Recommended (Pass)" if response.overall_decision else "Not Recommended (Fail)"

        return {
            "Resume Name": uploaded_file.name,
            "ATS Score Raw": ats_score,
            "ATS Overall Decision": overall_decision_text,
            "Overall Reasoning": response.overall_reasoning
        }

    except Exception as e:
        error_message = f"Error during processing: {e}"
        # st.error remains for persistent error messages, as users need to see these.
        st.error(f"Failed to process {resume_name}: {e}") 
        return {
            "Resume Name": uploaded_file.name,
            "ATS Score Raw": 0.0,
            "ATS Overall Decision": "Processing Failed",
            "Overall Reasoning": error_message
        }

async def run_screening_async(uploaded_files, job_description, criteria, ResumeScreenerPack, llm, progress_bar_placeholder):
    """Creates and runs all single-resume processing tasks concurrently, updating the progress bar."""
    all_results = []
    tasks = []
    total_files = len(uploaded_files)
    completed_tasks = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            tasks.append(
                process_single_resume_async(file, job_description, criteria, ResumeScreenerPack, llm, temp_dir)
            )
        
        # Iterate over tasks as they complete (concurrently)
        for future in asyncio.as_completed(tasks):
            result = await future
            if result is not None:
                all_results.append(result)

            # Update progress bar
            completed_tasks += 1
            progress_percent = int((completed_tasks / total_files) * 100)
            progress_bar_placeholder.progress(
                progress_percent, 
                text=f"Processing resumes... {completed_tasks}/{total_files} ({progress_percent}%)"
            )
    
    return [r for r in all_results if r is not None]

# --- UI INPUTS ---

# 1. Job Description Input
st.header("1. Job Description")
job_description = st.text_area(
    "Paste the full Job Description (JD) here:",
    height=200,
    value="""Meta is embarking on the most transformative change to its business and technology in company history, and our Machine Learning Engineers are at the forefront of this evolution. By leading crucial projects and initiatives that have never been done before, you have an opportunity to help us advance the way people connect around the world."""
)

# 2. Criteria Input
st.header("2. Screening Criteria")
criteria_text = st.text_area(
    "List specific, mandatory criteria (one per line):",
    height=150,
    value="""2+ years of experience in one or more of the following areas: machine learning, recommendation systems, pattern recognition, data mining, artificial intelligence, or related technical field
Experience demonstrating technical leadership working with teams, owning projects, defining and setting technical direction for projects
Bachelor's degree in Computer Science, Computer Engineering, relevant technical field, or equivalent practical experience."""
)
# Convert multiline text into a list
criteria = [c.strip() for c in criteria_text.split('\n') if c.strip()]

# 3. Resume Upload
st.header("3. Upload Resumes (PDF, DOCX)")
uploaded_files = st.file_uploader(
    "Upload one or more resumes:",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

# --- PROCESSING ---
if st.button("Screen Resumes Concurrently", type="primary"):
    if not job_description or not criteria or not uploaded_files:
        st.warning("Please fill in the Job Description, Criteria, and upload at least one resume.")
        st.stop()

    st.subheader("Processing Results Summary (Running Concurrently)")

    # 1. Create a placeholder for the progress bar that we will clear later
    progress_bar_placeholder = st.empty()
    
    # This block uses a spinning status indicator that will disappear when complete
    results_count = 0
    with st.status("Running Concurrent Screening...", expanded=True) as status:
        try:
            # Pass the progress bar placeholder to the async function
            results_data = asyncio.run(
                run_screening_async(
                    uploaded_files, job_description, criteria, ResumeScreenerPack, llm, progress_bar_placeholder
                )
            )
            results_count = len(results_data)
            # Refined completion message
            status.update(label=f"Concurrent screening complete! ({results_count} resumes processed)", state="complete")
        except Exception as e:
            status.update(label="Screening failed!", state="error")
            st.error(f"An error occurred during asynchronous screening: {e}")
            results_data = []

    # 2. Clear the progress bar from the screen after processing is fully complete
    progress_bar_placeholder.empty()

    # --- Display Results ---
    if results_data:
        df = pd.DataFrame(results_data)
        df = df.sort_values(by="ATS Score Raw", ascending=False)

        def format_score(row):
            if row["ATS Overall Decision"] in ["Processing Failed", "Skipped"]:
                return row["ATS Overall Decision"]
            return f"{row['ATS Score Raw']:.2f}%"

        df["ATS Score (%)"] = df.apply(format_score, axis=1)
        df = df.drop(columns=["ATS Score Raw"])
        df = df[["Resume Name", "ATS Score (%)", "ATS Overall Decision", "Overall Reasoning"]]

        st.dataframe(df, width='stretch', hide_index=True)