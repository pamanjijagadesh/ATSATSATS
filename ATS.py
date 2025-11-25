import streamlit as st
import tempfile
import os
import pandas as pd
import docx2txt
import asyncio
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Direct import of pack (NO download on Streamlit Cloud)
from llama_index.packs.resume_screener import ResumeScreenerPack
from llama_index.llms.azure_inference import AzureAICompletionsModel

st.set_page_config(
    page_title="FriskaAi Resume Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
#  🔥 HARD-CODED ENDPOINT & API KEY (as requested)
# -----------------------------------------
AZURE_ENDPOINT = "https://mistral-small-2503-Pamanji-test.southcentralus.models.ai.azure.com"
AZURE_CREDENTIAL = "5SKKbylMh5ueyeSfvUre68vknfYZMVAr"
# -----------------------------------------

st.title("📄 Friska AI Resume Tracker")
st.markdown("Upload resumes, provide a job description, and list the specific criteria for automatic ATS scoring and decision making.")

@st.cache_resource
def initialize_llm_and_pack():
    """Initializes and caches the LLM + Resume Screener Pack."""
    try:
        with st.status("Initializing AI Resources...", expanded=True) as status:

            # 1. Initialize Azure AI model
            status.update(label="Initializing Azure AI Model...")
            llm = AzureAICompletionsModel(
                endpoint=AZURE_ENDPOINT,
                credential=AZURE_CREDENTIAL
            )

            # 2. Direct import of pack class
            status.update(label="Loading Resume Screener Pack...")
            pack_class = ResumeScreenerPack

            status.update(label="AI Resources Ready!", state="complete")
            return pack_class, llm

    except Exception as e:
        st.error(f"Error initializing LLM or LlamaPack: {e}")
        st.stop()

ResumeScreenerPack, llm = initialize_llm_and_pack()

# ------------------------------------------------------
# DOCX → PDF conversion
# ------------------------------------------------------
def _docx_to_pdf_sync(input_path: str, output_path: str = None) -> str:
    if not input_path.lower().endswith(".docx"):
        raise ValueError("Input file must be a .docx file")

    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".pdf"

    text = docx2txt.process(input_path)
    pdf = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y = height - 50

    for line in text.splitlines():
        if y <= 50:
            pdf.showPage()
            y = height - 50
        pdf.drawString(50, y, line)
        y -= 15

    pdf.save()
    return output_path

async def docx_to_pdf(input_path: str, output_path: str = None) -> str:
    return await asyncio.to_thread(_docx_to_pdf_sync, input_path, output_path)


# Run LLM pack in thread
def _run_screener_sync(screener_instance, resume_path):
    return screener_instance.run(resume_path=resume_path)


# ------------------------------------------------------
# PROCESS SINGLE RESUME ASYNC
# ------------------------------------------------------
async def process_single_resume_async(uploaded_file, job_description, criteria, ResumeScreenerPack, llm, temp_dir):
    resume_name = uploaded_file.name
    temp_file_path = os.path.join(temp_dir, resume_name)

    def save_file():
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return temp_file_path

    try:
        temp_file_path = await asyncio.to_thread(save_file)

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

        screener = ResumeScreenerPack(
            job_description=job_description,
            criteria=criteria,
            llm=llm,
        )

        response = await asyncio.to_thread(_run_screener_sync, screener, temp_file_path)

        total_criteria = len(response.criteria_decisions)
        passed_criteria = sum(1 for cd in response.criteria_decisions if cd.decision)
        ats_score = (passed_criteria / total_criteria) * 100 if total_criteria else 0
        overall_decision_text = "Recommended (Pass)" if response.overall_decision else "Not Recommended (Fail)"

        return {
            "Resume Name": uploaded_file.name,
            "ATS Score Raw": ats_score,
            "ATS Overall Decision": overall_decision_text,
            "Overall Reasoning": response.overall_reasoning
        }

    except Exception as e:
        st.error(f"Failed to process {resume_name}: {e}")
        return {
            "Resume Name": uploaded_file.name,
            "ATS Score Raw": 0.0,
            "ATS Overall Decision": "Processing Failed",
            "Overall Reasoning": f"Error: {e}"
        }

# ------------------------------------------------------
# RUN ALL RESUMES
# ------------------------------------------------------
async def run_screening_async(uploaded_files, job_description, criteria, ResumeScreenerPack, llm, progress_bar_placeholder):
    all_results = []
    tasks = []
    total_files = len(uploaded_files)
    completed = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            tasks.append(process_single_resume_async(file, job_description, criteria, ResumeScreenerPack, llm, temp_dir))

        for future in asyncio.as_completed(tasks):
            result = await future
            if result:
                all_results.append(result)

            completed += 1
            pct = int((completed / total_files) * 100)
            progress_bar_placeholder.progress(
                pct,
                text=f"Processing {completed}/{total_files} ({pct}%)"
            )

    return all_results


# ------------------------------------------------------
# UI SECTION
# ------------------------------------------------------
st.header("1. Job Description")
job_description = st.text_area("Paste the full Job Description (JD):", height=200)

st.header("2. Screening Criteria")
criteria_text = st.text_area("List mandatory criteria (one per line):", height=150)
criteria = [c.strip() for c in criteria_text.split("\n") if c.strip()]

st.header("3. Upload Resumes (PDF or DOCX)")
uploaded_files = st.file_uploader("Upload resumes", type=["pdf", "docx"], accept_multiple_files=True)

if st.button("Screen Resumes Concurrently", type="primary"):
    if not job_description or not criteria or not uploaded_files:
        st.warning("Please enter JD, criteria and upload resumes.")
        st.stop()

    st.subheader("Processing Results Summary")
    progress = st.empty()

    with st.status("Running Concurrent Screening...", expanded=True) as status:
        try:
            results = asyncio.run(
                run_screening_async(uploaded_files, job_description, criteria, ResumeScreenerPack, llm, progress)
            )
            status.update(label=f"Completed! ({len(results)} resumes processed)", state="complete")
        except Exception as e:
            status.update(label="Screening failed!", state="error")
            st.error(f"Error: {e}")
            results = []

    progress.empty()

    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="ATS Score Raw", ascending=False)

        def fmt(row):
            if row["ATS Overall Decision"] in ["Skipped", "Processing Failed"]:
                return row["ATS Overall Decision"]
            return f"{row['ATS Score Raw']:.2f}%"

        df["ATS Score (%)"] = df.apply(fmt, axis=1)
        df = df.drop(columns=["ATS Score Raw"])

        df = df[["Resume Name", "ATS Score (%)", "ATS Overall Decision", "Overall Reasoning"]]
        st.dataframe(df, width="stretch", hide_index=True)
