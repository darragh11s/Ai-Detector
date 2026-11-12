# Ai-Detector
A local, open-source Python &amp; Streamlit app that compares a blank template against a filled-out document (.docx, .pdf, .txt), isolates only the newly added content—including Word tables—and highlights potential AI-generated writing using a local DeBERTa-v3 detector.
# Word Document Diff & Open-Source AI Detector

A local web app built with Streamlit and Hugging Face Transformers that compares a blank Word form (`.docx`) against a completed version—including standard paragraphs, tables, merged cells, and nested tables—and evaluates only the newly added content for AI generation.

## Setup & Installation
Make sure you have python installed
Save Files into a folder requirements.txt and app.py
navigate to the directory in powershell

cd C:/WhereverYouSavedTheFolder

1. Install the required packages:

   pip install -r requirements.txt


2. Run the application:

		python -m streamlit run app.py


3. Open the local URL shown in your terminal (usually `http://localhost:8501`). The first run will download the open-source ai-detector model (~500 MB) and cache it locally.
