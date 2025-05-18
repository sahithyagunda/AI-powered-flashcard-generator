import streamlit as st
import os
import fitz  # PyMuPDF
from docx import Document
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
import spacy
import en_core_web_sm

# Load models
summarizer = pipeline("summarization", model='facebook/bart-large-cnn')
tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl", use_fast=False)
model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")
qg_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
nlp = spacy.load("en_core_web_sm")

# Text extraction
def extract_text_pdf(file, start, end):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for i in range(start - 1, end):
        if i < len(pdf):
            page_text = pdf[i].get_text("text").replace("-\n", "")
            text += " ".join(page_text.split()) + "\n\n"
    return text.strip()

def extract_text_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_txt(file):
    return file.read().decode("utf-8")

def chunk_text(text, max_words=500):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def summarize_text(text):
    chunks = chunk_text(text)
    summaries = [summarizer(chunk, max_length=500, min_length=200, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

def generate_qa(summary):
    doc = nlp(summary)
    answers = set(ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "EVENT", "NORP", "CARDINAL", "ORDINAL"])
    qa_pairs = []
    for answer in answers:
        if answer in summary:
            highlighted = summary.replace(answer, f"<hl> {answer} <hl>")
            prompt = f"generate question: {highlighted}"
            result = qg_pipeline(prompt)
            question = result[0]['generated_text']
            qa_pairs.append((question, answer))
    return qa_pairs


def render_flashcards(qa_pairs):
    # Custom CSS for the flashcards
    st.markdown("""
    <style>
    .flashcard {
        height: 180px;
        perspective: 1000px;
        margin-bottom: 20px;
    }
    .card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.8s;
        transform-style: preserve-3d;
    }
    .flashcard:hover .card-inner {
        transform: rotateY(180deg);
    }
    .card-front, .card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
        font-family: 'Segoe UI', sans-serif;
        background-color: white;
        color: #222;
        border: 1px solid #eee;
        box-sizing: border-box;
        word-wrap: break-word;
        overflow: hidden;
    }
    .card-back {
        transform: rotateY(180deg);
    }
    </style>
    """, unsafe_allow_html=True)

    # Create rows with 4 columns each
    for i in range(0, len(qa_pairs), 4):
        cols = st.columns(4)
        row_pairs = qa_pairs[i:i+4]
        for col, (q, a) in zip(cols, row_pairs):
            with col:
                safe_q = str(q).replace('"', '&quot;').replace("'", "&apos;")
                safe_a = str(a).replace('"', '&quot;').replace("'", "&apos;")
                
                st.markdown(f"""
                <div class="flashcard">
                    <div class="card-inner">
                        <div class="card-front"><strong>Q:</strong> {safe_q}</div>
                        <div class="card-back"><strong>A:</strong> {safe_a}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# Streamlit UI
st.title("📘 Smart Flashcard Generator from Files")
st.write("Upload your file to generate a summary and flashcards with questions and answers.")

uploaded_file = st.file_uploader("Choose a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    text = ""

    if file_ext == ".pdf":
        start = st.number_input("Start Page", min_value=1, value=1)
        end = st.number_input("End Page", min_value=start, value=start)
        if st.button("Extract & Summarize"):
            with st.spinner("Processing PDF..."):
                text = extract_text_pdf(uploaded_file, start, end)

    elif file_ext == ".docx":
        if st.button("Extract & Summarize"):
            with st.spinner("Processing DOCX..."):
                text = extract_text_docx(uploaded_file)

    elif file_ext == ".txt":
        if st.button("Extract & Summarize"):
            with st.spinner("Processing TXT..."):
                text = extract_text_txt(uploaded_file)

    if text:
        with st.spinner("Summarizing..."):
            summary = summarize_text(text)
        st.subheader("Summary")
        st.write(summary)

        with st.spinner("Generating Flashcards..."):
            qa_pairs = generate_qa(summary)

        if qa_pairs:
            st.subheader(" Flashcards (Hover to Flip)")
            render_flashcards(qa_pairs)
        else:
            st.warning("No flashcards could be generated from the summary.")
