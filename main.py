# import streamlit as st
# import PyPDF2
# from transformers import BertTokenizer, BertModel
# import torch
# from sklearn.metrics.pairwise import cosine_similarity

# # ... (Your existing calculate_text_similarity and extract_text_from_pdf functions) ...

# st.title("PDF Analyzer")

# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
# job_description = st.text_area("Enter job description")

# if uploaded_file is not None and job_description:
#     extracted_text = extract_text_from_pdf(uploaded_file)
#     similarity = calculate_text_similarity(extracted_text, job_description)

#     if similarity is not None:
#         st.write(f"Similarity Score: {similarity:.4f}")
#         if similarity > 0.9:
#             st.write("The resume and job description are quite similar.")
#         elif similarity > 0.5 and similarity <= 0.9:
#             st.write("The resume and job description are not similar.")
#         else:
#             st.write("The resume and job description are not very similar.")
#     else:
#         st.write("Failed to calculate similarity.")

import streamlit as st
import PyPDF2
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def calculate_text_similarity(text1, text2):
    tokenizer, model = load_bert_model()
    try:
        inputs1 = tokenizer(text1, return_tensors='pt', truncation=True, padding=True)
        outputs1 = model(**inputs1)
        embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()

        inputs2 = tokenizer(text2, return_tensors='pt', truncation=True, padding=True)
        outputs2 = model(**inputs2)
        embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()

        similarity_score = cosine_similarity(embeddings1, embeddings2)[0][0]
        return similarity_score
    except Exception as e:
        st.error(f"Error calculating text similarity: {e}")
        return None

st.title("Resume to Job Matching engine")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
job_description = st.text_area("Enter job description")

if uploaded_file is not None and job_description:
    extracted_text = extract_text_from_pdf(uploaded_file)
    if extracted_text:
        similarity = calculate_text_similarity(extracted_text, job_description)

        if similarity is not None:
            st.write(f"Similarity Score: {similarity:.4f}")
            if similarity > 0.8:
                st.success("The resume and job description are highly similar.")
            elif similarity > 0.6 and similarity <= 0.8:
                st.info("The resume and job description show some similarity.")
            else:
                st.warning("The resume and job description are not very similar.")
        else:
            st.write("Failed to calculate similarity.")