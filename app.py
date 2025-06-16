import streamlit as st
import os
from resume_parser import ResumeParser
from nlp_processor import NLPProcessor
from model import ResumeClassifier
from utils import (
    create_directory, save_json, load_json, plot_metrics,
    format_percentage, is_valid_file_type, create_sample_dataset,
    load_sample_dataset, prepare_training_data
)

# Initialize components
resume_parser = ResumeParser()
nlp_processor = NLPProcessor()
classifier = ResumeClassifier()

# Set page config
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🔍",
    layout="wide"
)

# Create necessary directories and sample data
create_directory('data/raw')
create_directory('data/processed')
create_directory('models')

# Create sample dataset if it doesn't exist
sample_data_path = os.path.join('data/raw', 'sample_data.json')
if not os.path.exists(sample_data_path):
    create_sample_dataset()

def main():
    st.title("🔍📄 AI Resume Screening System")
    st.write("Upload resumes and job descriptions to automatically screen candidates.")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Resume Screening", "Model Training", "About"]
    )
    
    if page == "Resume Screening":
        show_screening_page()
    elif page == "Model Training":
        show_training_page()
    else:
        show_about_page()

def show_screening_page():
    st.header("Resume Screening")
    
    # Job Description Input
    st.subheader("Job Description")
    job_description = st.text_area(
        "Enter the job description",
        height=200
    )
    
    # Resume Upload
    st.subheader("Resume Upload")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF, DOCX, or TXT)",
        type=['pdf', 'docx', 'txt']
    )
    
    if uploaded_file and job_description:
        # Save uploaded file
        file_path = os.path.join('data/raw', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        try:
            # Parse resume
            resume_text = resume_parser.parse_resume(file_path)
            cleaned_text = resume_parser.clean_text(resume_text)
            
            # Extract entities
            entities = nlp_processor.extract_entities(cleaned_text)
            
            # Compute similarity
            similarity_score = nlp_processor.compute_similarity(
                cleaned_text,
                job_description
            )
            
            # Make prediction
            prediction, probability = classifier.predict(
                cleaned_text,
                job_description
            )
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Resume Analysis")
                st.write("**Extracted Skills:**")
                st.write(", ".join(entities['SKILLS']) if entities['SKILLS'] else "No skills found")
                
                st.write("**Work Experience:**")
                for exp in entities['ORGANIZATIONS']:
                    st.write(f"- {exp}")
            
            with col2:
                st.subheader("Match Results")
                st.write(f"**Similarity Score:** {format_percentage(similarity_score)}")
                st.write(f"**Prediction:** {'Shortlisted' if prediction == 1 else 'Rejected'}")
                st.write(f"**Confidence:** {format_percentage(probability)}")
                
                # Display match status
                if prediction == 1:
                    st.success("✅ This resume matches the job requirements!")
                else:
                    st.error("❌ This resume does not match the job requirements.")
            
        except Exception as e:
            st.error(f"Error processing resume: {str(e)}")
        
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

def show_training_page():
    st.header("Model Training")
    
    # Create sample dataset
    if st.button("Create Sample Dataset"):
        create_sample_dataset()
        st.success("Sample dataset created successfully!")
    
    # Load and prepare data
    try:
        data = load_sample_dataset()
        resumes, job_descriptions, labels = prepare_training_data(data)
        
        # Display sample data
        st.subheader("Sample Data")
        st.write("**Resumes:**")
        for i, resume in enumerate(resumes):
            st.write(f"{i+1}. {resume}")
        
        st.write("**Job Descriptions:**")
        for i, jd in enumerate(job_descriptions):
            st.write(f"{i+1}. {jd}")
        
        # Train model
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                metrics = classifier.train(resumes, job_descriptions, labels)
                
                # Display metrics
                st.subheader("Model Performance")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", format_percentage(metrics['accuracy']))
                with col2:
                    st.metric("Precision", format_percentage(metrics['precision']))
                with col3:
                    st.metric("Recall", format_percentage(metrics['recall']))
                with col4:
                    st.metric("F1 Score", format_percentage(metrics['f1']))
                
                # Plot metrics
                st.pyplot(plot_metrics(metrics))
                
                st.success("Model trained successfully!")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

def show_about_page():
    st.header("About")
    st.write("""
    This AI Resume Screening System helps automate the initial screening process by:
    
    1. **Parsing Resumes**: Extracts text from PDF, DOCX, and TXT files
    2. **NLP Processing**: Analyzes text using advanced NLP techniques
    3. **Matching**: Computes similarity between resumes and job descriptions
    4. **Classification**: Predicts if a resume matches the job requirements
    
    The system uses:
    - Natural Language Processing (NLP)
    - Machine Learning (Logistic Regression)
    - TF-IDF Vectorization
    - Cosine Similarity
    
    Built with Python, Streamlit, and scikit-learn.
    """)

if __name__ == "__main__":
    main() 