# AI Resume Screening System 🔍📄

An intelligent resume screening system that automates the initial candidate evaluation process using Natural Language Processing (NLP) and Machine Learning.

## Features ✨

- **Resume Parsing**: Extract text from PDF, DOCX, and TXT files
- **NLP Processing**: Advanced text analysis and entity extraction
- **Smart Matching**: Compute similarity between resumes and job descriptions
- **AI Classification**: Predict candidate-job match using machine learning
- **Interactive UI**: User-friendly Streamlit interface
- **Model Training**: Train and evaluate the classifier with sample data

## Tech Stack 🛠️

- Python 3.x
- Streamlit (Web Interface)
- scikit-learn (Machine Learning)
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Cosine Similarity

## Installation 📥

1. Clone the repository:
```bash
git clone https://github.com/yourusername/resume-screening-classifier.git
cd resume-screening-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 🚀

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Use the navigation sidebar to access different features:
   - **Resume Screening**: Upload resumes and job descriptions for analysis
   - **Model Training**: Train and evaluate the classifier
   - **About**: Learn more about the system

## Project Structure 📁

```
resume-screening-classifier/
├── app.py                 # Main Streamlit application
├── resume_parser.py       # Resume parsing functionality
├── nlp_processor.py       # NLP processing and analysis
├── model.py              # Machine learning model
├── utils.py              # Utility functions
├── data/                 # Data directory
│   ├── raw/             # Raw resume data
│   └── processed/       # Processed data
└── models/              # Saved model files
```

## Features in Detail 📋

### Resume Screening
- Upload resumes in PDF, DOCX, or TXT format
- Enter job descriptions
- View extracted skills and work experience
- Get similarity scores and match predictions
- See confidence levels for predictions

### Model Training
- Create and manage sample datasets
- Train the classifier
- View model performance metrics
- Visualize results with interactive plots

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Built with Streamlit
- Powered by scikit-learn
- Inspired by the need for efficient resume screening

