import os
import json
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_directory(directory: str):
    """
    Create directory if it doesn't exist
    """
    os.makedirs(directory, exist_ok=True)

def save_json(data: Dict[str, Any], filepath: str):
    """
    Save data to JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_metrics(metrics: Dict[str, float], title: str = "Model Performance Metrics"):
    """
    Plot model performance metrics
    """
    plt.figure(figsize=(10, 6))
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Score'])
    sns.barplot(x='Metric', y='Score', data=metrics_df)
    plt.title(title)
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def format_percentage(value: float) -> str:
    """
    Format float as percentage string
    """
    return f"{value:.1%}"

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename
    """
    return os.path.splitext(filename)[1].lower()

def is_valid_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Check if file has valid extension
    """
    return get_file_extension(filename) in allowed_extensions

def create_sample_dataset(output_dir: str = 'data/raw'):
    """
    Create a sample dataset for testing with both positive and negative examples
    """
    create_directory(output_dir)
    
    # Sample data with both positive and negative examples
    sample_data = {
        'resumes': [
            {
                'text': 'Experienced Python developer with 5 years of experience in web development, machine learning, and data analysis. Proficient in Django, Flask, and scikit-learn.',
                'label': 1
            },
            {
                'text': 'Senior Software Engineer with 8 years of experience in Java development, Spring Framework, and microservices architecture.',
                'label': 0
            },
            {
                'text': 'Data Scientist with 3 years of experience in Python, pandas, and scikit-learn. Strong background in statistical analysis and machine learning.',
                'label': 1
            },
            {
                'text': 'Frontend Developer with 4 years of experience in React, JavaScript, and CSS. No backend or Python experience.',
                'label': 0
            }
        ],
        'job_descriptions': [
            'Looking for a Python developer with experience in web development and machine learning. Must have strong knowledge of Django/Flask and data analysis.',
            'Senior Java Developer position requiring extensive experience with Spring Framework and microservices.',
            'Data Scientist position requiring Python, pandas, and machine learning experience.',
            'Frontend Developer position focusing on React and modern JavaScript frameworks.'
        ]
    }
    
    # Save sample data
    save_json(sample_data, os.path.join(output_dir, 'sample_data.json'))

def load_sample_dataset(input_dir: str = 'data/raw') -> Dict[str, Any]:
    """
    Load sample dataset
    """
    return load_json(os.path.join(input_dir, 'sample_data.json'))

def prepare_training_data(data: Dict[str, Any]) -> tuple:
    """
    Prepare data for training
    """
    resumes = [item['text'] for item in data['resumes']]
    labels = [item['label'] for item in data['resumes']]
    job_descriptions = data['job_descriptions']
    
    return resumes, job_descriptions, labels 