import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from typing import Tuple, Dict, List
import os

class ResumeClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.model_path = 'models/resume_classifier.joblib'
        self.vectorizer_path = 'models/vectorizer.joblib'
        self.is_fitted = False
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Try to load existing model and vectorizer
        self.load_model()
    
    def prepare_features(self, resumes: List[str], job_descriptions: List[str]) -> np.ndarray:
        """
        Prepare features by combining resume and job description text
        """
        combined_texts = [
            f"{resume} {jd}" for resume, jd in zip(resumes, job_descriptions)
        ]
        return self.vectorizer.fit_transform(combined_texts)
    
    def train(self, resumes: List[str], job_descriptions: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Train the classifier on resume-JD pairs
        """
        # Prepare features
        X = self.prepare_features(resumes, job_descriptions)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Make predictions on test set
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Save model and vectorizer
        self.save_model()
        
        return metrics
    
    def predict(self, resume: str, job_description: str) -> Tuple[int, float]:
        """
        Predict if a resume matches a job description
        """
        if not self.is_fitted:
            # If model is not fitted, use a simple similarity-based approach
            from sklearn.metrics.pairwise import cosine_similarity
            combined_text = f"{resume} {job_description}"
            X = self.vectorizer.fit_transform([combined_text])
            similarity = cosine_similarity(X[0:1], X[0:1])[0][0]
            prediction = 1 if similarity > 0.5 else 0
            return prediction, similarity
        
        # Prepare features
        X = self.vectorizer.transform([f"{resume} {job_description}"])
        
        # Get prediction and probability
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]
        
        return prediction, probability
    
    def save_model(self):
        """
        Save the trained model and vectorizer
        """
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
    
    def load_model(self):
        """
        Load the trained model and vectorizer
        """
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.is_fitted = True
            return True
        return False
    
    def evaluate(self, resumes: List[str], job_descriptions: List[str], labels: List[int]) -> Dict[str, float]:
        """
        Evaluate the model on new data
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before evaluation")
            
        X = self.vectorizer.transform([
            f"{resume} {jd}" for resume, jd in zip(resumes, job_descriptions)
        ])
        y = np.array(labels)
        
        y_pred = self.model.predict(X)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        } 