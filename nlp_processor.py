import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple, Dict
import spacy

class NLPProcessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            spacy.cli.download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by tokenizing, removing stopwords, and lemmatizing
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
        
        return ' '.join(processed_tokens)
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities using spaCy
        """
        doc = self.nlp(text)
        entities = {
            'SKILLS': [],
            'ORGANIZATIONS': [],
            'DATES': [],
            'DEGREES': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                entities['ORGANIZATIONS'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['DATES'].append(ent.text)
        
        return entities
    
    def compute_similarity(self, resume_text: str, jd_text: str) -> float:
        """
        Compute cosine similarity between resume and job description
        """
        # Preprocess texts
        processed_resume = self.preprocess_text(resume_text)
        processed_jd = self.preprocess_text(jd_text)
        
        # Create TF-IDF vectors
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_resume, processed_jd])
        
        # Compute cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract top keywords from text using TF-IDF
        """
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Create TF-IDF vector
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text])
        
        # Get feature names and scores
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Create keyword-score pairs and sort by score
        keyword_scores = list(zip(feature_names, scores))
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores[:top_n]
    
    def extract_experience(self, text: str) -> List[Dict]:
        """
        Extract work experience information using spaCy
        """
        doc = self.nlp(text)
        experiences = []
        
        # Simple rule-based extraction
        # This can be enhanced with more sophisticated patterns
        current_org = None
        current_date = None
        
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                current_org = ent.text
            elif ent.label_ == 'DATE':
                current_date = ent.text
                if current_org:
                    experiences.append({
                        'organization': current_org,
                        'date': current_date
                    })
        
        return experiences 