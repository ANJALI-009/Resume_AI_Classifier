import pdfplumber
from docx import Document
import re
from typing import Union, List
import os

class ResumeParser:
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt']
    
    def parse_resume(self, file_path: str) -> str:
        """
        Parse resume from different file formats and extract text
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format. Supported formats are: {self.supported_formats}")
        
        if file_ext == '.pdf':
            return self._parse_pdf(file_path)
        elif file_ext == '.docx':
            return self._parse_docx(file_path)
        else:  # .txt
            return self._parse_txt(file_path)
    
    def _parse_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF file
        """
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error parsing PDF file: {str(e)}")
        return text
    
    def _parse_docx(self, file_path: str) -> str:
        """
        Extract text from DOCX file
        """
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise Exception(f"Error parsing DOCX file: {str(e)}")
        return text
    
    def _parse_txt(self, file_path: str) -> str:
        """
        Extract text from TXT file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            raise Exception(f"Error parsing TXT file: {str(e)}")
        return text
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing extra whitespace and special characters
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        return text.strip()
    
    def extract_skills(self, text: str, skills_list: List[str]) -> List[str]:
        """
        Extract skills from resume text based on a predefined skills list
        """
        found_skills = []
        text = text.lower()
        
        for skill in skills_list:
            if skill.lower() in text:
                found_skills.append(skill)
        
        return found_skills 