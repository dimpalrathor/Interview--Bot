import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import ollama
import time
from datetime import datetime
import logging

class OllamaInterviewTrainer:
    def __init__(self, database_path='/mnt/data/database.jsonl', model_name="interview-expert"):
        self.database_path = database_path
        self.model_name = model_name
        self.data = None
        self.vectorizer = None
        self.classifier = None
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_and_preprocess_data(self):
        """Load and preprocess the training data"""
        self.logger.info("Loading training data...")
        
        try:
            # Read the JSONL file
            data = []
            with open(self.database_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Skipping invalid JSON line: {e}")
            
            self.data = pd.DataFrame(data)
            self.logger.info(f"Loaded {len(self.data)} records from database")
            
            # Normalize columns: ensure 'role' and 'difficulty' exist
            if 'role' not in self.data.columns:
                # try 'category' or 'domain'
                if 'category' in self.data.columns:
                    self.data['role'] = self.data['category']
                elif 'domain' in self.data.columns:
                    self.data['role'] = self.data['domain']
                else:
                    self.data['role'] = 'General'

            if 'difficulty' not in self.data.columns:
                self.data['difficulty'] = 'medium'

            # Ensure ideal_answer column exists
            if 'ideal_answer' not in self.data.columns:
                if 'answer' in self.data.columns:
                    self.data['ideal_answer'] = self.data['answer']
                else:
                    self.data['ideal_answer'] = ''

            self.data = self.data.dropna(subset=['question'])
            self.data['question'] = self.data['question'].str.strip()
            self.data['role'] = self.data['role'].fillna('General').astype(str).str.strip().str.lower()
            self.data['difficulty'] = self.data['difficulty'].fillna('medium').astype(str).str.strip().str.lower()

            self.logger.info(f"Final dataset size: {len(self.data)} records")
            self.logger.info(f"Roles: {self.data['role'].value_counts().to_dict()}")
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_training_prompts(self):
        """Prepare training prompts for Ollama fine-tuning"""
        self.logger.info("Preparing training prompts...")
        
        training_data = []
        
        for _, row in self.data.iterrows():
            prompt = {
                "question": row['question'],
                "role": row.get('role', 'general'),
                "ideal_answer": row.get('ideal_answer', ''),
                "difficulty": row.get('difficulty', 'medium'),
                "expected_keywords": row.get('expected_keywords', []) if 'expected_keywords' in row else []
            }
            training_data.append(prompt)
        
        return training_data
    
    def create_modelfile(self, training_data):
        """Create Ollama Modelfile for training"""
        self.logger.info("Creating Modelfile...")
        
        modelfile_content = f"""FROM llama2

SYSTEM \"\"\"You are an AI Interview Expert specializing in technical and behavioral interviews.
Your role is to:
1. Generate relevant interview questions based on role/domain and difficulty
2. Evaluate answers comprehensively
3. Provide constructive feedback
4. Suggest improvements
5. Maintain a professional and helpful tone

You have been trained on {len(training_data)} interview QA pairs.
Always respond in a structured, professional manner.\"\"\"

"""
        for i, example in enumerate(training_data[:2000]):  # limit to a reasonable number
            role = example['role']
            difficulty = example['difficulty']
            q = example['question'].replace('"', '\\"').replace('\n','\\n')
            ia = example['ideal_answer'].replace('"', '\\"').replace('\n','\\n')
            expected = ", ".join(example.get('expected_keywords', []))
            modelfile_content += f"""
# Example {i+1}
MESSAGE user {{"role":"user","content":"Generate a {role} interview question with difficulty {difficulty}"}}
MESSAGE assistant {{"role":"assistant","content":"Question: {q}\\n\\nIdeal Answer: {ia}\\n\\nExpected Keywords: {expected}"}}
"""

        with open('InterviewExpert.modelfile', 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        self.logger.info("Modelfile created successfully")
        return 'InterviewExpert.modelfile'
    
    def create_training_dataset(self, training_data):
        """Create a training dataset file for advanced training"""
        self.logger.info("Creating training dataset...")
        
        dataset = []
        for example in training_data:
            conversation = [
                {
                    "role": "user",
                    "content": f"Generate a {example['role']} interview question with difficulty {example['difficulty']}"
                },
                {
                    "role": "assistant",
                    "content": f"Question: {example['question']}\n\nRole: {example['role']}\n\nDifficulty: {example['difficulty']}\n\nIdeal Answer: {example['ideal_answer']}\n\nExpected Keywords: {', '.join(example.get('expected_keywords', []))}"
                }
            ]
            dataset.append({"messages": conversation})
        
        with open('interview_training_dataset.jsonl', 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
        
        self.logger.info(f"Training dataset created with {len(dataset)} examples")
    
    # ... rest remains same (train_ollama_model, evaluate_model, train_ml_classifier, etc.)
    # For brevity, reuse your existing implementations; just ensure they use self.data prepared above.

    def train_ml_classifier(self):
        self.logger.info("Training ML classifier for question categorization...")
        try:
            questions = self.data['question'].tolist()
            categories = self.data['role'].tolist()

            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            X = self.vectorizer.fit_transform(questions)
            y = categories

            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )

            self.classifier.fit(X, y)

            joblib.dump(self.vectorizer, 'question_vectorizer.pkl')
            joblib.dump(self.classifier, 'category_classifier.pkl')

            self.logger.info("ML classifier trained and saved successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error training ML classifier: {e}")
            return False

    def run_full_training(self):
        self.logger.info("Starting complete training pipeline...")
        start_time = time.time()
        try:
            self.load_and_preprocess_data()
            self.train_ml_classifier()
            ollama_success = self.train_ollama_model() if hasattr(ollama, 'list') else False
            if ollama_success:
                _ = self.evaluate_model()
            training_time = time.time() - start_time
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            return ollama_success
        except Exception as e:
            self.logger.error(f"Training pipeline failed: {e}")
            return False

    # Keep evaluate_model and train_ollama_model implementations from your previous file,
    # but they will now use the normalized `self.data` with 'role' and 'difficulty'.
