import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

class SchemeMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = MultinomialNB()
        self.is_trained = False
        
        # Try to load pre-trained model
        if os.path.exists('scheme_router_model.joblib'):
            self.load_model()
        else:
            self.train_model()
    
    def train_model(self):
        # Sample training data - in practice, use your dataset
        queries = [
            "pm kisan farmer scheme",
            "ration card food grains",
            "pension for old people",
            "kisan samman nidhi",
            "food security card",
            "old age pension",
            "farmer financial help",
            "subsidized food",
            "retirement benefits",
            "agriculture support"
        ]
        
        labels = [
            'PM-Kisan',
            'Ration Card',
            'Pension',
            'PM-Kisan',
            'Ration Card',
            'Pension',
            'PM-Kisan',
            'Ration Card',
            'Pension',
            'PM-Kisan'
        ]
        
        # Train TF-IDF and classifier
        X = self.vectorizer.fit_transform(queries)
        self.model.fit(X, labels)
        self.is_trained = True
        
        # Save model
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model
        }, 'scheme_router_model.joblib')
    
    def load_model(self):
        try:
            model_data = joblib.load('scheme_router_model.joblib')
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            self.is_trained = True
        except:
            self.train_model()
    
    def route_scheme(self, query):
        if not self.is_trained:
            self.train_model()
        
        # Transform query and predict
        X = self.vectorizer.transform([query])
        prediction = self.model.predict(X)
        return prediction[0]
    
    def check_eligibility(self, scheme_name, profile):
        age = profile.get('age', 0)
        income = profile.get('income', 0)
        landholding = profile.get('landholding', 0)
        caste = profile.get('caste', '').lower()
        residence = profile.get('residence', 'rural').lower()
        
        reasons = []
        eligible = True
        
        if scheme_name == 'PM-Kisan':
            if landholding <= 0:
                eligible = False
                reasons.append("Must have cultivable land")
            if income > 150000:
                eligible = False
                reasons.append("Annual income exceeds ₹1,50,000 limit")
            if age < 18:
                eligible = False
                reasons.append("Must be at least 18 years old")
                
        elif scheme_name == 'Ration Card':
            if income > 100000:
                eligible = False
                reasons.append("Annual income exceeds ₹1,00,000 limit for priority households")
            if not residence in ['rural', 'urban']:
                eligible = False
                reasons.append("Residence must be specified")
                
        elif scheme_name == 'Pension':
            if age < 60:
                eligible = False
                reasons.append("Must be at least 60 years old for old age pension")
            if income > 100000:
                eligible = False
                reasons.append("Annual income exceeds ₹1,00,000 limit")
        
        # Special categories
        if caste in ['sc', 'st', 'obc']:
            reasons.append(f"Eligibility enhanced for {caste.upper()} category")
        
        return {
            'eligible': eligible,
            'reasons': reasons
        }