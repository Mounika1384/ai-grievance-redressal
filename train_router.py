import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib
import json

def train_scheme_router():
    # Load or create training data
    try:
        # Try to load from dataset
        queries_data = pd.read_csv('hackathon_dataset/sample_queries.csv')
        queries = queries_data['query'].tolist()
        labels = queries_data['scheme'].tolist()
    except:
        # Fallback to manual data
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
            "agriculture support",
            "how to apply for pm kisan",
            "ration card application",
            "pension scheme for senior citizens",
            "farmer income support",
            "get ration card",
            "national pension scheme",
            "kisan credit card",
            "public distribution system",
            "widow pension scheme",
            "disabled person pension"
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
            'PM-Kisan',
            'PM-Kisan',
            'Ration Card',
            'Pension',
            'PM-Kisan',
            'Ration Card',
            'Pension',
            'PM-Kisan',
            'Ration Card',
            'Pension',
            'Pension'
        ]
    
    # Create and train model
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(queries)
    
    model = MultinomialNB()
    model.fit(X, labels)
    
    # Save model
    joblib.dump({
        'vectorizer': vectorizer,
        'model': model
    }, 'scheme_router_model.joblib')
    
    print("Scheme router model trained and saved successfully!")
    
    # Test the model
    test_queries = [
        "I need help as a farmer",
        "How to get food grains",
        "Pension for my grandmother"
    ]
    
    X_test = vectorizer.transform(test_queries)
    predictions = model.predict(X_test)
    
    for query, pred in zip(test_queries, predictions):
        print(f"Query: '{query}' -> Scheme: {pred}")

if __name__ == '__main__':
    train_scheme_router()