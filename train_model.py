import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_training_data():
    """Load training data from CSV files or create synthetic data"""
    try:
        # Try to load from dataset
        train_data = pd.read_csv('hackathon_dataset/sample_queries.csv')
        print(f"Loaded {len(train_data)} training examples")
        return train_data
    except FileNotFoundError:
        print("Training data file not found. Creating synthetic data...")
        # Create synthetic training data
        queries = [
            "pm kisan farmer scheme", "PM-KISAN registration", "kisan samman nidhi yojana",
            "ration card application", "food security card", "ration card eligibility",
            "old age pension", "pension scheme for senior citizens", "widow pension",
            "how to apply for pm kisan", "pm kisan status check", "kisan credit card",
            "ration card renewal", "new ration card apply", "ration card online",
            "national pension scheme", "pension for farmers", "disability pension",
            "pm kisan beneficiary status", "pm kisan installment", "kisan registration",
            "ration card documents required", "ration card list", "ration card eligibility criteria",
            "pension amount for senior citizens", "pension scheme for widows", "pension application form",
            "agriculture pension scheme", "farmers pension", "kisan pension yojana",
            "ration card for bpl family", "ration card address change", "ration card correction",
            "pension for retired employees", "state pension scheme", "central government pension",
            "pm kisan helpline number", "kisan call center", "pm kisan official website",
            "ration card helpline", "ration card customer care", "ration card portal",
            "pension portal", "pension disbursement", "pension bank account linking"
        ]
        
        labels = [
            'PM-Kisan', 'PM-Kisan', 'PM-Kisan',
            'Ration Card', 'Ration Card', 'Ration Card',
            'Pension', 'Pension', 'Pension',
            'PM-Kisan', 'PM-Kisan', 'PM-Kisan',
            'Ration Card', 'Ration Card', 'Ration Card',
            'Pension', 'Pension', 'Pension',
            'PM-Kisan', 'PM-Kisan', 'PM-Kisan',
            'Ration Card', 'Ration Card', 'Ration Card',
            'Pension', 'Pension', 'Pension',
            'Pension', 'Pension', 'Pension',
            'Ration Card', 'Ration Card', 'Ration Card',
            'Pension', 'Pension', 'Pension',
            'PM-Kisan', 'PM-Kisan', 'PM-Kisan',
            'Ration Card', 'Ration Card', 'Ration Card',
            'Pension', 'Pension', 'Pension'
        ]
        
        return pd.DataFrame({'query': queries, 'scheme': labels})

def load_test_data():
    """Load test data from CSV files or create synthetic data"""
    try:
        # Try to load from dataset
        test_data = pd.read_csv('hackathon_dataset/test_queries.csv')
        print(f"Loaded {len(test_data)} test examples")
        return test_data
    except FileNotFoundError:
        print("Test data file not found. Creating synthetic data...")
        # Create synthetic test data
        queries = [
            "I am a farmer how to get pm kisan benefits",
            "need food ration card for my family",
            "pension for my old parents",
            "kisan yojana money not received",
            "ration card online application status",
            "how much pension will I get at age 65",
            "pm kisan scheme eligibility for small farmers",
            "documents needed for new ration card",
            "pension scheme for disabled persons",
            "pm kisan registration process"
        ]
        
        labels = [
            'PM-Kisan', 'Ration Card', 'Pension',
            'PM-Kisan', 'Ration Card', 'Pension',
            'PM-Kisan', 'Ration Card', 'Pension',
            'PM-Kisan'
        ]
        
        return pd.DataFrame({'query': queries, 'scheme': labels})

def train_and_evaluate_models():
    """Train multiple models and evaluate their performance"""
    # Load data
    train_data = load_training_data()
    test_data = load_test_data()
    
    # Prepare features and labels
    X_train = train_data['query']
    y_train = train_data['scheme']
    X_test = test_data['query']
    y_test = test_data['scheme']
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000, 
        stop_words='english',
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )
    
    # Transform texts to TF-IDF vectors
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Define models to evaluate
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SVM': SVC(kernel='linear'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    best_accuracy = 0
    best_model = None
    best_vectorizer = None
    
    # Create directory for training results
    os.makedirs('training_results', exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train the model
        model.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_vec)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': classification_report(y_test, y_pred, output_dict=True),
            'matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        
        # Check if this is the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name
    
    # Save the best model
    if best_model is not None:
        model_data = {
            'vectorizer': vectorizer,
            'model': best_model,
            'model_name': best_model_name,
            'accuracy': best_accuracy,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        joblib.dump(model_data, 'scheme_router_model.joblib')
        print(f"Saved best model ({best_model_name}) with accuracy: {best_accuracy:.4f}")
    
    # Generate evaluation reports and visualizations
    generate_evaluation_reports(results, vectorizer, X_test, y_test)
    
    return results

def generate_evaluation_reports(results, vectorizer, X_test, y_test):
    """Generate evaluation reports and visualizations"""
    # Create a summary report
    summary = []
    for name, result in results.items():
        summary.append({
            'Model': name,
            'Accuracy': result['accuracy'],
            'Precision (weighted)': result['report']['weighted avg']['precision'],
            'Recall (weighted)': result['report']['weighted avg']['recall'],
            'F1-Score (weighted)': result['report']['weighted avg']['f1-score']
        })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('training_results/model_comparison.csv', index=False)
    
    # Plot accuracy comparison
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    plt.bar(models, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('training_results/accuracy_comparison.png')
    plt.close()
    
    # Plot confusion matrix for the best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    cm = results[best_model_name]['matrix']
    labels = sorted(set(y_test))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('training_results/confusion_matrix.png')
    plt.close()
    
    # Print detailed report for the best model
    best_report = results[best_model_name]['report']
    print(f"\nBest Model: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, results[best_model_name]['model'].predict(
        vectorizer.transform(X_test))))
    
    # Save feature importance for Logistic Regression
    if hasattr(results[best_model_name]['model'], 'coef_'):
        feature_names = vectorizer.get_feature_names_out()
        coefs = results[best_model_name]['model'].coef_
        
        # For each class, get top features
        classes = results[best_model_name]['model'].classes_
        feature_importance = {}
        
        for i, class_name in enumerate(classes):
            # Get top 10 features for this class
            top_indices = np.argsort(coefs[i])[-10:][::-1]
            top_features = [(feature_names[j], coefs[i][j]) for j in top_indices]
            feature_importance[class_name] = top_features
        
        # Save to CSV
        importance_df = pd.DataFrame({
            'Class': [cls for cls, features in feature_importance.items() for _ in features],
            'Feature': [feat for features in feature_importance.values() for feat, _ in features],
            'Coefficient': [coef for features in feature_importance.values() for _, coef in features]
        })
        importance_df.to_csv('training_results/feature_importance.csv', index=False)

if __name__ == '__main__':
    print("Training scheme classification models...")
    results = train_and_evaluate_models()
    print("Training completed. Results saved in training_results/ directory.")