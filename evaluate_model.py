import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_test_data():
    """Create synthetic test data if test file doesn't exist"""
    queries = [
        "how to apply for pm kisan scheme",
        "need ration card for my family",
        "pension for senior citizen",
        "pm kisan beneficiary status",
        "ration card eligibility criteria",
        "old age pension application",
        "kisan samman nidhi registration",
        "food security card documents",
        "pension scheme for widows",
        "farmer financial assistance",
        "ration card renewal process",
        "national pension scheme details"
    ]
    
    labels = [
        'PM-Kisan', 'Ration Card', 'Pension',
        'PM-Kisan', 'Ration Card', 'Pension',
        'PM-Kisan', 'Ration Card', 'Pension',
        'PM-Kisan', 'Ration Card', 'Pension'
    ]
    
    return pd.DataFrame({'query': queries, 'scheme': labels})

def evaluate_model_on_new_data(model_path, test_data_path):
    """Evaluate the trained model on new test data"""
    try:
        # Load the trained model
        model_data = joblib.load(model_path)
        vectorizer = model_data['vectorizer']
        model = model_data['model']
        model_name = model_data['model_name']
        
        print(f"Loaded {model_name} model trained on {model_data['training_date']}")
        print(f"Model accuracy during training: {model_data['accuracy']:.4f}")
        
        # Load test data or create if doesn't exist
        if os.path.exists(test_data_path):
            test_data = pd.read_csv(test_data_path)
            print(f"Loaded test data from {test_data_path}")
        else:
            print(f"Test data file {test_data_path} not found. Creating synthetic test data...")
            test_data = create_test_data()
            # Save the synthetic test data for future use
            test_data.to_csv(test_data_path, index=False)
            print(f"Created and saved synthetic test data to {test_data_path}")
        
        X_test = test_data['query']
        y_test = test_data['scheme']
        
        # Transform test data
        X_test_vec = vectorizer.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_vec)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy on test data: {accuracy:.4f}")
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        labels = sorted(set(y_test))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name} (Test Data)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('training_results/confusion_matrix_test_data.png')
        plt.close()
        
        # Print some example predictions
        print("\nExample Predictions:")
        print("-" * 60)
        for i, (true, pred, text) in enumerate(zip(y_test, y_pred, X_test)):
            if i < 5:  # Show first 5 examples
                status = "✓" if true == pred else "✗"
                print(f"{status} '{text}' -> True: {true}, Predicted: {pred}")
        
        return accuracy
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        import traceback
        traceback.print_exc()
        return None

def cross_validate_model(model_path, k_folds=5):
    """Perform cross-validation on the model"""
    try:
        # Load the trained model
        model_data = joblib.load(model_path)
        vectorizer = model_data['vectorizer']
        model = model_data['model']
        
        # Load or create training data
        try:
            train_data = pd.read_csv('hackathon_dataset/sample_queries.csv')
        except:
            print("Training data not found. Creating synthetic data for cross-validation...")
            train_data = create_test_data()  # Reuse the function for simplicity
        
        X = train_data['query']
        y = train_data['scheme']
        
        # Transform data
        X_vec = vectorizer.transform(X)
        
        # Perform cross-validation
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X_vec, y, cv=k_folds)
        
        print(f"\nCross-Validation Results ({k_folds}-fold):")
        print(f"Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"Individual Fold Scores: {cv_scores}")
        
        return cv_scores
        
    except Exception as e:
        print(f"Error in cross-validation: {e}")
        return None

if __name__ == '__main__':
    # Create training_results directory if it doesn't exist
    os.makedirs('training_results', exist_ok=True)
    
    # Evaluate the model on test data
    accuracy = evaluate_model_on_new_data('scheme_router_model.joblib', 'hackathon_dataset/test_queries.csv')
    
    # Perform cross-validation
    cv_scores = cross_validate_model('scheme_router_model.joblib')
    
    if accuracy is not None:
        print(f"\nModel evaluation completed. Accuracy on test data: {accuracy:.4f}")
    else:
        print("Model evaluation failed.")