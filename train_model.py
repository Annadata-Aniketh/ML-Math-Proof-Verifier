import pandas as pd
import numpy as np
import time
import joblib  # <-- ADD THIS IMPORT
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

def main():
    """
    Loads, cleans, classifies, and **SAVES** the best model for the Streamlit app.
    """
    
    # --- 1. Load Data ---
    file_path = "math_proof_verifier_dataset_14000_13.csv" #
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please make sure the dataset file is in the same directory as this script.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    # --- 1. Data Cleaning ---
    data = df[['proof_text', 'is_correct']].copy()
    data.dropna(subset=['proof_text', 'is_correct'], inplace=True)
    data['is_correct'] = data['is_correct'].astype(int)

    X = data['proof_text']
    y = data['is_correct']

    print("Data loaded and cleaned successfully.")
    print(f"Total samples: {len(y)}")
    
    # --- Data Balance Check ---
    print("\n--- Data Balance Check ---")
    balance = y.value_counts(normalize=True)
    print("Distribution of target variable 'is_correct':")
    print(balance)
    is_balanced = balance.min() > 0.4 and balance.max() < 0.6
    if is_balanced:
        print("Dataset is well-balanced. This is excellent.")
    else:
        print("Dataset is imbalanced. 'class_weight' is important.")
    print("-" * 30)
    
    # --- 1. Data Splitting ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("-" * 30)

    # --- 2. Model Selection & Justification (with Fine-Tuning) ---
    print("--- 2. Model Selection: Fine-Tuning with GridSearchCV ---")
    print("This will prevent overfitting by finding the best hyperparameters.")
    print("We are now comparing 4 models: LR, SVC, RF, and Gradient Boosting.")
    
    # Define the pipelines
    pipelines = {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(class_weight='balanced', solver='liblinear', max_iter=2000, random_state=42))
        ]),
        'Linear SVC': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LinearSVC(class_weight='balanced', max_iter=2000, random_state=42, dual=True))
        ]),
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', GradientBoostingClassifier(random_state=42))
        ])
    }
    
    # Define the "grid" of parameters to search for each model.
    param_grids = {
        'Logistic Regression': {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__min_df': [5],
            'tfidf__stop_words': ['english', None],
            'clf__C': [1.0, 10.0]
        },
        'Linear SVC': {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__min_df': [5],
            'tfidf__stop_words': ['english', None],
            'clf__C': [1.0, 10.0]
        },
        'Random Forest': {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__min_df': [5],
            'tfidf__stop_words': ['english', None],
            'clf__n_estimators': [100, 200]
        },
        'Gradient Boosting': {
            'tfidf__ngram_range': [(1, 2)],
            'tfidf__min_df': [5],
            'tfidf__stop_words': ['english', None],
            'clf__n_estimators': [100, 200],
            'clf__learning_rate': [0.1]
        }
    }

    print("This process will be slow as it runs many models...")
    print("-" * 30)

    # --- 3. Model Training (with GridSearch) ---
    best_overall_score = 0.0
    best_model = None
    best_model_name = ""

    for name in pipelines.keys():
        print(f"--- 3. Fine-Tuning {name} ---")
        start_time = time.time()
        
        grid_search = GridSearchCV(
            pipelines[name], 
            param_grids[name], 
            cv=5, 
            scoring='accuracy', 
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        tune_time = time.time() - start_time
        print(f"Tuning complete in {tune_time:.2f}s")
        
        best_cv_score = grid_search.best_score_
        print(f"Best CV Accuracy for {name}: {best_cv_score * 100:.2f}%")
        print(f"Best Parameters found for {name}:")
        print(grid_search.best_params_)
        print("-" * 30)

        if best_cv_score > best_overall_score:
            best_overall_score = best_cv_score
            # We save the *entire pipeline* (TF-IDF + Classifier)
            best_model = grid_search.best_estimator_ 
            best_model_name = name

    print(f"\n--- Best Model Chosen ---")
    print(f"The best model after fine-tuning is: {best_model_name}")
    print(f"It achieved a {best_overall_score * 100:.2f}% cross-validated accuracy.")
    print("-" * 30)


    # --- 4. Model Evaluation (Detailed) ---
    # (This section runs as before, to show you the test results)
    print("\n--- 4. Model Evaluation (Detailed Report for Best Tuned Model) ---")
    y_pred_best = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_best)
    print(f"\nFinal Test Set Accuracy: {test_accuracy * 100:.2f}%")
    print("\nFull Classification Report:")
    report_text = classification_report(
        y_test, 
        y_pred_best, 
        target_names=['Incorrect (0)', 'Correct (1)']
    )
    print(report_text)
    
    # --- 5 & 6 are skipped as we don't need the CSV for the app ---

    # --- 7. NEW: Saving the Best Model for the App ---
    print("\n--- 7. Saving Model for Streamlit App ---")
    # We save the entire 'best_model' pipeline, which includes
    # the TfidfVectorizer and the classifier (e.g., LinearSVC)
    model_filename = "proof_verifier_pipeline.joblib"
    try:
        joblib.dump(best_model, model_filename)
        print(f"Successfully saved the best model pipeline to '{model_filename}'")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    print("----------------------------------")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    main()