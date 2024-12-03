# Phishing Email Detection using Machine Learning
#
# This script performs the following tasks:
# 1. Loads and preprocesses email data from a CSV file.
# 2. Balances the dataset to have equal numbers of phishing and safe emails.
# 3. Trains and evaluates multiple machine learning models using grid search or randomized search for hyperparameter tuning.
# 4. Compares the performance of different models using various metrics (accuracy, precision, recall, F1-score).
# 5. Identifies the best performing model based on F1-score.
#
# The script uses a variety of classifiers including Logistic Regression, KNN, Random Forest, Gradient Boosting, XGBoost, 
# Naive Bayes, Decision Tree, and Neural Network. It runs the process multiple times with different random states to ensure 
# consistency in results.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def load_and_preprocess_data(file_path, sample_size=None):
    # Load data from CSV file, preprocess it, and return features and labels.
    #
    # Args:
    # file_path (str): Path to the CSV file.
    # sample_size (int, optional): Number of samples to use (for testing purposes).
    #
    # Returns:
    # tuple: Preprocessed features (X) and labels (y).

    # Load data
    data = pd.read_csv(file_path)
    if sample_size:
        data = data.sample(n=sample_size, random_state=42)

    # Separate features and labels
    X = data.drop(['Email Type', 'POS Tags', 'Named Entities', 'Most Common Bigrams'], axis=1)
    y = data['Email Type'].map({'Phishing': 1, 'Safe': 0})

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled.astype(np.float32), y.values.astype(np.int32)


def balance_classes(X, y):
    # Balance the dataset by undersampling the majority class.
    #
    # Args:
    # X (numpy.ndarray): Features.
    # y (numpy.ndarray): Labels.
    #
    # Returns:
    # tuple: Balanced features and labels.

    class_0, class_1 = np.bincount(y)
    min_class = min(class_0, class_1)

    idx_class_0 = np.where(y == 0)[0]
    idx_class_1 = np.where(y == 1)[0]

    idx_class_0 = np.random.choice(idx_class_0, min_class, replace=False)
    idx_class_1 = np.random.choice(idx_class_1, min_class, replace=False)

    under_sample_idx = np.concatenate([idx_class_0, idx_class_1])

    X_balanced = X[under_sample_idx]
    y_balanced = y[under_sample_idx]

    return X_balanced, y_balanced


def evaluate_model(model, X_test, y_test):
    # Evaluate the model performance.
    #
    # Args:
    # model: Trained model.
    # X_test (numpy.ndarray): Test features.
    # y_test (numpy.ndarray): Test labels.
    #
    # Returns:
    # dict: Performance metrics.

    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def train_and_evaluate(X, y, random_states=[42, 123]):
    # Train and evaluate multiple models with different random states.
    #
    # Args:
    # X (numpy.ndarray): Features.
    # y (numpy.ndarray): Labels.
    # random_states (list): List of random states to use.
    #
    # Returns:
    # list: Results for each model and random state.

    # Define models and their hyperparameters
    models = [
        ('Logistic Regression', LogisticRegression(max_iter=1000), {
            'C': [0.1, 1, 10],
            'penalty': ['l2']
        }),
        ('KNN', KNeighborsClassifier(), {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }),
        ('Random Forest', RandomForestClassifier(), {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }),
        ('Gradient Boosting', GradientBoostingClassifier(), {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 4]
        }),
        ('XGBoost', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 4, 5],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }),
        ('Naive Bayes', GaussianNB(), {}),
        ('Decision Tree', DecisionTreeClassifier(), {
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }),
        ('Neural Network', MLPClassifier(max_iter=1000), {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu'],
            'alpha': [0.0001, 0.001]
        })
    ]

    best_results = []

    for random_state in random_states:
        print(f"\nRandom State: {random_state}")

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

        # Balance the training data
        X_train_balanced, y_train_balanced = balance_classes(X_train, y_train)

        print(
            f"Training set - Total: {len(y_train_balanced)}, Safe: {np.sum(y_train_balanced == 0)}, Phishing: {np.sum(y_train_balanced == 1)}")
        print(f"Test set - Total: {len(y_test)}, Safe: {np.sum(y_test == 0)}, Phishing: {np.sum(y_test == 1)}")

        for name, model, param_grid in models:
            print(f"Training {name}...")
            # Use RandomizedSearchCV for XGBoost, GridSearchCV for others
            if name == 'XGBoost':
                search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=10, cv=3, scoring='f1',
                                            n_jobs=-1, random_state=random_state)
            else:
                search = GridSearchCV(model, param_grid, cv=3, scoring='f1', n_jobs=-1)

            # Fit the model
            search.fit(X_train_balanced, y_train_balanced)

            # Evaluate the best model
            best_model = search.best_estimator_
            scores = evaluate_model(best_model, X_test, y_test)
            best_results.append((name, scores, search.best_params_, random_state))

    return best_results


# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_and_preprocess_data('emailanalized.csv', sample_size=10000)  # Adjust sample_size as needed

    # Train and evaluate models
    results = train_and_evaluate(X, y)

    # Print results
    for name, scores, best_params, random_state in results:
        print(f"\n{name} Results (Random State: {random_state}):")
        print(f"Best Parameters: {best_params}")
        print(f"Accuracy: {scores['accuracy']:.4f}")
        print(f"Precision: {scores['precision']:.4f}")
        print(f"Recall: {scores['recall']:.4f}")
        print(f"F1-score: {scores['f1']:.4f}")
        print("Confusion Matrix:")
        print(scores['confusion_matrix'])

    # Find the best model based on F1-score
    best_model = max(results, key=lambda x: x[1]['f1'])
    print(f"\nBest Overall Model: {best_model[0]} (Random State: {best_model[3]}, F1-score: {best_model[1]['f1']:.4f})")
    print(f"Best Parameters: {best_model[2]}")