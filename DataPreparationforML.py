import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Load the analyzed CSV file into a DataFrame
data = pd.read_csv('emailanalized.csv')

# Separate features and target variable
X = data.drop(['Email Type', 'POS Tags', 'Named Entities', 'Most Common Bigrams'], axis=1)
y = data['Email Type'].map({'Phishing': 1, 'Safe': 0})

# Identify columns with all missing values and fill them with a placeholder value (e.g., 0)
columns_with_all_na = X.columns[X.isna().all()]
X[columns_with_all_na] = X[columns_with_all_na].fillna(0)

# Initialize an imputer with a median filling strategy
imputer = SimpleImputer(strategy='median')

# Impute missing values in X
X_imputed = imputer.fit_transform(X)

# Convert to DataFrame - now the shape should match as we've handled columns with all missing values
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

# Convert any non-numeric columns that may have been overlooked
for column in X.columns:
    X_imputed[column] = pd.to_numeric(X_imputed[column], errors='coerce').fillna(0)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Undersampling the majority class (normal messages)
undersampler = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
X_undersampled, y_undersampled = undersampler.fit_resample(X_scaled, y)

# Split the undersampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_undersampled, y_undersampled, test_size=0.2, random_state=42, stratify=y_undersampled)


# Create and train the models
models = [
    ('Logistic Regression', LogisticRegression()),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC(kernel='linear')),
    ('Random Forest', RandomForestClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('XGBoost', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')),  # Configured to avoid warnings
    ('Naive Bayes', GaussianNB()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Neural Network', MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500))
]

results = []
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append((name, accuracy, precision, recall, f1))

# Print the results
print("\nResults:")
for name, accuracy, precision, recall, f1 in results:
    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")

# Find the winner
winner = max(results, key=lambda x: x[1])
print(f"\nWinner: {winner[0]} (Accuracy: {winner[1]:.4f})")


from sklearn.model_selection import cross_val_score
import xgboost as xgb
import numpy as np

# Assuming X_scaled and y are already defined and prepared as per previous code

# Initialize the XGBoost classifier with the best parameters found (as an example)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=3, learning_rate=0.1)

# Perform cross-validation
cv_scores = cross_val_score(xgb_model, X_scaled, y, cv=5)  # 5-fold cross-validation

# Print the results
print("CV Scores for XGBoost:", cv_scores)
print("Average CV Score for XGBoost:", np.mean(cv_scores))
