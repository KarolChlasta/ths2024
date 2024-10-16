### Developed by Karol Chlasta
### The Hack Summit 2024 Warsaw, Poland
### https://github.com/KarolChlasta/ths2024
### Rate this presentation: https://thehacksummit.com/user.html#!/lecture/THS24-a467/rate

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import joblib
import time
import sys
from io import StringIO

# Define an in-memory string buffer
captured_output = StringIO()

# Redirect standard output temporarily
sys.stdout = captured_output


# Define the path to your Excel file
file_path = "/Users/k.chlasta/Dev/taf-fake/RealData.xls"
model_save_path = "/Users/k.chlasta/Dev/taf-fake/modelTAF.sav"

# Read the data from Excel

data = pd.read_excel(file_path)

# Preprocess the text data
def preprocess_text(text):
    text = text.lower()  # Lowercase
    # We can add other preprocessing steps here (e.g., removing special characters)
    return text

data["Fqdn"] = data["Fqdn"].apply(preprocess_text)

# Separate features (Entry) and target variable (Tribe)
X = data["Fqdn"]
y = data["Tribe"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)


# Create TF-IDF vectorizer for feature extraction
vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed

# Transform text data into numerical features
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

# Selected multi-class classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=500),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Support Vector Machine (Linear Kernel)": SVC(kernel="linear", probability=True),  # Explicit linear kernel for multi-class
}

# Training loop with statistics collection
best_accuracy = 0
best_classifier_name = None
training_stats = []
for name, clf in classifiers.items():
    start_time = time.time()  # Start time for training

    clf.fit(X_train_features, y_train)
    y_pred = clf.predict(X_test_features)

    end_time = time.time()  # End time for training
    training_time = end_time - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    training_stats.append({
        "Classifier": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Training Time (s)": training_time
    })

    print(f"{name} Results:")
    print(f"\tAccuracy: {accuracy:.4f}")
    print(f"\tPrecision: {precision:.4f}")
    print(f"\tRecall: {recall:.4f}")
    print(f"\tF1-Score: {f1:.4f}")
    print(f"\tTraining Time: {training_time:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_precision = precision
        best_recall = recall
        best_f1 = f1
        best_training_time = training_time
        best_classifier_name = name

# Preview correctly classified data (added lines)
correct_indices = (y_test == y_pred).to_numpy().nonzero()[0]
print("\nCorrectly Classified Examples:")
for i in correct_indices[:100]:  # Limit to show top 100
    print(f"\tEntry: {X_test.iloc[i]}")
    print(f"\tPredicted Tribe: {y_pred[i]} (Actual: {y_test.iloc[i]})")
    print("-" * 100)

# Preview incorrectly classified data (added lines)
incorrect_indices = (y_test != y_pred).to_numpy().nonzero()[0]
print("\nIncorrectly Classified Examples:")
for i in incorrect_indices[:100]:  # Limit to show top 100
    print(f"\tEntry: {X_test.iloc[i]}")
    print(f"\tPredicted Tribe: {y_pred[i]} (Actual: {y_test.iloc[i]})")
    print("-" * 100)


# Print the best performing classifier
print(f"\nBest Performing Classifier: {best_classifier_name} with Accuracy: {best_accuracy:.4f}")
print(f"Full {best_classifier_name} Results:")
print(f"\tAccuracy: {best_accuracy:.4f}")
print(f"\tPrecision: {best_precision:.4f}")
print(f"\tRecall: {best_recall:.4f}")
print(f"\tF1-Score: {best_f1:.4f}")
print(f"\tTraining Time: {best_training_time:.4f}")


# Export the model
joblib.dump(best_classifier_name, model_save_path)
print(f"Model saved successfully to: {model_save_path}")

# Restore standard output to the console
sys.stdout = sys.__stdout__

# Write the captured output to a file
with open("/Users/k.chlasta/Dev/modelTAF.txt", "w") as f:
    f.write(captured_output.getvalue())