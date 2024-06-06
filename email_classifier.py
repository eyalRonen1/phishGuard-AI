import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import tkinter as tk
from tkinter import ttk
import torch
from collections import Counter

# Load the dataset
file_path = 'cleaned_phishing_data_test.csv'  # Update the path
data = pd.read_csv(file_path)
#data = data.sample(frac=0.001, random_state=42)

# Fill NaN values with an empty string or another placeholder
data['Email Text'] = data['Email Text'].fillna('')

# Assuming 'Email Text' is the column with the email texts and 'Email Type' is the label column
X = data['Email Text']
y = data['Email Type'].apply(lambda x: 1 if x == 1 else 0)  # Ensure labels are 0 or 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)  # 10% test split

# Check the class distribution before balancing
print("Before balancing:")
print(Counter(y_train))

# Separate the training data by class
train_data = pd.concat([X_train, y_train], axis=1)
phishing_data = train_data[train_data['Email Type'] == 1]
non_phishing_data = train_data[train_data['Email Type'] == 0]

# Determine the minimum number of samples between the two classes
min_samples = min(len(phishing_data), len(non_phishing_data))

# Randomly sample an equal number of samples from each class
phishing_data_balanced = phishing_data.sample(min_samples, random_state=42)
non_phishing_data_balanced = non_phishing_data.sample(min_samples, random_state=42)

# Combine the balanced datasets
train_data_balanced = pd.concat([phishing_data_balanced, non_phishing_data_balanced])

# Update X_train and y_train with the balanced data
X_train = train_data_balanced['Email Text']
y_train = train_data_balanced['Email Type']

# Check the class distribution after balancing
print("After balancing:")
print(Counter(y_train))

# Load the trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained('./phishing_model')

# Tokenize the data
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=512)

# Create PyTorch datasets
class PhishingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PhishingDataset(train_encodings, y_train.tolist())
test_dataset = PhishingDataset(test_encodings, y_test.tolist())

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch'
)


# Save the trained model
model.save_pretrained('./phishing_model')

class PhishingDetectorGUI(tk.Tk):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.initUI()

    def initUI(self):
        self.title("Phishing Email Detector")
        self.geometry("600x400")

        self.email_text = tk.Text(self, height=10, width=60)
        self.email_text.pack(pady=10)

        self.predict_button = ttk.Button(self, text="Predict", command=self.predict)
        self.predict_button.pack(pady=5)

        self.result_label = ttk.Label(self, text="")
        self.result_label.pack()

        self.risk_level_label = ttk.Label(self, text="")
        self.risk_level_label.pack()

        self.confidence_score_label = ttk.Label(self, text="")
        self.confidence_score_label.pack()

    def predict(self):
        email_text = self.email_text.get("1.0", tk.END).strip()
        encoding = self.tokenizer(email_text, truncation=True, padding=True, max_length=512, return_tensors='pt')
        output = self.model(**encoding)
        probabilities = torch.softmax(output.logits, dim=1)
        phishing_probability = probabilities[0][1].item()
        decision_threshold = 0.5  # Adjust this value as needed

        if phishing_probability >= decision_threshold:
            self.result_label.config(text=f"Prediction: Phishing Email\nProbability: {phishing_probability:.2f}")
            self.risk_level_label.config(text="Risk Level: High")
        else:
            self.result_label.config(
                text=f"Prediction: Not a Phishing Email\nProbability: {1 - phishing_probability:.2f}")
            self.risk_level_label.config(text="Risk Level: Low")

        confidence_score = min(5, int(phishing_probability * 5) + 1)
        self.confidence_score_label.config(text=f"Confidence Score: {'⭐' * confidence_score}")

        self.email_text.delete("1.0", tk.END)


# Load the trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained('./phishing_model')

if __name__ == '__main__':
    app = PhishingDetectorGUI(model, tokenizer)
    app.mainloop()