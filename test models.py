import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained('./phishing_model')

# Load the CSV file with email examples and labels
file_path = 'cleaned_phishing_data_test.csv'  # Replace with the path to your CSV file
test_data = pd.read_csv(file_path)

# Extract email texts and labels from the CSV file
test_emails = test_data['Email Text'].astype(str)  # Convert email texts to strings
test_labels = test_data['Email Type']

# Set the batch size according to your available memory
batch_size = 32

# Tokenize and process the test emails in batches
num_correct = 0
num_total = len(test_labels)
num_batches = (num_total + batch_size - 1) // batch_size

print(f"Total test emails: {num_total}")
print(f"Number of batches: {num_batches}")

for i in range(0, num_total, batch_size):
    batch_emails = test_emails[i:i + batch_size]
    batch_labels = test_labels[i:i + batch_size]

    batch_encodings = tokenizer(batch_emails.tolist(), truncation=True, padding=True, max_length=512,
                                return_tensors='pt')

    with torch.no_grad():
        outputs = model(**batch_encodings)
        predictions = torch.argmax(outputs.logits, dim=1)

    num_correct += (predictions == torch.tensor(batch_labels.values)).sum().item()

    print(f"Processed batch {i // batch_size + 1}/{num_batches}")

accuracy = num_correct / num_total
print(f"Accuracy: {accuracy:.2%}")