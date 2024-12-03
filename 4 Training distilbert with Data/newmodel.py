import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import gc
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MODEL_NAME = "distilbert-base-uncased"
NUM_EPOCHS = 5  # Increased epochs
BATCH_SIZE = 32  # Increased batch size
LEARNING_RATE = 1e-5  # Slightly lower learning rate
MAX_LENGTH = 512
SAVE_PATH = "./models/fine_tuned_phishing_model"


def load_data(file_path):
    logging.info("Loading the dataset...")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded. Shape: {df.shape}")
        df['Email Type'] = df['Email Type'].fillna(-1).astype(int)
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def prepare_dataloaders(df, tokenizer):
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Email Text'].tolist(),
        df['Email Type'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['Email Type']  # Ensure balanced split
    )

    # Calculate class weights
    class_counts = np.bincount(train_labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)

    train_dataset = EmailDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = EmailDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    return train_loader, val_loader


def focal_loss(logits, labels, alpha=0.25, gamma=2):
    ce_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1 - pt) ** gamma * ce_loss).mean()
    return focal_loss


def train_model(model, train_loader, val_loader, optimizer, scheduler, device):
    best_val_f1 = 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = focal_loss(outputs.logits, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Average training loss: {avg_train_loss:.4f}")
        val_f1 = evaluate_model(model, val_loader, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_model(model, "best_model.pth")
            logging.info(f"Best model saved with F1-score: {best_val_f1:.4f}")
    return model


def evaluate_model(model, val_loader, device):
    model.eval()
    val_predictions = []
    val_true_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            val_predictions.extend(preds.cpu().tolist())
            val_true_labels.extend(labels.cpu().tolist())
    logging.info("Validation metrics:")
    report = classification_report(val_true_labels, val_predictions, output_dict=True)
    logging.info(classification_report(val_true_labels, val_predictions))
    return report['macro avg']['f1-score']


def predict_phishing(model, tokenizer, emails, device, temperature=1.5):
    model.eval()
    predictions = []
    probabilities = []
    with torch.no_grad():
        for email in tqdm(emails, desc="Predicting"):
            inputs = tokenizer(email, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits / temperature  # Apply temperature scaling
            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(probs, dim=1)
            predictions.append(preds.item())
            probabilities.append(probs[0, 1].item())  # Probability of being phishing
    return predictions, probabilities


def save_model(model, filename):
    try:
        os.makedirs(SAVE_PATH, exist_ok=True)
        full_path = os.path.join(SAVE_PATH, filename)
        torch.save(model.state_dict(), full_path)
        logging.info(f"Model saved to {full_path}")

        # Save in .bin format as well
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, 'pytorch_model.bin'))

        # Save config.json with model settings
        config = {
            "architectures": ["DistilBertForSequenceClassification"],
            "num_labels": 2,
            "model_type": "distilbert",
            "hidden_size": model.config.hidden_size,
            "vocab_size": model.config.vocab_size,
            "max_position_embeddings": model.config.max_position_embeddings,
            "num_attention_heads": model.config.num_attention_heads,
            "num_hidden_layers": model.config.num_hidden_layers,
            "dropout": model.config.dropout
        }

        config_path = os.path.join(SAVE_PATH, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        logging.info(f"Config saved to {config_path}")

    except Exception as e:
        logging.error(f"Error saving model: {e}")


def load_model(model, filename):
    try:
        full_path = os.path.join(SAVE_PATH, filename)
        model.load_state_dict(torch.load(full_path))
        logging.info(f"Model loaded from {full_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    try:
        df = load_data('cleaned_phishing_data_test.csv')
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(device)

        train_loader, val_loader = prepare_dataloaders(df, tokenizer)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        total_steps = len(train_loader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        model = train_model(model, train_loader, val_loader, optimizer, scheduler, device)

        save_model(model, "final_model.pth")
        tokenizer.save_pretrained(SAVE_PATH)

        # Test the model
        new_emails = [
            "Your account has been locked. Click here to verify your identity immediately!",
            "Meeting scheduled for 3 PM in the conference room."
        ]
        predictions, probabilities = predict_phishing(model, tokenizer, new_emails, device)
        for email, pred, prob in zip(new_emails, predictions, probabilities):
            logging.info(f"Email: {email}")
            logging.info(f"Prediction: {'Phishing' if pred == 1 else 'Safe'}")
            logging.info(f"Phishing Probability: {prob:.2f}\n")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()