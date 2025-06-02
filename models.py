from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

class EmailClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.labels = ["Incident", "Request", "Change", "Problem"]
        self.label_encoder.fit(self.labels)
        
        # Initialize model with proper label mappings
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            id2label={i: label for i, label in enumerate(self.labels)},
            label2id={label: i for i, label in enumerate(self.labels)}
        )

    def load_dataset(self, csv_path="combined_emails_with_natural_pii.csv"):
        # Load CSV with pandas
        df = pd.read_csv(csv_path)
        
        # Encode labels
        texts = df["text"].tolist()
        labels = self.label_encoder.transform(df["label"].tolist())
        
        return Dataset.from_dict({
            "text": texts,
            "label": labels
        })

    def train(self, csv_path="emails.csv"):
        dataset = self.load_dataset(csv_path)
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            lambda x: self.tokenizer(x["text"], truncation=True, padding="max_length", max_length=512),
            batched=True
        )
        
        # Training configuration
        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=8,
            num_train_epochs=3,
            save_strategy="no"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        trainer.train()
        self.save_model()

    def save_model(self, path="./email_classifier"):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        return self.labels[predicted_class]