from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmailClassifier:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
        self.labels = ["Incident", "Request", "Change", "Problem"]
    
    def train(self, train_texts, train_labels):
        # Implement training logic if needed
        pass
    
    def predict(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        return self.labels[predicted_class]