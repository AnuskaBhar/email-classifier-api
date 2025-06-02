from fastapi import FastAPI
from pydantic import BaseModel
from models import EmailClassifier
from utils import PIIMasker
import uvicorn

app = FastAPI()

# Initialize components
classifier = EmailClassifier()
masker = PIIMasker()

class EmailRequest(BaseModel):
    input_email_body: str

@app.post("/classify")  # Fixed the typo here
async def classify_email(request: EmailRequest):
    # Mask PII
    masked_email, masked_entities = masker.mask_pii(request.input_email_body)
    
    # Classify email
    category = classifier.predict(masked_email)
    
    return {
        "input_email_body": request.input_email_body,
        "list_of_masked_entities": masked_entities,
        "masked_email": masked_email,
        "category_of_the_email": category
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)