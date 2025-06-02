from utils import mask_pii, restore_pii

text = "Hello I’m John Doe. My email is john@example.com and phone is 9876543210."
masked, ents = mask_pii(text)
print("Masked:", masked)
print("Entities:", ents)
print("Restored:", restore_pii(masked, ents))
