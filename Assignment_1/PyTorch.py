from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pathlib import Path

batch = tokenizer(data, padding=True, truncation=True, max_length=512, return_tensors="pt")

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels= torch.argmax(predictions, dim=1)
    print(labels)

save_dir = Path.cwd()
tokenizer.save_pretrained(save_dir) 
model.save_pretrained(save_dir)

