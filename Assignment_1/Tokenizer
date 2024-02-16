from transformers import AutoTokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)

str = "What is the difference between pokemon and yugioh?"
encoding = tokenizer(str)
print(encoding)

pt_batch = tokenizer(
    [str],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

print(pt_batch)
