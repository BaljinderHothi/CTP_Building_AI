from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier(
    "This is a review on a brand new type of oreo cookie",
    candidate_labels=['games','snack','food','kitchen supply']
)
print(res)

#result is {'sequence': 'This is a review on a brand new type of oreo cookie', 'labels': ['food', 'snack', 'kitchen supply', 'games'], 'scores': [0.8161830902099609, 0.17804944515228271, 0.0032798994798213243, 0.002487555844709277]}
