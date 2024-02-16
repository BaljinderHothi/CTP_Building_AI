from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="SamLowe/roberta-base-go_emotions")
print(classifier("I enjoy hearing about surfing. I hope zack talks more about surfing today")) 

#correctly labeled this as [{'label': 'joy', 'score': 0.8331648707389832}]
