from transformers import pipeline
from PIL import Image
img = Image.open("mother-earth-day.png")
image_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
print(image_classifier(img))

#result is [{'score': 0.9997088313102722, 'label': 'normal'}, {'score': 0.00029112485935911536, 'label': 'nsfw'}]
