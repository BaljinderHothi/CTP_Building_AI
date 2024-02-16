from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """
    Dr. Baljinder's new AI research, focusing on natural language processing techniques in 2024, aims to bridge the gap between human linguistic intricacies and machine understanding. 
    This includes tackling challenges like idiomatic expressions, varying syntax across languages, and the seamless integration of slang and neologisms. 
    The goal is to enhance AI's interpretive capabilities to not only comprehend but also generate text that feels authentically human.
    """

print(summarizer(text, max_length=51, min_length=30, do_sample=False))

#this prints: [{'summary_text': "The goal is to enhance AI's capabilities to not only comprehend but also generate text that feels authentically human. This includes tackling challenges like idiomatic expressions and the seamless integration of slang and neologisms."}]
