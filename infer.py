import json
import copy
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding, pipeline

tokenizer = AutoTokenizer.from_pretrained("./tw-model")
model = AutoModelForSequenceClassification.from_pretrained("./tw-model")
pipe = pipeline(model=model, tokenizer=tokenizer, task="text-classification")

def load_json(ja):
    tmp = {"text": [], "label": []}
    for e in ja:
        text = e["text"]
        label = 1 if e["label"].casefold() == "yes" else 0
        tmp["text"].append(text)
        tmp["label"].append(label)
    return tmp

texts = load_json(json.load(open("fit.json")))['text']


# 
# texts = ["✌🏽amor-odio al stretching 🤸🏽‍♀️ ❤️ ", 
        #  "A bit of a shambles but made 19 pull-ups in 1 set…just 🤪 62 altogether - 19,12,11,10,10", 
        #  "this video makes me want to die :) ",
        #  "healthy eating in 12 steps.",
        #  "walk the dog",
        #  "In the time that I go to the bathroom",
        #  "bowling night!"]


from collections import defaultdict


counts = defaultdict(float)
items = []
for text in texts:
    lbl = pipe(text)[0]['label']
    counts[lbl] += 1
    items.append({
        'text': text,
        'label': 1 if lbl == 'LABEL_1' else 0
    })

print(counts)
json.dump(items, open("fitness_classified.json", "w"))