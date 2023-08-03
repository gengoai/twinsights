import json
import copy
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_metric
import evaluate


def load_json(ja):
    tmp = {"text": [], "label": []}
    for e in ja:
        text = e["text"]
        label = 1 if e["label"].casefold() == "yes" else 0
        tmp["text"].append(text)
        tmp["label"].append(label)
    return tmp

safe = load_json(json.load(open("safe.json")))
tw = load_json(json.load(open("tw.json")))


print(len(safe["text"]))
print(len(tw["text"]))

combined = copy.copy(safe)
combined["text"].extend(tw["text"])
combined["label"].extend(tw["label"])

num = len(combined["text"])
dataset = Dataset.from_dict(combined)

train = dataset.shuffle(seed=10).select(range(int(0.8 * num)))
test = dataset.shuffle(seed=10).select(range(int(0.8 * num), num))

print(train)
print(test)
# items = []
# for e in test:
#     items.append(e)

# json.dump(items, open("testing.json", "w"))
# exit()

# transformer_model = "distilroberta-base"
# transformer_model = "bert-base-uncased"
# transformer_model = "distilbert-base-uncased"
transformer_model = "roberta-base"
#transformer_model = "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(transformer_model)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized = train.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    transformer_model, num_labels=2
)

p = evaluate.load("precision")
r = evaluate.load("recall")
f = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "precision": p.compute(predictions=predictions, references=labels),
        "recall": r.compute(predictions=predictions, references=labels),
        "F1": f.compute(predictions=predictions, references=labels),
    }


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    use_mps_device=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    eval_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
tokenizer.save_pretrained("tw-model")
model.save_pretrained("tw-model")

tokenized = test.map(preprocess_function, batched=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    eval_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
print(trainer.evaluate())
