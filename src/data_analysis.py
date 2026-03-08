"""
This script performs basic exploratory data analysis on the training dataset.

Goals:
1. Load the dataset
2. Inspect dataset structure
3. Analyze NER entity distribution
4. Check if email addresses exist in the dataset
5. Display example sentences containing PERSON entities
"""

import json
import re
from collections import Counter


#1 LOAD DATASET

DATA_PATH = "..\Internship_task_data\data.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dataset loaded successfully")
print("Number of samples:", len(data))


#2 INSPECT DATA STRUCTURE

print("\nExample entry from dataset:\n")
print(data[0])

print("\nKeys present in dataset entries:")
print(data[0].keys())

print("\nExample sentence:")
print(data[0]["sequence"])


#3 ENTITY DISTRIBUTION ANALYSIS

tag_counter = Counter()

for sample in data:
    tag_counter.update(sample["ner_tags"])

print("\nEntity Distribution:")
for tag, count in tag_counter.items():
    print(f"{tag}: {count}")


#4 CHECK IF EMAILS EXIST IN DATASET

email_pattern = r"\S+@\S+\.\S+"
email_count = 0

for sample in data:
    if re.search(email_pattern, sample["sequence"]):
        email_count += 1

print("\nEmail occurrences found:", email_count)

if email_count == 0:
    print("No emails present in the dataset.")


#5 CHECK LANGUAGES

langs = set(sample["lang"] for sample in data)
print(langs)