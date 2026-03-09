"""
Zero-Shot PII Masking using an LLM

This script evaluates a direct masking approach for detecting and masking
Personally Identifiable Information (PII) using a large language model.

The model receives raw text and directly returns the masked version where:
- Person names are replaced with [NAME]
- Email addresses are replaced with [EMAIL]

Pipeline:
1. Load dataset
2. Generate masked text using an LLM prompt
3. Compare masked output with ground-truth labels
4. Compute evaluation metrics (Accuracy, Precision, Recall, F1, FPR, FNR)
5. Save masked outputs
"""

import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "/kaggle/input/datasets/abdullahshheikh/pii-masking-augmented/train_processed.json"

MAX_SAMPLES = 100
OUTPUT_FILE = "llm_direct_mask_output.json"


# -----------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------

def load_model():
    """Load tokenizer and language model."""
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


# -----------------------------------------------------------
# LLM MASKING
# -----------------------------------------------------------

def generate_masked_text(tokenizer, model, text):
    """
    Generate masked text using the LLM.
    """

    prompt = f"""
Mask all Personally Identifiable Information in the text.

Rules:
- Replace person names with [NAME]
- Replace email addresses with [EMAIL]
- Return only the masked text

Example:
Input: Contact Sarah Jenkins at sarah.j@email.com.
Output: Contact [NAME] at [EMAIL].

Text: {text}

Masked Text:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.01,
            do_sample=False
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Masked Text:" in response:
        return response.split("Masked Text:")[-1].strip()

    return response.strip()


# -----------------------------------------------------------
# METRIC CALCULATION
# -----------------------------------------------------------

def calculate_metrics(results):
    """
    Compute evaluation metrics for PERSON and EMAIL masking.
    """

    stats = {
        "PER": {"tp":0,"fp":0,"tn":0,"fn":0},
        "EMAIL": {"tp":0,"fp":0,"tn":0,"fn":0}
    }

    for item in results:

        tokens = item["tokens"]
        labels = item["ner_tags"]
        masked = item["masked_prediction"]

        for token, label in zip(tokens, labels):

            for cat, mask in [("PER", "[NAME]"), ("EMAIL", "[EMAIL]")]:

                actual = cat in label
                predicted = mask in masked and token not in masked

                if actual and predicted:
                    stats[cat]["tp"] += 1
                elif not actual and predicted:
                    stats[cat]["fp"] += 1
                elif actual and not predicted:
                    stats[cat]["fn"] += 1
                else:
                    stats[cat]["tn"] += 1

    report = []

    for cat in ["PER", "EMAIL"]:

        s = stats[cat]

        precision = s["tp"]/(s["tp"]+s["fp"]) if (s["tp"]+s["fp"]) else 0
        recall = s["tp"]/(s["tp"]+s["fn"]) if (s["tp"]+s["fn"]) else 0
        accuracy = (s["tp"]+s["tn"])/(s["tp"]+s["tn"]+s["fp"]+s["fn"])
        fpr = s["fp"]/(s["fp"]+s["tn"]) if (s["fp"]+s["tn"]) else 0
        fnr = s["fn"]/(s["fn"]+s["tp"]) if (s["fn"]+s["tp"]) else 0
        f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) else 0

        report.append({
            "Entity": cat,
            "Accuracy": round(accuracy,4),
            "Precision": round(precision,4),
            "Recall": round(recall,4),
            "F1": round(f1,4),
            "FPR": round(fpr,4),
            "FNR": round(fnr,4)
        })

    return pd.DataFrame(report)


# -----------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------

def main():

    with open(DATA_PATH) as f:
        data = json.load(f)

    tokenizer, model = load_model()

    samples = [x for x in data if "@" in x["sequence"]][:MAX_SAMPLES]

    results = []

    for item in tqdm(samples):

        text = item["sequence"]

        masked_text = generate_masked_text(
            tokenizer,
            model,
            text
        )

        results.append({
            "tokens": item["tokens"],
            "ner_tags": item["ner_tags"],
            "original": text,
            "masked_prediction": masked_text
        })

    metrics = calculate_metrics(results)

    print(metrics.to_string(index=False))

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()