"""
Zero-Shot PII Extraction and Masking using LLM

This script evaluates a zero-shot LLM approach for detecting and masking
Personally Identifiable Information (PII) in text.

The model extracts PERSON names and EMAIL addresses using prompt-based
generation. Extracted entities are then used to mask the original text.

Pipeline:
1. Load dataset
2. Generate PII entities using an LLM
3. Parse model output into structured JSON
4. Mask detected entities in text
5. Compute evaluation metrics (Accuracy, Precision, Recall, F1, FPR, FNR)
6. Save masked outputs
"""

import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH = "/kaggle/input/datasets/abdullahshheikh/pii-masking-augmented/train_processed.json"

MAX_SAMPLES = 400
MASKED_FILE = "masked_text_output.json"


# -----------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------

def load_model():
    """Load tokenizer and LLM model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model


# -----------------------------------------------------------
# ENTITY EXTRACTION
# -----------------------------------------------------------

def generate_extraction(tokenizer, model, text):
    """
    Prompt the LLM to extract PII entities from text.
    """

    prompt = f"""
Extract PII entities from the following text.

Rules:
- Identify PERSON names and EMAIL addresses only.
- Return ONLY a JSON list of objects.

Example:
Input: Contact Sarah Jenkins at sarah.j@email.com.
Output: [{"entity": "Sarah Jenkins", "type": "NAME"}, {"entity": "sarah.j@email.com", "type": "EMAIL"}]

Text: {text}

JSON:
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

    return response


def parse_json(text):
    """
    Extract JSON list from LLM response.
    """

    try:
        start = text.find("[")
        end = text.rfind("]") + 1

        if start != -1:
            return json.loads(text[start:end])

    except Exception:
        pass

    return []


# -----------------------------------------------------------
# TEXT MASKING
# -----------------------------------------------------------

def mask_text(text, entities):
    """
    Replace detected PII entities with placeholders.
    """

    if not isinstance(entities, list):
        return text

    valid_entities = [
        item for item in entities
        if isinstance(item, dict) and item.get("entity")
    ]

    sorted_entities = sorted(
        valid_entities,
        key=lambda x: len(str(x["entity"])),
        reverse=True
    )

    masked_text = text

    for item in sorted_entities:

        entity = str(item.get("entity"))
        label = item.get("type", "PII")

        if entity.strip():
            masked_text = re.sub(
                re.escape(entity),
                f"[{label}]",
                masked_text
            )

    return masked_text


# -----------------------------------------------------------
# METRIC CALCULATION
# -----------------------------------------------------------

def calculate_detailed_metrics(results, ground_truth):
    """
    Compute classification metrics for PER and EMAIL entities.
    """

    stats = {
        "PER": {"tp":0,"fp":0,"tn":0,"fn":0},
        "EMAIL": {"tp":0,"fp":0,"tn":0,"fn":0}
    }

    for i, item in enumerate(results):

        actual_sample = ground_truth[i]
        predicted_entities = item.get("entities_found", [])

        actual_per = set(
            t.lower() for t, tag in zip(
                actual_sample["tokens"],
                actual_sample["ner_tags"]
            ) if "PER" in tag
        )

        actual_email = set(
            t.lower() for t, tag in zip(
                actual_sample["tokens"],
                actual_sample["ner_tags"]
            ) if "EMAIL" in tag
        )

        pred_per, pred_email = set(), set()

        for e in predicted_entities:

            if not isinstance(e, dict):
                continue

            value = str(e.get("entity") or "").lower()
            label = str(e.get("type") or "").upper()

            if "NAME" in label or "PER" in label:
                pred_per.add(value)

            elif "EMAIL" in label:
                pred_email.add(value)

        tokens = set(t.lower() for t in actual_sample["tokens"])

        for token in tokens:

            for tag, actual_set, pred_set in [
                ("PER", actual_per, pred_per),
                ("EMAIL", actual_email, pred_email)
            ]:

                is_actual = token in actual_set
                is_pred = any(token in p for p in pred_set)

                if is_actual and is_pred:
                    stats[tag]["tp"] += 1
                elif not is_actual and is_pred:
                    stats[tag]["fp"] += 1
                elif is_actual and not is_pred:
                    stats[tag]["fn"] += 1
                else:
                    stats[tag]["tn"] += 1

    report = {}

    for tag in ["PER","EMAIL"]:

        s = stats[tag]

        precision = s["tp"]/(s["tp"]+s["fp"]) if (s["tp"]+s["fp"]) else 0
        recall = s["tp"]/(s["tp"]+s["fn"]) if (s["tp"]+s["fn"]) else 0
        accuracy = (s["tp"]+s["tn"])/(s["tp"]+s["tn"]+s["fp"]+s["fn"])
        fpr = s["fp"]/(s["fp"]+s["tn"]) if (s["fp"]+s["tn"]) else 0
        fnr = s["fn"]/(s["fn"]+s["tp"]) if (s["fn"]+s["tp"]) else 0
        f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) else 0

        report[tag] = {
            "Accuracy": round(accuracy,4),
            "Precision": round(precision,4),
            "Recall": round(recall,4),
            "F1-Score": round(f1,4),
            "FPR": round(fpr,4),
            "FNR": round(fnr,4)
        }

    return report


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

        response = generate_extraction(tokenizer, model, text)
        entities = parse_json(response)

        results.append({
            "original": text,
            "masked": mask_text(text, entities),
            "entities_found": entities
        })

    with open(MASKED_FILE, "w") as f:
        json.dump(results, f, indent=2)

    metrics = calculate_detailed_metrics(results, samples)

    print("\nEvaluation Results")
    print(f"{'Metric':<15} | {'PER':<10} | {'EMAIL':<10}")
    print("-"*40)

    for m in ["Accuracy","Precision","Recall","F1-Score","FPR","FNR"]:
        print(f"{m:<15} | {metrics['PER'][m]:<10.4f} | {metrics['EMAIL'][m]:<10.4f}")


if __name__ == "__main__":
    main()