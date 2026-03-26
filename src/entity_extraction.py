import json
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# ==============================
# STEP 1: LOAD DATA
# ==============================

with open("../data/patient_texts.json", "r") as f:
    texts = json.load(f)

print(f"Loaded {len(texts)} patient texts")


# ==============================
# STEP 2: LOAD MODEL
# ==============================

model_name = "d4data/biomedical-ner-all"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="none"   # we control merging
)


# ==============================
# STEP 3: CONFIG
# ==============================

VALID_TYPES = {
    "Disease_disorder",
    "Sign_symptom",
    "Biological_structure",
    "Therapeutic_procedure",
    "Medication"
}


# ==============================
# STEP 4: SENTENCE SPLITTING
# ==============================

def split_sentences(text):
    # split on ., ; , newline (important for clinical text)
    sentences = re.split(r'[.;,\n]+', text)

    # keep everything meaningful (light filtering only)
    cleaned = [s.strip() for s in sentences if len(s.strip()) > 3]

    return cleaned


# ==============================
# STEP 5: MERGE SUBWORDS
# ==============================

def merge_subwords(entities):
    merged = []
    i = 0

    while i < len(entities):
        current = entities[i]

        word = current["word"]
        label = current["entity"]
        start = current["start"]
        end = current["end"]

        while i + 1 < len(entities) and entities[i + 1]["word"].startswith("##"):
            next_ent = entities[i + 1]
            word += next_ent["word"].replace("##", "")
            end = next_ent["end"]
            i += 1

        merged.append({
            "word": word.replace("##", ""),
            "entity": label,
            "start": start,
            "end": end
        })

        i += 1

    return merged


# ==============================
# STEP 6: NORMALIZE LABELS
# ==============================

def normalize_labels(entities):
    normalized = []

    for ent in entities:
        label = ent["entity"]

        if "-" in label:
            label = label.split("-")[-1]

        normalized.append({
            "word": ent["word"],
            "entity_group": label,
            "start": ent["start"],
            "end": ent["end"]
        })

    return normalized


# ==============================
# STEP 7: VALIDATION FILTER
# ==============================

def is_valid_entity(word):
    word = word.strip()

    if len(word) < 3:
        return False

    INVALID_WORDS = {"my", "he", "she", "it"}
    if word.lower() in INVALID_WORDS:
        return False

    if not any(c.isalpha() for c in word):
        return False

    return True


# ==============================
# STEP 8: CLEAN ENTITIES
# ==============================

def clean_entities(entities):
    cleaned = []

    for ent in entities:
        word = ent["word"].strip()
        label = ent["entity_group"]

        if label not in VALID_TYPES:
            continue

        if not is_valid_entity(word):
            continue

        cleaned.append({
            "text": word.lower(),
            "type": label,
            "start": ent["start"],
            "end": ent["end"]
        })

    return cleaned


# ==============================
# STEP 9: SMART MERGE (NO OVER-MERGE)
# ==============================

def merge_entities(entities):
    merged = []
    i = 0

    while i < len(entities):
        current = entities[i]

        word = current["text"]
        label = current["type"]
        start = current["start"]
        end = current["end"]

        j = i + 1

        while j < len(entities):
            next_ent = entities[j]

            # stop if label changes
            if next_ent["type"] != label:
                break

            # stop if gap too large
            if next_ent["start"] - end > 2:
                break

            # stop if too long
            if len(word.split()) >= 3:
                break

            word += " " + next_ent["text"]
            end = next_ent["end"]
            j += 1

        merged.append({
            "text": word,
            "type": label,
            "start": start,
            "end": end
        })

        i = j

    return merged


# ==============================
# STEP 10: REMOVE BAD MERGES
# ==============================

def remove_bad_merges(entities):
    cleaned = []

    for ent in entities:
        words = ent["text"].split()

        if len(words) > 4:
            continue

        if any(w in ["and", "or"] for w in words):
            continue

        cleaned.append(ent)

    return cleaned


# ==============================
# STEP 11: DEDUPLICATION
# ==============================

def deduplicate(entities):
    unique = []
    seen = set()

    for ent in entities:
        key = (ent["text"], ent["type"], ent["start"])

        if key not in seen:
            unique.append(ent)
            seen.add(key)

    return unique


# ==============================
# STEP 12: MAIN PIPELINE
# ==============================

all_entities = []

for idx, text in enumerate(texts):
    print(f"\nProcessing patient {idx + 1}...")

    patient_data = []
    sentences = split_sentences(text)

    for sentence in sentences:
        try:
            raw = ner(sentence)

            step1 = merge_subwords(raw)
            step2 = normalize_labels(step1)
            step3 = clean_entities(step2)
            step4 = merge_entities(step3)
            step5 = remove_bad_merges(step4)
            final_entities = deduplicate(step5)

            if final_entities:
                patient_data.append({
                    "sentence": sentence,
                    "entities": final_entities
                })

        except Exception as e:
            print(f"Error in sentence: {e}")

    all_entities.append(patient_data)


# ==============================
# STEP 13: SAVE OUTPUT
# ==============================

with open("../data/entities_with_sentences.json", "w") as f:
    json.dump(all_entities, f, indent=2)

print("\n✅ Entity extraction complete with sentence mapping.")