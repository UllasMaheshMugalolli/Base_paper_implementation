import json
import re

# ==============================
# LOAD INPUT
# ==============================

with open("../data/entities_with_sentences.json", "r") as f:
    data = json.load(f)


# ==============================
# HELPER FUNCTIONS
# ==============================

def is_close(sentence, e1, e2, max_distance=40):
    pos1 = sentence.find(e1)
    pos2 = sentence.find(e2)

    if pos1 == -1 or pos2 == -1:
        return False

    return abs(pos1 - pos2) <= max_distance


def is_list_sentence(sentence):
    # Detect list-style (very common in your data)
    # No connectors like "in", "of", "due to"
    if not any(x in sentence for x in [" in ", " of ", " due to "]):
        return True
    return False


# ==============================
# RELATION EXTRACTION
# ==============================

def extract_relations(sentence, entities):
    relations = []
    seen = set()

    sentence = sentence.lower()

    # 🚫 Skip list-type sentences
    if is_list_sentence(sentence):
        return []

    entity_texts = [e["text"] for e in entities]

    for e1 in entity_texts:
        for e2 in entity_texts:

            if e1 == e2:
                continue

            relation = None
            head, tail = e1, e2

            # Rule 1: "X in Y"
            if f"{e1} in {e2}" in sentence and is_close(sentence, e1, e2):
                relation = "located_in"

            # Rule 2: "X of Y"
            elif f"{e1} of {e2}" in sentence and is_close(sentence, e1, e2):
                relation = "associated_with"

            # Rule 3: "X due to Y"
            elif f"{e1} due to {e2}" in sentence and is_close(sentence, e1, e2):
                relation = "causes"
                head, tail = e2, e1

            if relation:
                key = (head, relation, tail)

                if key not in seen:
                    relations.append({
                        "head": head,
                        "relation": relation,
                        "tail": tail,
                    })
                    seen.add(key)

    return relations


# ==============================
# PROCESS FULL DATASET
# ==============================

all_relations = []

for patient in data:
    for entry in patient:
        sentence = entry["sentence"]
        entities = entry["entities"]

        if len(entities) < 2:
            continue

        rels = extract_relations(sentence, entities)

        if rels:
            all_relations.extend(rels)


# ==============================
# SAVE OUTPUT
# ==============================

with open("../data/relations.json", "w") as f:
    json.dump(all_relations, f, indent=2)

print("✅ Relations extracted and saved to relations.json")