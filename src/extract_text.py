import pandas as pd
import json

# Load CSV
df = pd.read_csv("graph_triples.csv")

# 🔴 CHANGE THESE
PATIENT_COL = "head"
TEXT_COLUMN = "tail"

# Step 1: get first 10 unique patients
first_10_patients = df[PATIENT_COL].dropna().unique()[:10]

# Step 2: filter all rows of those patients
df_10 = df[df[PATIENT_COL].isin(first_10_patients)]

# Step 3: group into single text per patient
patient_texts = df_10.groupby(PATIENT_COL)[TEXT_COLUMN].apply(
    lambda x: " ".join(x.dropna())
)

# Step 4: convert to list
texts = patient_texts.tolist()

# Step 5: save (IMPORTANT)
with open("patient_texts.json", "w") as f:
    json.dump(texts, f, indent=2)

print("Saved", len(texts), "patients")