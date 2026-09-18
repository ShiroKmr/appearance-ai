# Re-name the season names from Italian to English
from pathlib import Path
import pandas as pd

annotationsPath = Path("assets/deep_armocromia/release/annotations.csv")
releaseRoot = Path("assets/deep_armocromia/release")

seasonMapping = {
    "primavera": "spring",
    "estate": "summer",
    "autunno": "autumn",
    "inverno": "winter",
}

subSeasonMapping = {
    "chiara": "light",
    "calda": "warm",
    "brillante": "bright",
    "fredda": "cool",
    "soft": "soft",
    "profonda": "deep",
}


annotations = pd.read_csv(annotationsPath)

requiredColumns = {"class", "sub_class"}

annotations["class"] = annotations["class"].astype(str).strip().lower().replace(seasonMapping)
annotations["sub_class"] = annotations["sub_class"].astype(str).strip().lower().replace(subSeasonMapping)

outputPath = annotationsPath.parent / "annotations_english.csv"

annotations.to_csv(outputPath,index=False)

print(f"Original annotations: {annotationsPath}")
print(f"Translated annotations: {outputPath}")

print("\nSeason distribution:")
print(annotations["class"].value_counts(dropna=False).to_string())

print("\nSub-season distribution:")
print(annotations["sub_class"].value_counts(dropna=False).to_string())