# Re-name the season names from Italian to English
from pathlib import Path

import pandas as pd

projectRoot = Path(__file__).resolve().parents[1]
datasetRoot = projectRoot / "assets" / "deep_armocromia"
releaseRoot = datasetRoot / "release"

annotationsCandidates = [
    datasetRoot / "annotations.csv",
    releaseRoot / "annotations.csv",
]

annotationsPath = next(
    (
        candidatePath
        for candidatePath in annotationsCandidates
        if candidatePath.is_file()
    ),
    None,
)

if annotationsPath is None:
    checkedPaths = "\n".join(
        str(candidatePath)
        for candidatePath in annotationsCandidates
    )

    raise FileNotFoundError(
        "annotations.csv was not found. Checked:\n"
        f"{checkedPaths}"
    )


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


requiredColumns = {
    "class",
    "sub_class",
}

missingColumns = requiredColumns - set(annotations.columns)

if missingColumns:
    raise ValueError(
        "Missing columns in annotations.csv: "
        f"{sorted(missingColumns)}"
    )


annotations["class"] = (
    annotations["class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace(seasonMapping)
)

annotations["sub_class"] = (
    annotations["sub_class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace(subSeasonMapping)
)


unknownSeasons = set(annotations["class"]) - set(seasonMapping.values())

unknownSubSeasons = (
    set(annotations["sub_class"])
    - set(subSeasonMapping.values())
)


if unknownSeasons:
    print(
        "Warning: untranslated season values:",
        sorted(unknownSeasons),
    )

if unknownSubSeasons:
    print(
        "Warning: untranslated sub-season values:",
        sorted(unknownSubSeasons),
    )


outputPath = annotationsPath.parent / "annotations_english.csv"

annotations.to_csv(
    outputPath,
    index=False,
)


print(f"Original annotations: {annotationsPath}")
print(f"Translated annotations: {outputPath}")

print("\nSeason distribution:")
print(
    annotations["class"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nSub-season distribution:")
print(
    annotations["sub_class"]
    .value_counts(dropna=False)
    .to_string()
)