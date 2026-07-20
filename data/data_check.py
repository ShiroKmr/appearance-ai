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


columnFolders = {
    "path_rgb_original": "RGB",
    "path_rgb_masked": "RGB-M",
    "path_mask": "BM",
}


def resolvePath(csvPath, columnName, partition):
    if pd.isna(csvPath) or pd.isna(partition):
        return None

    normalizedPath = str(csvPath).strip().replace("\\", "/")
    csvPath = Path(normalizedPath)

    targetFolder = columnFolders[columnName]
    relativeParts = csvPath.parts[1:]

    if relativeParts and relativeParts[0] in {"train", "test"}:
        return (
            releaseRoot
            / targetFolder
            / Path(*relativeParts)
        )

    return (
        releaseRoot
        / targetFolder
        / str(partition).strip().lower()
        / Path(*relativeParts)
    )


def checkPaths(dataFrame, columnName):
    foundCount = 0
    missingPaths = []

    for rowIndex, row in dataFrame.iterrows():
        csvPath = row[columnName]
        partition = row["partition"]

        resolvedPath = resolvePath(
            csvPath,
            columnName,
            partition,
        )

        if resolvedPath is not None and resolvedPath.is_file():
            foundCount += 1
        else:
            missingPaths.append(
                {
                    "row": rowIndex,
                    "partition": partition,
                    "csvPath": csvPath,
                    "resolvedPath": resolvedPath,
                }
            )

    print(f"\nColumn: {columnName}")
    print(f"Found: {foundCount}/{len(dataFrame)}")
    print(f"Missing: {len(missingPaths)}")

    if missingPaths:
        print("\nFirst missing paths:")

        for missingPath in missingPaths[:10]:
            print(f"Row: {missingPath['row']}")
            print(f"Partition: {missingPath['partition']}")
            print(f"CSV path: {missingPath['csvPath']}")
            print(f"Resolved path: {missingPath['resolvedPath']}")
            print()

    return missingPaths


if annotationsPath is None:
    checkedPaths = "\n".join(
        str(candidatePath)
        for candidatePath in annotationsCandidates
    )

    raise FileNotFoundError(
        "annotations.csv was not found. Checked:\n"
        f"{checkedPaths}"
    )


annotations = pd.read_csv(annotationsPath)

requiredColumns = {
    "partition",
    "path_rgb_original",
    "path_rgb_masked",
    "path_mask",
}

missingColumns = requiredColumns - set(annotations.columns)

if missingColumns:
    raise ValueError(
        "Missing columns in annotations.csv: "
        f"{sorted(missingColumns)}"
    )


print(f"Annotations file: {annotationsPath}")
print(f"Number of rows: {len(annotations)}")

allMissingPaths = []

for columnName in columnFolders:
    missingPaths = checkPaths(
        annotations,
        columnName,
    )

    allMissingPaths.extend(missingPaths)


if not allMissingPaths:
    print("\nAll dataset paths are correct.")
else:
    print(
        "\nTotal missing files: "
        f"{len(allMissingPaths)}"
    )