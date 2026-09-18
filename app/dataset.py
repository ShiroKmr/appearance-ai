# Check that all pictures are present in the dataset
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


classToIndex = {
    "spring": 0,
    "summer": 1,
    "autumn": 2,
    "winter": 3
}

subClassToIndex = {
    "bright": 0,
    "cool": 1,
    "deep": 2,
    "light": 3,
    "soft": 4,
    "warm": 5
}

# A helper function to sort the columns in thw dataset
def parseBoolean(value):
    if isinstance(value, bool):
        return value

    normalizedValue = str(value).strip().lower()

    if normalizedValue in {"true", "1", "yes"}:
        return True

    if normalizedValue in {"false", "0", "no"}:
        return False

    raise ValueError(f"Unknown boolean value: {value}")

# A helper to work with strings
def normalizeString(value):
    return str(value).strip().lower()

class SeasonDataset(Dataset):
    def __init__(self, annotationsPath, releaseRoot, partition, transform=None, celebaOnly=None):
        self.annotationsPath = Path(annotationsPath)
        self.releaseRoot = Path(releaseRoot)
        self.partition = str(partition).strip().lower()
        self.transform = transform
        self.celebaOnly = celebaOnly

        if not self.annotationsPath.is_file():
            raise FileNotFoundError(
                "Annotations file was not found:\n"
                f"{self.annotationsPath}"
            )

        self.annotations = pd.read_csv(self.annotationsPath)

        # Get all the text in columns to the needed format
        self.annotations["partition"] = self.annotations["partition"].map(normalizeString)
        self.annotations["class"] = self.annotations["class"].map(normalizeString)
        self.annotations["sub_class"] = self.annotations["sub_class"].map(normalizeString)
        self.annotations["celeba"] = self.annotations["celeba"].map(parseBoolean)

        partitionMask = self.annotations["partition"] == self.partition
        self.annotations = self.annotations[partitionMask]

        if self.celebaOnly is not None:
            self.annotations = self.annotations[self.annotations["celeba"] == bool(self.celebaOnly)]

        self.annotations = self.annotations.reset_index(drop=True)

        if len(self.annotations) == 0:
            raise ValueError(
                "No annotations found for partition: "
                f"{self.partition}"
            )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]

        imagePath = self.releaseRoot / row["path_rgb_masked"]
        image = Image.open(imagePath).convert("RGB")

        className = row["class"]
        subClassName = row["sub_class"]

        classIndex = classToIndex[className]
        subClassIndex = subClassToIndex[subClassName]

        if self.transform is not None:
            image = self.transform(image)

        return image, classIndex, subClassIndex