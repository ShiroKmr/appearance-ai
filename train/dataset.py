# Check that all pictures are present in the dataset
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


classToIndex = {
    "spring": 0,
    "summer": 1,
    "autumn": 2,
    "winter": 3,
}


class SeasonDataset(Dataset):
    def __init__(
        self,
        annotationsPath,
        releaseRoot,
        partition,
        transform=None,
    ):
        self.annotationsPath = Path(annotationsPath)
        self.releaseRoot = Path(releaseRoot)
        self.partition = str(partition).strip().lower()
        self.transform = transform

        if not self.annotationsPath.is_file():
            raise FileNotFoundError(
                "Annotations file was not found:\n"
                f"{self.annotationsPath}"
            )

        self.annotations = pd.read_csv(
            self.annotationsPath
        )

        requiredColumns = {
            "class",
            "partition",
            "path_rgb_masked",
        }

        missingColumns = (
            requiredColumns
            - set(self.annotations.columns)
        )

        if missingColumns:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missingColumns)}"
            )

        self.annotations["partition"] = (
            self.annotations["partition"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        self.annotations["class"] = (
            self.annotations["class"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        validPartitions = {
            "train",
            "validation",
            "test",
        }

        if self.partition not in validPartitions:
            raise ValueError(
                "Unknown partition: "
                f"{self.partition}. "
                "Expected train, validation or test."
            )

        self.annotations = self.annotations[
            self.annotations["partition"]
            == self.partition
        ].reset_index(drop=True)

        if len(self.annotations) == 0:
            raise ValueError(
                "No annotations found for partition: "
                f"{self.partition}"
            )

        unknownClasses = (
            set(self.annotations["class"].unique())
            - set(classToIndex.keys())
        )

        if unknownClasses:
            raise ValueError(
                "Unknown classes found: "
                f"{sorted(unknownClasses)}"
            )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        row = self.annotations.iloc[index]

        imagePath = self.resolveImagePath(row)

        if not imagePath.is_file():
            raise FileNotFoundError(
                "Image was not found:\n"
                f"{imagePath}"
            )

        image = Image.open(imagePath).convert("RGB")

        className = row["class"]
        classIndex = classToIndex[className]

        if self.transform is not None:
            image = self.transform(image)

        return image, classIndex

    def resolveImagePath(self, row):
        csvPath = str(
            row["path_rgb_masked"]
        ).strip().replace("\\", "/")

        csvPath = Path(csvPath)
        relativeParts = csvPath.parts[1:]

        if (
            relativeParts
            and relativeParts[0] in {"train", "test"}
        ):
            physicalPartition = relativeParts[0]
            relativeParts = relativeParts[1:]

        elif self.partition == "validation":
            physicalPartition = "train"

        else:
            physicalPartition = self.partition

        return (
            self.releaseRoot
            / "RGB-M"
            / physicalPartition
            / Path(*relativeParts)
        )