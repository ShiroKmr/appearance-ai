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

subClassToIndex = {
    "bright": 0,
    "cool": 1,
    "deep": 2,
    "light": 3,
    "soft": 4,
    "warm": 5,
}


class SeasonDataset(Dataset):
    def __init__(
        self,
        annotationsPath,
        releaseRoot,
        partition,
        transform=None,
        includeSubClass=False,
        celebaOnly=None,
        includePivotalValidation=False,
    ):
        self.annotationsPath = Path(annotationsPath)
        self.releaseRoot = Path(releaseRoot)
        self.partition = str(partition).strip().lower()
        self.transform = transform
        self.includeSubClass = includeSubClass
        self.celebaOnly = celebaOnly
        self.includePivotalValidation = includePivotalValidation

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
            "sub_class",
            "partition",
            "celeba",
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

        self.annotations["sub_class"] = (
            self.annotations["sub_class"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        self.annotations["celeba"] = (
            self.annotations["celeba"]
            .map(self.parseBoolean)
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

        partitionMask = self.annotations["partition"] == self.partition

        if (
            self.partition == "train"
            and self.includePivotalValidation
        ):
            pivotalValidationMask = (
                (self.annotations["partition"] == "validation")
                & ~self.annotations["celeba"]
            )
            partitionMask = partitionMask | pivotalValidationMask

        self.annotations = self.annotations[partitionMask]

        if self.celebaOnly is not None:
            self.annotations = self.annotations[
                self.annotations["celeba"]
                == bool(self.celebaOnly)
            ]

        self.annotations = self.annotations.reset_index(drop=True)

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

        unknownSubClasses = (
            set(self.annotations["sub_class"].unique())
            - set(subClassToIndex.keys())
        )

        if unknownSubClasses:
            raise ValueError(
                "Unknown sub-classes found: "
                f"{sorted(unknownSubClasses)}"
            )

    @staticmethod
    def parseBoolean(value):
        if isinstance(value, bool):
            return value

        normalizedValue = str(value).strip().lower()

        if normalizedValue in {"true", "1", "yes"}:
            return True

        if normalizedValue in {"false", "0", "no"}:
            return False

        raise ValueError(f"Unknown boolean value: {value}")

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

        if self.includeSubClass:
            subClassName = row["sub_class"]
            subClassIndex = subClassToIndex[subClassName]

            return image, classIndex, subClassIndex

        return image, classIndex

    def getSourceSamplingWeights(self, celebaWeight=2.0):
        return [
            celebaWeight if isCeleba else 1.0
            for isCeleba in self.annotations["celeba"].tolist()
        ]

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
