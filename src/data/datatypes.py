import os
from dataclasses import dataclass
from functools import reduce
from typing import Any, Callable, Optional

import numpy as np
import torch
import polars as pl
from rich.progress import track

from src.settings import ID_COLUMN, TARGET_COLUMN, TEXT_COLUMN

@dataclass
class Lookups:
    data_info: dict[str, Any]
    code2description: Optional[dict[str, str]] = None
    code_system2code_indices: Optional[dict[str, torch.Tensor]] = None
    split2code_indices: Optional[dict[str, torch.Tensor]] = None

@dataclass
class Data:
    """Dataclass containing the dataset and the code occurrences of each code system."""

    df: pl.DataFrame
    code_system2code_counts: dict[str, dict[str, int]]

    @property
    def train(self) -> list[tuple]:
        """Get the training data."""
        return self._split_data("train")

    @property
    def val(self) -> list[tuple]:
        """Get the validation data."""
        return self._split_data("val")

    @property
    def test(self) -> list[tuple]:
        """Get the test data."""
        return self._split_data("test")

    def _split_data(self, split: str) -> list[tuple]:
        """Helper to split and process data by split type."""
        filtered_df = self.df.filter(self.df["split"] == split).sort("num_words")
        examples = []

        for row in track(filtered_df.iter_rows(named=True), description=f"Processing {split} data"):
            token_ids = torch.tensor(row["token_ids"])
            targets = np.array(row[TARGET_COLUMN], dtype=object)
            id = row[ID_COLUMN]
            num_tokens = len(token_ids)
            attention_mask = torch.ones_like(token_ids)
            examples.append((token_ids, targets, id, num_tokens, attention_mask))
        return examples

    @property
    def get_documents(self) -> list[str]:
        """Get all the documents in the dataset."""
        return self.df[TEXT_COLUMN].to_list()

    @property
    def all_target_counts(self) -> dict[str, int]:
        """Get the number of occurrences of each code in the dataset."""
        return reduce(lambda x, y: {**x, **y}, self.code_system2code_counts.values())

    @property
    def get_train_documents(self) -> list[str]:
        """Get the training documents."""
        return self.df.filter(self.df["split"] == "train")[TEXT_COLUMN].to_list()

    def split_targets(self, name: str) -> set[str]:
        """Get the targets of a split."""
        filtered = self.df.filter(self.df["split"] == name)
        targets = filtered[TARGET_COLUMN].explode()
        return set(targets.unique())

    def split_size(self, name: str) -> int:
        """Get the size of a split."""
        return self.df.filter(self.df["split"] == name).height

    def num_split_targets(self, name: str) -> int:
        """Get the number of targets of a split."""
        return len(self.split_targets(name))

    @property
    def all_targets(self) -> set[str]:
        """Get all the targets in the dataset."""
        all_codes = set()
        for codesystem in self.code_system2code_counts.values():
            all_codes |= set(codesystem.keys())
        return all_codes

    @property
    def info(self) -> dict[str, int]:
        """Get information about the dataset."""
        return {
            "num_classes": len(self.all_targets),
            "num_examples": len(self.df),
            "num_train_tokens": self.df.filter(self.df["split"] == "train")["num_words"].sum(),
            "average_tokens_per_example": self.df["num_words"].mean(),
            "num_train_examples": self.split_size("train"),
            "num_val_examples": self.split_size("val"),
            "num_test_examples": self.split_size("test"),
            "num_train_classes": self.num_split_targets("train"),
            "num_val_classes": self.num_split_targets("val"),
            "num_test_classes": self.num_split_targets("test"),
            "average_classes_per_example": sum(
                sum(codesystem.values()) for codesystem in self.code_system2code_counts.values()
            )
            / len(self.df),
        }

    def truncate_text(self, max_length: int) -> None:
        """Truncate text to a maximum length."""
        if max_length is None:
            return
        self.df = self.df.with_columns(
            pl.col(TEXT_COLUMN)
            .str.split(" ")
            .list.slice(0, max_length)
            .list.join(" ")
            .alias(TEXT_COLUMN)
        )

    def transform_text(self, batch_transform: Callable[[list[str]], str]) -> None:
        """Transform the text using a batch transform function."""
        token_ids_list = []
        for batch in self.df.iter_slices(n_rows=10000):
            texts = batch[TEXT_COLUMN].to_list()
            transformed_texts = batch_transform(texts)
            token_ids_list += transformed_texts

        self.df = self.df.with_columns(
            pl.Series("token_ids", token_ids_list)
        )
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

@dataclass
class Batch:
    """Batch class. Used to store a batch of data."""

    data: torch.Tensor
    targets: torch.Tensor
    ids: torch.Tensor
    code_descriptions: Optional[torch.Tensor] = None
    num_tokens: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None

    def to(self, device: Any) -> "Batch":
        """Move the batch to a device.

        Args:
            device (Any): Device to move the batch to.

        Returns:
            self: Moved batch.
        """
        self.data = self.data.to(device, non_blocking=True)
        self.targets = self.targets.to(device, non_blocking=True)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.to(device, non_blocking=True)
        return self

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.data = self.data.pin_memory()
        self.targets = self.targets.pin_memory()
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        return self
