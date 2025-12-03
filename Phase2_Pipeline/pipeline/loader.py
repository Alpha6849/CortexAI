
"""
loader.py

Responsible for loading CSV files safely, validating inputs,
handling encoding issues, and returning a clean pandas DataFrame.

Part of the CortexAI Phase 2 production pipeline.
"""

import os
import logging
import pandas as pd
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [DataLoader] - %(levelname)s - %(message)s"
)


class DataLoader:
    """
    Handles CSV loading, validation, and safe DataFrame creation.

    Responsibilities:
    - Check file path validity
    - Validate CSV extension
    - Handle encoding errors gracefully
    - Detect auto separator
    - Return a pandas DataFrame
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def _check_exists(self) -> None:
        logging.info(f"Checking if file exists: {self.file_path}")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

    def _check_extension(self) -> None:
        logging.info("Checking file extension...")
        if not self.file_path.lower().endswith(".csv"):
            raise ValueError("Input file must be a .csv file.")

    def _check_file_size(self, max_mb: int = 200) -> None:
        logging.info("Checking file size...")
        file_size = os.path.getsize(self.file_path) / (1024 * 1024)  # convert bytes to MB

        if file_size > max_mb:
            raise ValueError(
                f"File too large ({file_size:.2f} MB). Maximum allowed size is {max_mb} MB."
            )

    def _detect_separator(self) -> str:
        """
        Detects the most likely separator by reading a small sample.
        """
        logging.info("Detecting separator...")

        # latin1 is used here because it never throws decoding errors
        with open(self.file_path, "r", encoding="latin1") as f:
            sample = f.read(2048)  # read first 2KB

        possible_separators = [",", ";", "\t", "|"]
        counts = {sep: sample.count(sep) for sep in possible_separators}

        best_sep = max(counts, key=counts.get)
        logging.info(f"Detected separator: '{best_sep}'")

        return best_sep

    def _safe_read_csv(self) -> Tuple[pd.DataFrame, str, str]:
        """
        Attempts to read a CSV using multiple encodings to avoid crashes.
        Returns: (df, encoding_used, separator_used)
        """
        logging.info("Attempting safe CSV read with fallback encodings...")

        sep = self._detect_separator()
        encodings_to_try = ["utf-8", "iso-8859-1", "latin1"]

        for enc in encodings_to_try:
            try:
                logging.info(f"Trying encoding: {enc}")
                df = pd.read_csv(self.file_path, encoding=enc, sep=sep)
                return df, enc, sep
            except UnicodeDecodeError:
                logging.warning(f"Encoding failed: {enc}")

        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1,
            "Unable to read CSV with common encodings."
        )


    def load(self) -> Tuple[pd.DataFrame, dict]:
        """
        Public method to load CSV safely.

        Returns:
            (DataFrame, metadata_dict)
        """
        logging.info("Starting CSV load pipeline...")

        try:
            self._check_exists()
            self._check_extension()
            self._check_file_size()

            df, encoding_used, sep_used = self._safe_read_csv()
            metadata = self._build_metadata(df, encoding_used, sep_used)

            logging.info("CSV loaded successfully with metadata.")
            return df, metadata

        except Exception as e:
            self._raise_error(f"Failed to load CSV: {e}")


    
    def _build_metadata(self, df: pd.DataFrame, encoding: str, sep: str) -> dict:
        """
        Creates metadata dictionary about the loaded CSV.
        """
        file_size_mb = os.path.getsize(self.file_path) / (1024 * 1024)

        return {
            "file_path": self.file_path,
            "file_size_mb": round(file_size_mb, 2),
            "rows": df.shape[0],
            "columns": df.shape[1],
            "encoding_used": encoding,
            "separator_used": sep,
        }
        
    def _raise_error(self, message: str):
        """Raise a clean, user-friendly ValueError."""
        logging.error(message)
        raise ValueError(message)
    
    def load_df(self) -> pd.DataFrame:
        """
        Shortcut method to return only the DataFrame.
        Useful for simple UI flows.
        """
        df, _ = self.load()
        return df


    




