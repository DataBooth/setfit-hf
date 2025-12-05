import pathlib
import pickle
from typing import Any, List, Optional, Tuple

import duckdb
import pandas as pd
from loguru import logger
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import classification_report


class DuckDBSetFitClassifier:
    """
    Class for few-shot text classification with SetFit using DuckDB as the data backend.
    Supports stratified sampling for training and test split, logging, and result caching.
    """

    def __init__(
        self,
        data_source: str,
        table_or_query: str,
        desc_col: str = "description",
        label_col: str = "label",
        model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
        model_cache_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the classifier with DuckDB source, column names and model.
        """
        self.data_source = data_source
        self.table_or_query = table_or_query
        self.desc_col = desc_col
        self.label_col = label_col
        self.model_name = model_name
        # For model and prediction caching
        self.model_cache_path = model_cache_path or "setfit_model.pkl"
        self.model: Optional[SetFitModel] = None
        self.labels: Optional[List[str]] = None
        self.train_samples: Optional[List[Tuple[str, str]]] = None
        self.test_samples: Optional[List[str]] = None
        self.test_labels: Optional[List[str]] = None

    def load_and_split(
        self, few_shot_per_label: int = 8, random_state: int = 42
    ) -> None:
        """
        Loads labeled data from DuckDB DB or file and splits into few-shot stratified train/test.
        """
        logger.info("Loading data: {}", self.data_source)
        if self.table_or_query:  # Will be None for file-based datasets
            # This is for DuckDB DB files
            con = duckdb.connect(self.data_source)
            query = (
                f"SELECT {self.desc_col}, {self.label_col} FROM {self.table_or_query} "
                f"WHERE {self.desc_col} IS NOT NULL AND {self.label_col} IS NOT NULL"
            )
            df: pd.DataFrame = con.execute(query).fetchdf()
            con.close()
        else:
            # This is for CSV/Parquet/URL/file-like sources
            con = duckdb.connect()  # NEW temporary connection
            query = (
                f"SELECT {self.desc_col}, {self.label_col} "
                f"FROM '{self.data_source}' "
                f"WHERE {self.desc_col} IS NOT NULL AND {self.label_col} IS NOT NULL"
            )
            df: pd.DataFrame = con.execute(query).fetchdf()
            logger.info(f"Columns in loaded dataframe: {df.columns.tolist()}")
            logger.info(f"First few rows: {df.head()}")
            logger.info(f"Number of rows in df: {len(df)}")

            con.close()

    def train(
        self, batch_size: int = 8, num_epochs: int = 10, use_cache: bool = True
    ) -> None:
        """
        Trains SetFit model using the few-shot training data. Supports optional disk caching.
        """
        if use_cache and pathlib.Path(self.model_cache_path).exists():
            logger.info("Loading SetFit model from cache: {}", self.model_cache_path)
            with open(self.model_cache_path, "rb") as f:
                self.model = pickle.load(f)
            return

        if not self.train_samples or not self.labels:
            raise RuntimeError(
                "No training samples or label list. Call load_and_split first."
            )

        logger.info("Training SetFit model...")
        self.model = SetFitModel.from_pretrained(self.model_name, labels=self.labels)
        trainer = Trainer(
            model=self.model,
            train_dataset=self.train_samples,
            args=TrainingArguments(batch_size=batch_size, num_epochs=num_epochs),
        )
        trainer.train()
        logger.success("SetFit model training complete")

        # Cache to disk
        if use_cache:
            with open(self.model_cache_path, "wb") as f:
                pickle.dump(self.model, f)
            logger.info("Model cached to: {}", self.model_cache_path)

    def predict(self, samples: Optional[List[str]] = None) -> List[Any]:
        """
        Predicts labels for the supplied samples (or test set if not provided).
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Run train() first.")
        batch = samples or self.test_samples
        logger.info("Predicting labels for {} samples...", len(batch))
        return self.model.predict(batch)

    def evaluate(self, print_report: bool = True) -> Any:
        """
        Predicts labels on test set and prints scikit-learn classification report.
        """
        if not self.test_labels or not self.test_samples:
            raise RuntimeError("No test data found. Call load_and_split first.")
        preds = self.predict(self.test_samples)
        report = classification_report(
            self.test_labels, preds, digits=3, output_dict=not print_report
        )
        if print_report:
            logger.success(
                "Evaluation report:\n{}",
                classification_report(self.test_labels, preds, digits=3),
            )
        return report
