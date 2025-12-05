import tomllib

import click
from loguru import logger

from setfit_hf.oneshot import DuckDBSetFitClassifier


@click.command()
@click.option(
    "--config", "-c", default="conf/config.toml", help="Path to config.toml file."
)
@click.option(
    "--few-shot", default=None, type=int, help="Override few-shot examples per label."
)
@click.option("--epochs", default=None, type=int, help="Override number of epochs.")
def main(config: str, few_shot: int, epochs: int):
    """CLI to run few-shot classification with SetFit and DuckDB-backed datasets."""
    with open(config, "rb") as f:
        params = tomllib.load(f)

    # Read parameters from config, allowing CLI overrides
    data_source = params["dataset"]["data_source"]
    table_or_query = params["dataset"].get("table_or_query", None)
    desc_col = params["dataset"]["description_column"]
    label_col = params["dataset"]["label_column"]

    model_name = params["model"]["model_name"]
    batch_size = params["model"]["batch_size"]
    num_epochs = params["model"]["num_epochs"]
    model_cache_path = params["model"]["model_cache_path"]

    few_shot_per_label = (
        few_shot if few_shot is not None else params["train"]["few_shot_per_label"]
    )
    random_state = params["train"]["random_state"]
    print_report = params["evaluation"]["print_report"]

    logger.info(f"Launching pipeline with: {data_source}, {model_name}.")

    # Initialize and run
    clf = DuckDBSetFitClassifier(
        data_source=data_source,
        table_or_query=table_or_query,
        desc_col=desc_col,
        label_col=label_col,
        model_name=model_name,
        model_cache_path=model_cache_path,
    )
    clf.load_and_split(few_shot_per_label=few_shot_per_label, random_state=random_state)
    clf.train(num_epochs=epochs or num_epochs)
    clf.evaluate(print_report=print_report)


if __name__ == "__main__":
    main()
