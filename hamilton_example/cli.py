"""Console script for hamilton_example."""

import sys
import click
import hamilton_example.data as data
import hamilton_example.functions as functions

from hamilton import base, driver


def get_model_config(model_type: str) -> dict:
    if model_type == "svm":
        return {"classifier": "svm", "gamma": 0.001}
    elif model_type == "logistic":
        return {"classifier": "logistic", "penalty": "l2"}
    else:
        raise ValueError(f"Unsupported model {model_type}.")


@click.command()
@click.option("--model_type")
def main(model_type):

    dag_config = {
        "test_size_fraction": 0.5,
        "shuffle_train_test_split": True,
    }
    dag_config.update(get_model_config(model_type))

    adapter = base.DefaultAdapter()
    dr = driver.Driver(dag_config, data, functions, adapter=adapter)

    results = dr.execute(["classification_report", "confusion_matrix", "fit_model"])

    for k, v in results.items():
        print(f"\n {k}: \n {v}")

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
