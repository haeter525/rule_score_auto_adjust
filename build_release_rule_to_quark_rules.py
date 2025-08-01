import functools
from pathlib import Path
from typing import Generator, Iterator
import polars as pl
from prefect import flow, task
from tqdm import tqdm
import data_preprocess.rule as rule_lib
from create_rule_folder import update_rule_content
import shutil


@task
def get_rule_scores_from(apk_prediction: Path, revert_score: bool) -> list[tuple[str, float]]:
    rule_score_df = (
        pl.read_csv(apk_prediction, n_rows=1, skip_rows_after_header=1)
        .drop(["y_truth", "y_score", "y_pred"], strict=False)
        .transpose(include_header=True, column_names="sha256", header_name="rule")
    )

    if revert_score:
        rule_score_df = rule_score_df.select(pl.col("rule"), (pl.col("rule_score") * -1).alias("rule_score"))

    return list(rule_score_df.iter_rows(named=False))


@task
def apply_rule_scores(rule_scores: list[tuple[str, float]]) -> None:
    for rule_name, score in tqdm(rule_scores):
        rule_path = rule_lib.get(rule_name)

        update_rule_content(
            rule_path,
            lambda _, content: content | {"score": round(score, 3)},
        )


@task
def get_rule_descriptions_and_labels_from(rule_review: Path) -> pl.DataFrame:
    return pl.read_csv(rule_review, columns=["rule", "description", "label"])


@task
def apply_rule_description_and_labels(rule_description_and_labels: pl.DataFrame) -> None:
    def update_crime(name: str, content: dict, description: str, labels: list[str]) -> dict:
        print(f"Update rule {name} with description: {description}")
        return content | {"crime": f"{description}", "label": labels}

    for rule_name, description, label_str in tqdm(
        rule_description_and_labels.iter_rows(named=False)
    ):
        rule_path = rule_lib.get(rule_name)
        labels = label_str.split("|")

        update_rule_content(
            rule_path, functools.partial(update_crime, description=description, labels=labels)
        )


@task(log_prints=True)
def index_rule(rule_names: list[str], start_index: int) -> dict[str, str]:
    print(f"Indexing {len(rule_names)} rules starting from index {start_index}")
    return {name: f"{idx:05d}.json" for idx, name in enumerate(rule_names, start_index)}


@task(log_prints=True)
def copy_rule_to_quark_rules(rule_name_mappings: dict[str, str], rule_set_path: Path) -> None:
    for original_name, new_name in rule_name_mappings.items():
        original_path = rule_lib.get(original_name)
        new_path = rule_set_path / "rules" / new_name
        new_path.unlink(missing_ok=True)
        shutil.copyfile(original_path, new_path)
        print(f"Copy rule {original_path} to {new_path}")


@flow
def update_rule_set(
    apk_prediction: Path,
    rule_review: Path,
    builtin_rule_set: Path,
    start_index: int,
    revert_score: bool,
) -> None:
    # TODO - Add a check to auto determine if revert_score is needed
    # revert_score = revert_score or check_rule_score(apk_prediction)
    
    rule_scores = get_rule_scores_from(apk_prediction, revert_score)
    apply_rule_scores(rule_scores)

    rule_description_and_labels = get_rule_descriptions_and_labels_from(rule_review)
    apply_rule_description_and_labels(rule_description_and_labels)
    # TODO - Combine apply_rule_scores and apply_rule_description_and_labels to a unified task called update_rules(new_contents: dict[str, dict[str, Any]]) -> None

    builtin_rule_mappings = {
        rule_path.name: rule_path.name for rule_path in (builtin_rule_set / "rules").glob("*.json")
    }
    indexed_rule_mappings = index_rule(rule_description_and_labels["rule"].to_list(), start_index)
    copy_rule_to_quark_rules(builtin_rule_mappings | indexed_rule_mappings, builtin_rule_set)


if __name__ == "__main__":
    update_rule_set(
        apk_prediction=Path("/mnt/storage/haeter/rule_score_auto_adjust_haeter/apk_prediction.csv"),
        rule_review=Path("/mnt/storage/haeter/rule_score_auto_adjust_haeter/rule_reviews.csv"),
        builtin_rule_set=Path("/mnt/storage/quark-rules"),
        start_index=234,
        revert_score=False,
    )
