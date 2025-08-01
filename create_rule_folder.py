import json
from pathlib import Path
import tempfile
from typing import Any, Callable
import polars as pl
import data_preprocess.rule as rule_lib
from prefect import task
import prefect.cache_policies as cache_policies


def update_rule_content(
    rule_path: Path,
    update_func: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> None:
    with rule_path.open("r") as in_stream:
        rule = json.load(in_stream)

    rule = update_func(rule_path.name, rule)

    with rule_path.open("w") as out_stream:
        json.dump(rule, out_stream, indent=4)


@task(cache_policy=cache_policies.INPUTS, log_prints=True)
def collect_rules_to_folder_from_apk_prediction(
    apk_prediction: Path,
) -> Path:

    apk_prediction_df = pl.read_csv(apk_prediction, n_rows=3)
    all_rule_names = apk_prediction_df.columns[4:]

    # Create a rule folder and link all rules to it
    rule_folder = collect_rules_to_folder(
        rule_names=all_rule_names,
    )

    # Get rule score from apk_prediction
    rule_scores: dict[str, float] = (
        apk_prediction_df.filter(pl.col("sha256").str.ends_with("rule_score"))
        .select(
            [col for col in apk_prediction_df.columns if col.endswith(".json")]
        )
        .to_dicts()[0]
    )

    # Apply rule score
    def update_func(rule_name: str, content: dict[str, Any]) -> dict[str, Any]:
        content["score"] = rule_scores[rule_name]
        return content

    for rule_name in all_rule_names:
        rule_path = rule_folder / rule_name
        update_rule_content(rule_path, update_func)

    return rule_folder


@task
def collect_rules_to_folder(
    rule_names: list[str],
) -> Path:
    # create rule folder
    rule_folder = create_rule_folder()

    # Get rule paths
    rules = [rule_lib.get(rule_name) for rule_name in rule_names]
    # Create symbolic link to candidate rules
    create_symbolic_links_to_rules(
        rules=rules,
        rule_folder=rule_folder,
    )

    # Check if missing any rules
    num_of_rules_in_folder = sum(1 for _ in rule_folder.glob("*.json"))
    assert num_of_rules_in_folder == len(
        rule_names
    ), f"Expected rule num: {len(rule_names)}, got {num_of_rules_in_folder}"

    return rule_folder

@task(log_prints=True)
def create_symbolic_links_to_rules(rules: list[Path], rule_folder: Path):
    for rule in rules:
        target_path = rule_folder / rule.name
        if target_path.exists():
            print(f"Warning: {target_path} already exists, removing it.")
            target_path.unlink()

        target_path.symlink_to(rule)


@task
def create_rule_folder() -> Path:
    rule_folder = Path(
        tempfile.TemporaryDirectory(prefix="rule_folder_", delete=False).name
    )
    print(f"Create rule folder: {str(rule_folder)}")
    return rule_folder


if __name__ == "__main__":
    output_path = collect_rules_to_folder_from_apk_prediction(
        apk_prediction=Path(
            "/mnt/storage/rule_score_auto_adjust_haeter/2025-06-28T18:38:53.csv"
        )
    )
    print(output_path)
