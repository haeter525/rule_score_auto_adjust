from pathlib import Path
import polars as pl
import os
import dotenv

dotenv.load_dotenv()

RULE_LIST_SCHEMA = {"rule": pl.String()}


def get_folder() -> Path:
    return Path(os.getenv("GENERATED_RULE_FOLDER"))


def load_list(rule_list: str) -> pl.DataFrame:
    return pl.read_csv(
        rule_list,
        schema_overrides=RULE_LIST_SCHEMA,
        has_header=True,
        columns=list(RULE_LIST_SCHEMA.keys()),
    )


def get(rule_name: str) -> Path:
    rule_path = get_folder() / f"{rule_name}"
    return rule_path.resolve()


def build_rule_folder(rule_names: list[str], folder: Path) -> Path:
    for rule in rule_names:
        source_rule_path = get(rule)
        target_rule_path = folder / (source_rule_path.name)

        if target_rule_path.exists():
            target_rule_path.unlink()

        target_rule_path.symlink_to(source_rule_path)
