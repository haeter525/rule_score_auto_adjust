import shutil
import click
import polars as pl
from pathlib import Path
from prefect import flow, task


@task(log_prints=True)
def index_rule(rule_names: list[str], start_index: int) -> dict[str, str]:
    print(f"Indexing {len(rule_names)} rules starting from index {start_index}")
    return {name: f"{idx:05d}.json" for idx, name in enumerate(rule_names, start_index)}


@flow(name="copy_rule_to_quark_rules")
def copy_rule_to_quark_rules(
    quark_rule_folder: Path,
    rule_list: list[Path],
    rule_base_folder: Path,
    start_index: int,
) -> None:
    quark_rule_mappings = {
        rule_path.name: rule_path.name for rule_path in (quark_rule_folder / "rules").glob("*.json")
    }

    rule_names = (
        pl.concat([pl.read_csv(list_path, columns=["rule"]) for list_path in rule_list])
        .to_series()
        .to_list()
    )
    indexed_rule_mappings = index_rule(rule_names, start_index)

    # Also update the Quark rules with new weights to the Quark rule folder
    rule_name_mappings = quark_rule_mappings | indexed_rule_mappings

    for original_name, new_name in rule_name_mappings.items():
        original_path = rule_base_folder / original_name
        new_path = quark_rule_folder / "rules" / new_name
        new_path.unlink(missing_ok=True)
        shutil.copyfile(original_path, new_path)
        print(f"Copy rule {original_path} to {new_path}")

    print(f"Copied {len(rule_name_mappings)} rules to {quark_rule_folder / 'rules'}")


@click.command()
@click.option(
    "--rule_list",
    "-r",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    multiple=True,
    help="List of rules to collect.",
)
@click.option(
    "--rule_base_folder",
    "-b",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Base folder where rules are stored.",
)
@click.option(
    "--quark_rule_folder",
    "-q",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Folder where Quark rules are stored.",
)
@click.option(
    "--start_index",
    "-i",
    type=int,
    required=True,
    help="Starting index for rule names.",
)
def entry_point(
    quark_rule_folder: Path,
    rule_list: list[Path],
    rule_base_folder: Path,
    start_index: int,
):
    """Copy rules from a list to the Quark rules folder with indexed names.
    
    Example usage:
    uv run tools/copy_rule_to_quark_rules.py \
        -r /mnt/storage/data/rule_to_release/golddream/rule_added.csv \
        -b /mnt/storage/data/generated_rules \
        -q /mnt/storage/data/quark_rules \
        -i 1000
    """
    copy_rule_to_quark_rules(
        quark_rule_folder=quark_rule_folder,
        rule_list=rule_list,
        rule_base_folder=rule_base_folder,
        start_index=start_index,
    )


if __name__ == "__main__":
    entry_point()
