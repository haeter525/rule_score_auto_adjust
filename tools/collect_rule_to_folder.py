from collections import Counter
import click
import polars as pl
from pathlib import Path
from tools.backup.collect_rules_to_folder import create_symbolic_links_to_rules


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
    "--output_folder",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Output folder to collect rules into.",
)
def entry_point(rule_list: list[Path], rule_base_folder: Path, output_folder: Path):
    """Collect rules from multiple lists to a single folder using symbolic links.
    
    Example usage:
    uv run tools/collect_rule_to_folder.py \
        -r /mnt/storage/data/rule_to_release/golddream/rule_added.csv \
        -b /mnt/storage/data/generated_rules \
        -o /tmp/rule_folder
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Flatten the rule list
    rules = (
        pl.concat([pl.read_csv(str(rule_list_path), columns=["rule"]) for rule_list_path in rule_list], how="vertical")
        .to_series()
        .to_list()
    )

    # Show duplicate rules
    counts = Counter(rules)
    duplicates = [rule for rule, count in counts.items() if count > 1]
    if duplicates:
        print(f"Duplicate rules found: {duplicates}")

    rule_paths = {rule_base_folder / rule_name for rule_name in rules}

    # Create symbolic links to rules
    create_symbolic_links_to_rules(rule_paths, output_folder)

    print(f"Collected {len(rule_paths)} rules to {output_folder}")


if __name__ == "__main__":
    entry_point()
