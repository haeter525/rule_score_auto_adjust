from pathlib import Path
import os
import dotenv
dotenv.load_dotenv()

def get_folder() -> Path:
    return Path(os.getenv("GENERATED_RULE_FOLDER"))

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
