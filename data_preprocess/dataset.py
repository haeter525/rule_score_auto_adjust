from pathlib import Path
import torch
import polars as pl
from data_preprocess import rule

class RuleDataset(torch.utils.data.Dataset):

    SCHEMA = {
        "index": pl.Int8(),
        "rule_name": pl.String()
    }

    def __init__(self, rule_names: list[str], dataset_folder: Path) -> None:
        super().__init__()

        real_rule_paths = [rule.get(rule_name) for rule_name in rule_names]
        assert all(p.exists() for p in real_rule_paths)

        self.folder = dataset_folder / "rules"
        self.folder.mkdir()
        rule.build_rule_folder(rule_names, self.folder)
        
        pl.DataFrame()

        self.rule_paths = [rule.get(rule_name) for rule_name in rule_names]
        # TODO    


class ApkDataset():
    # TODO
    pass