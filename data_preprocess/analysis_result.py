import os
from pathlib import Path
import json
import enum
from warnings import deprecated
from diskcache import FanoutCache
from quark.core.quark import Quark
from quark.core.struct.ruleobject import RuleObject as Rule
import polars as pl
import functools

SCHEMA = {
    "rule_name": pl.String(),
    "passing_stage": pl.Int8(),
}

cache = FanoutCache(f"{os.getenv("CACHE_FOLDER")}/analysis_result_status")


class ANALYSIS_STATUS(enum.Enum):
    SUCCESS: int = 0
    FAILED: int = 1


def get_folder() -> Path:
    return Path(os.getenv("ANALYSIS_RESULT_FOLDER", "NOT_DEFINED"))


@deprecated("Use get_file instead")
def get_file_old(sha256: str) -> Path:
    return get_folder() / f"{sha256}.apk_progress.json"


def get_file_new(sha256: str) -> Path:
    return get_folder() / f"{sha256}.csv"


get_file = get_file_new


@deprecated("Use load instead")
def load_old(sha256: str) -> tuple[str, int]:
    def parse_item(item: tuple[str, int] | str) -> tuple[str, int]:
        return (item, -1) if isinstance(item, str) else item

    file = get_file_old(sha256=sha256)
    if not file.exists():
        return tuple()

    try:
        with get_file_old(sha256=sha256).open("r") as file:
            content = json.load(file)
            return [parse_item(item) for item in content]
    except json.JSONDecodeError as e:
        print(f"Failed to parser JSON file: {file}")
        return []


def load_as_dataframe(sha256: str) -> pl.DataFrame:
    file = get_file_new(sha256=sha256)
    if not file.exists():
        return pl.DataFrame()

    table = pl.read_csv(file, schema=SCHEMA, has_header=True)
    return table


@functools.lru_cache(maxsize=512)
def load_as_dict(sha256: str) -> dict[str, int]:
    return {rule: stage for rule, stage in load_as_dataframe(sha256=sha256).rows()}


def save_as_dict(sha256: str, analysis_result: dict[str, int]):
    return save(sha256, list(analysis_result.items()))


def load_new(sha256: str) -> list[str, int]:
    table = load_as_dataframe(sha256=sha256)
    return [r for r in table.rows()]


def load(sha256: str) -> list[str, int]:
    combined = [tuple(item) for item in load_new(sha256=sha256)]
    return list(set(combined))


def save(sha256: str, analysis_result: list[str, int]) -> Path:
    file = get_file_new(sha256=sha256)
    pl.DataFrame(analysis_result, schema=SCHEMA, orient="row").write_csv(
        file, include_header=True
    )
    return file


@functools.lru_cache(maxsize=6)
def _get_quark(apk_path: Path) -> Quark:
    return Quark(str(apk_path))


def analyze_rules(
    sha256: str,
    apk_path: Path,
    rule_paths: list[Path],
    use_cache: bool = True,
) -> dict[str, int]:
    results = {
        rule.name: analyze(sha256, rule, apk_path, use_cache) for rule in rule_paths
    }
    return results


def analyze(
    sha256: str,
    rule_path: Path,
    apk_path: Path,
    use_cache: bool = True,
) -> int:
    assert apk_path.exists(), f"apk_path {apk_path} does not exist"
    assert rule_path.exists(), f"rule_path {rule_path} does not exist"

    subcache = cache.get(sha256, cache.cache(sha256, disk=sha256))

    rule_name = rule_path.name
    if (not use_cache) or rule_name not in subcache:
        existing_result = load_as_dict(sha256)
        if rule_name in existing_result:
            # Migrating: Check if result exists in the analysis_result file
            # print(f"Find analysis result for {sha256} and {rule_name} in analysis_result file")
            stage = existing_result[rule_name]
            subcache.set(
                rule_name,
                ANALYSIS_STATUS.SUCCESS if stage >= 0 else ANALYSIS_STATUS.FAILED,
            )
        else:
            # Run Quark Analysis
            # print(f"Run Quark Analysis for {sha256} and {rule_name}")
            try:
                quark = _get_quark(apk_path)
                stage = quark.run(Rule(str(rule_path)))

                _append_result(sha256, {rule_name: stage})
                subcache.set(rule_name, ANALYSIS_STATUS.SUCCESS)

            except Exception as e:
                print(f"Error analyzing {sha256} {rule_name}: {e}")
                subcache.set(rule_name, ANALYSIS_STATUS.FAILED)
    else:
        # print(f"Analysis result for {sha256} and {rule_name} is in cache")
        pass

    match subcache[rule_name]:
        case ANALYSIS_STATUS.SUCCESS:
            return load_as_dict(sha256)[rule_name]
        case ANALYSIS_STATUS.FAILED:
            return -1


def _append_result(sha256: str, results: dict[str, int]) -> Path:
    existing_results = load_as_dict(sha256)

    for rule, stage in results.items():
        if stage > existing_results.get(rule, -1):
            existing_results[rule] = stage

    return save_as_dict(sha256, existing_results)
