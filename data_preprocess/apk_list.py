from pathlib import Path
import itertools
import polars as pl
import dotenv
import os

dotenv.load_dotenv()


def get_androzoo_list() -> Path:
    return Path(os.getenv("/mnt/storage/data/latest.csv"))


def load_androzoo_list(androzoo_list: Path = get_androzoo_list()) -> pl.LazyFrame:
    return pl.scan_csv(androzoo_list).with_columns(
        pl.col("vt_scan_date").str.strptime(pl.Datetime(), "%Y-%m-%d %H:%M:%S")
    )


def get_market_from_androzoo(market: str) -> pl.LazyFrame:
    return (
        load_androzoo_list()
        .filter(pl.col("markets").str.contains(market))
        .with_columns(
            pl.col("vt_scan_date").str.strptime(pl.Datetime(), "%Y-%m-%d %H:%M:%S")
        )
    )


def get_apk_list_by_size_range(size_start: int, size_end: int) -> pl.LazyFrame:
    return (
        load_androzoo_list()
        .filter(
            (pl.col("apk_size").ge(size_start)).and_(pl.col("apk_size").lt(size_end))
        )
        .limit(100)
        .sort("apk_size", "vt_scan_date")
    )


size_stage = [
    1_000_000,
    2_000_000,
    4_000_000,
    8_000_000,
    16_000_000,
    32_000_000,
    64_000_000,
    128_000_000,
]

apk_lists = [
    apk_list.filter(filter_size_range(size_start, size_end))
    .limit(50)
    .sort("apk_size", "vt_scan_date")
    .limit(5)
    for size_start, size_end in itertools.pairwise(size_stage)
]

apk_lists = pl.collect_all(apk_lists)

for apk_list, ranges in zip(apk_lists, itertools.pairwise(size_stage)):
    assert (
        len(apk_list) == 5
    ), f"APK list between size {ranges} is not 50, got {len(apk_list)}"

pl.concat(apk_lists, how="vertical").sort("apk_size", "vt_scan_date").write_csv(
    "apk_list_per_size.csv"
)
