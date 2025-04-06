from pathlib import Path
import polars as pl

def load_genome_apks(apk_list = "data/latest.csv", market = "genome"):
    assert Path(apk_list).exists(), "Androzoo APK list didn't not exists."
    return pl.scan_csv(apk_list).filter(pl.col("markets").str.contains(market))

market_apks = load_genome_apks()

# 创建软链接
local_rule_path = Path("path/to/local/rule")
link_path = Path("path/to/link")

# 如果链接已存在，先删除
if link_path.exists():
    link_path.unlink()

# 创建新的软链接
link_path.symlink_to(local_rule_path)

malware_apks = (market_apks.sort(pl.col("vt_scan_date"), descending=True).head(len(market_apks)*0.1)) 