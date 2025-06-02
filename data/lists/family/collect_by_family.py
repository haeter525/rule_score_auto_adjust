# %%
# 載入 APK 清單
import polars as pl
import data_preprocess.apk as apk_lib

DATASET_PATHS = ["data/lists/maliciousAPKs_top_0.4_vt_scan_date.csv"]

dataset = pl.concat((apk_lib.read_csv(ds) for ds in DATASET_PATHS)).unique(
    "sha256", keep="any"
)
print(dataset.schema)
# 對於每個樣本，透過 VT 取得其 threat_label，並進一步拆解成 major, middle, minor
import data_preprocess.virust_total as vt
import tqdm
import re


with tqdm.tqdm(desc="Getting Threat Label", total=len(dataset)) as progress:
    ThreatLabels = pl.Struct(
        {
            "major_threat_label": pl.String(),
            "middle_threat_label": pl.String(),
            "minor_threat_label": pl.String(),
        }
    )

    def get_threat_label(sha256: str) -> dict[str, str]:
        try:
            report, status = vt.get_virus_total_report(sha256)

            threat_label = (
                report.get("data", {})
                .get("attributes", {})
                .get("popular_threat_classification", {})
                .get("suggested_threat_label", "./")
            )

            major, middle, minor = re.split("[./]", threat_label)
            return {
                "major_threat_label": major,
                "middle_threat_label": middle,
                "minor_threat_label": minor,
            }

        except BaseException as e:
            print(f"Error on {sha256}: {e}")
            return {
                "major_threat_label": "",
                "middle_threat_label": "",
                "minor_threat_label": "",
            }
        finally:
            progress.update()

    dataset = dataset.with_columns(
        pl.col("sha256")
        .map_elements(get_threat_label, return_dtype=ThreatLabels, strategy="threading")
        .alias("threat_labels")
    ).unnest("threat_labels")


dataset = dataset.with_columns(
    pl.col("middle_threat_label").str.replace("^kungfu$", "droidkungfu")
)

print(dataset.head(5))

print("major_threat_label:")
print(dataset["major_threat_label"].value_counts().sort(by="count", descending=True))

print("middle_threat_label:")
print(dataset["middle_threat_label"].value_counts().sort(by="count", descending=True))

# %%
# 挑選一個 middle threat label 作為要釋出規則的惡意程式家族
import click

target_middle_threat_label = click.prompt(
    "Enter target middle threat label",
    type=str,
    default="droidkungfu",
    show_default=True,
)

# 篩選出屬於此家族的樣本
target_dataset = dataset.filter(
    pl.col("middle_threat_label").eq(target_middle_threat_label)
)

print(target_dataset.head(5))
print(f"Num of apk: {len(target_dataset)}")

# %%
# 將屬於此家族的樣本寫入 csv
target_dataset.write_csv(f"data/lists/family/{target_middle_threat_label}.csv")

# %%
