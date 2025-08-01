from pathlib import Path
import polars as pl
import tqdm
import data_preprocess.rule as rule_lib
import json
import dotenv

dotenv.load_dotenv()
import os
from generate_rule_description import BehaviorDescriptionAgent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RULE_LISTS = [
    Path("/mnt/storage/data/rule_to_release/golddream/rule_added.csv"),
    Path("/mnt/storage/data/rule_to_release/default_rules.csv"),
]

rule_df = pl.concat(
    [
        pl.read_csv(rule_list, has_header=True, columns=["rule"])
        for rule_list in RULE_LISTS
    ],
    how="vertical",
)

agent = BehaviorDescriptionAgent(OPENAI_API_KEY)

with tqdm.tqdm(desc="Getting rule description", total=len(rule_df)) as progress:

    Schema = pl.Struct(
        [
            pl.Field("description", pl.String()),
            pl.Field("api1", pl.String()),
            pl.Field("api2", pl.String()),
            pl.Field("label", pl.String()),
        ]
    )

    def get_description(rule: str):
        progress.update()
        rule_path = rule_lib.get(rule)
        with rule_path.open("r") as content:
            json_obj = json.loads(content.read())
            api_pair = json_obj["api"]

        api1 = api_pair[0]["class"] + api_pair[0]["method"] + api_pair[0]["descriptor"]
        api2 = api_pair[1]["class"] + api_pair[1]["method"] + api_pair[1]["descriptor"]

        return {
            "description": json_obj["crime"],
            "api1": api1,
            "api2": api2,
            "label": "|".join(json_obj.get("label", [])),
        }

    rule_df = (
        rule_df.with_columns(
            pl.col("rule")
            .map_elements(get_description, return_dtype=Schema, strategy="thread_local")
            .alias("combined")
        )
        .unnest("combined")
        .select(["rule", "description", "label", "api1", "api2"])
    )

print(rule_df.head(5))
rule_df.write_csv("./rule_reviews.csv")
