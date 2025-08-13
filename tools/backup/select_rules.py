# %%
import data_preprocess.rule as rule_lib

RULE_SOURCE = "/mnt/storage/data/rule_to_release/0627/unselected_rules.csv"

rules = rule_lib.load_list(RULE_SOURCE)
rules
# %%
import polars as pl

APK_PREDICTION = "/mnt/storage/rule_score_auto_adjust/apk_prediction.csv"

prediction = (
    pl.read_csv(APK_PREDICTION, has_header=True)
    .filter(pl.col("sha256").eq("rule_score"))
    .drop(["y_truth", "y_pred_row", "y_pred"], strict=False)
    .transpose(include_header=True, header_name="rule", column_names="sha256")
    .filter(pl.col("rule").ne("y_score"))
)
prediction = prediction.sort(by="rule_score", descending=True)
prediction

# %%

# %%
rules_with_score = (
    rules.join(other=prediction, on="rule", how="left")
    .fill_nan(float("-inf"))
    .fill_null(float("-inf"))
    .sort(by="rule_score", descending=True)
)
rules_with_score
# %%
# Select first n rules
import math

n = 0.985
selected_rules = rules_with_score.head(math.floor(len(rules_with_score) * (1 - n)))
selected_rules

# %%
# 規則權重為負數，表示無法偵測惡意樣本，為了節省時間，應該只挑權重為正的規則
min_score = selected_rules.min().item(0, "rule_score")
assert (
    min_score > 0
), f'Current threshold ({min_score}) is less or equal to 0. It\'s suggested to only select rules with positive wights since only those rules are able to detect malicious behavior. Consider adjust variable "n" to a higher value.'
# %%
out_path = f"/mnt/storage/data/rule_to_release/0627/rules_first_{str(n).replace(".", "_")}.csv"
selected_rules.select("rule","rule_score").write_csv(out_path)
print(f"Output to {out_path}")
selected_rules
# %%
