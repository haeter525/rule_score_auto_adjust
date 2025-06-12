# %%
import datetime
from pathlib import Path
import dotenv

dotenv.load_dotenv()

PATH_TO_DATASET = [
    "data/lists/family/droidkungfu.csv",
    # "/mnt/storage/rule_score_auto_adjust/data/lists/family/apk-sample.csv",
    "data/lists/benignAPKs_top_0.4_vt_scan_date.csv",
]

PATH_TO_RULE_LIST = ["/mnt/storage/data/rule_to_release/0611/all_rules.csv"]

# %%
import data_preprocess.rule as rule_lib
import polars as pl

rules = pl.concat(list(map(rule_lib.load_list, PATH_TO_RULE_LIST)))["rule"].to_list()
rule_paths = [rule_lib.get(r) for r in rules]

# %%
from dataclasses import dataclass
import data_preprocess.apk as apk_lib
import data_preprocess.analysis_result as analysis_result_lib
import tqdm
from pathlib import Path
import polars as pl


@dataclass()
class ApkInfo:
    sha256: str
    is_malicious: int
    path: Path | None
    analysis_result: dict[str, int] | None

    def __init__(self, sha256: str, is_malicious: int):
        self.sha256 = sha256
        self.is_malicious = is_malicious
        self.path = apk_lib.download(sha256, dry_run=True)
        
        if self.path is not None:
            self.analysis_result = analysis_result_lib.analyze_rules(
                sha256, self.path, rule_paths, dry_run=True
            )
        else:
            self.analysis_result = {}

sha256_table = pl.concat(list(map(apk_lib.read_csv, PATH_TO_DATASET)))
original_apk_info_list = [
    ApkInfo(sha256, is_malicious)
    for sha256, is_malicious in tqdm.tqdm(
        sha256_table.rows(),
        total=len(sha256_table),
        desc="Preparing APK analysis results",
    )
]

# %%
# Prepare to clean the data
from typing import Generator, Iterable, Callable


def show_and_filter(apk_info_list: Iterable[ApkInfo], filter_func: Callable[[ApkInfo], bool]) -> Generator[ApkInfo, None, None]:
    to_drop = []
    
    for apk_info in apk_info_list:
        if filter_func(apk_info):
            to_drop.append(apk_info.sha256)
        else:
            yield apk_info
    
    print(f'Drop {len(to_drop)} APKs, set checkpoint in this function and see "to_drop" for details')

apk_info_list = original_apk_info_list
# %%
print("Filter out apk not exits.")
apk_not_exist = lambda info: info.path is None or not info.path.exists()
apk_info_list = show_and_filter(apk_info_list, apk_not_exist)
apk_info_list = list(apk_info_list)

# %%
print("Filter out apk have no analysis result.")
apk_no_analysis_result = lambda info: any(
    v < 0 for v in info.analysis_result.values()
)
apk_info_list = show_and_filter(apk_info_list, apk_no_analysis_result)
apk_info_list = list(apk_info_list)

# %%
print("Filter out apk didn't pass 5 stage on any rule.")
apk_no_passing_5_stage = lambda info: not any(
    v >= 5 for v in info.analysis_result.values()
)
apk_info_list = show_and_filter(apk_info_list, apk_no_passing_5_stage)
apk_info_list = list(apk_info_list)

# %%
print("Filter out apks that Quark failed to analyze due to memory issue.")

memory_issue_apks = {
    "00015824995BC2F452BBDE2833F79423A8DC6DA8364A641DFB6D068D44C557DF"
}

apk_info_list = show_and_filter(apk_info_list, lambda info: info.sha256 in memory_issue_apks)
apk_info_list = list(apk_info_list)

# %%
print("Balance the dataset by removing extra benign APKs.")

from itertools import count
benign_counter = count()
num_of_malware = sum(1 for info in apk_info_list if info.is_malicious == 1)

malicious_or_enough_benign = lambda info: info.is_malicious != 1 and next(benign_counter) >= num_of_malware
apk_info_list = show_and_filter(apk_info_list, malicious_or_enough_benign)
apk_info_list = list(apk_info_list)

# %%
# 確認所有惡意樣本都還存在

malware_sha256 = {
    apk.sha256 for apk in original_apk_info_list if apk.is_malicious == 1
}
all_sha256s = {apk.sha256 for apk in apk_info_list}

missing_malware = [
    m for m in malware_sha256 if m not in all_sha256s
]

assert len(missing_malware) == 0 , f"Some malware is missing, check \"missing_malware\""
    
# %%
# 確認所有內建樣本都還存在（apk-sample.csv）

builtin_sample = apk_lib.read_csv("/mnt/storage/rule_score_auto_adjust/data/lists/family/apk-sample.csv")["sha256"].to_list()

missing_apks = [
    sha256 for sha256 in builtin_sample if sha256 not in all_sha256s
]
assert len(missing_malware) == 0 , f"Some apk is missing, check \"missing_apks\""

# %%
# Build Dataset
from data_preprocess import dataset as dataset_lib

dataset = dataset_lib.ApkDataset(
    sha256s = [apk.sha256 for apk in apk_info_list],
    is_malicious = [apk.is_malicious for apk in apk_info_list],
    rules = rules,
)

print(f"Num of APK: {len(dataset)}")
print(f"APK distribution: {dataset.apk_info['is_malicious'].value_counts()}")
print(f"Num of rules: {len(dataset.rules)}")

# %%
# Preload Dataset into cache
dataset.preload()

# %%
# Create dataloader
from torch.utils.data.dataloader import DataLoader
# dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

# %%
# Build Model
from model import (
    RuleAdjustmentModel_NoTotalScore_Percentage,
    RuleAdjustmentModel,
)

model = RuleAdjustmentModel(len(dataset.rules))
# model = RuleAdjustmentModel_NoTotalScore_Percentage(len(dataset.rules))
print(model)
# %%
# Load Model From File
# import torch
# model.load_state_dict(
#     torch.load("./model_logs/model_20250602_024356_99", weights_only=True)
# )
# model.eval()

# %%
# Move Model to GPU
import torch

assert torch.cuda.is_available()
device = torch.device("cuda")
model = model.to(device)


# %%
# Loss Function
import torch

# 測試 Loss
loss_fn = torch.nn.BCELoss().to(device)

# 測試數據
y_pred = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0])  # 模擬不同的預測值
y_exp = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0])
loss_value = loss_fn(y_pred, y_exp)

print("Loss:", loss_value.item())
best_model_param_path = None


# %%
# Train
def train_one_epoch(epoch_index, tb_writer, optimizer):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        # my_lr_scheduler.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 10000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


# %%
# Initializing in a separate cell so we can easily add more epochs to the same run
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime


def load_model_from_path(model_path, model):
    model.load_state_dict(torch.load(model_path, weights_only=True))


def run_epochs(learning_rate, epochs=100):
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
    epoch_number = 0

    EPOCHS = epochs
    model_path = None

    best_vloss = 1_000_000.0

    from tqdm import tqdm

    for epoch in tqdm(list(range(EPOCHS))):
        # print("EPOCH {}:".format(epoch_number + 1))
        # print(f"learning rate: {optimizer.param_groups[0]['lr']}")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, optimizer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(dataloader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)

                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(
            "EP {}, LR {}, LOSS train {} valid {}".format(
                epoch_number + 1,
                optimizer.param_groups[0]["lr"],
                avg_loss,
                avg_vloss,
            )
        )

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_folder = Path("model_logs")
            model_folder.mkdir(parents=True, exist_ok=True)
            model_path = "model_logs/model_{}_{}".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    return model_path


# %%
accuracy = 0
for i in range(2):
    if accuracy == 1.0:
        break
    
    lrs = [10] * 1
    best_model_param_path = None
    # lrs = [1]
    for lr in lrs:
        best_model_param_path = run_epochs(lr, epochs=100)
        if best_model_param_path is not None:
            load_model_from_path(best_model_param_path, model)

    print("Down")

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    if best_model_param_path is not None:
        load_model_from_path(best_model_param_path, model)


    def model_inference(model, x):
        with torch.no_grad():
            return model(x).item()


    x_input, y_truth = [], []
    for x, y in dataset:
        x_input.append(x.to(device))
        y_truth.append(y)

    y_pred_row = [model_inference(model, x) for x in x_input]

    y_pred = [1 if y_row > 0.5 else 0 for y_row in y_pred_row]

    accuracy = accuracy_score(y_truth, y_pred)
    precision = precision_score(y_truth, y_pred)
    recall = recall_score(y_truth, y_pred)
    f1 = f1_score(y_truth, y_pred)
    print(f"{accuracy=}")
    print(f"{precision=}")
    print(f"{recall=}")
    print(f"{f1=}")

# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def model_inference(model, x):
    with torch.no_grad():
        return model(x).item()

def model_calculate(model, x):
    with torch.no_grad():
        return model.calculate_apk_scores(x)

x_input, y_truth = [], []
for x, y in dataset:
    x_input.append(x.to(device))
    y_truth.append(y)

y_pred_row = [model_inference(model, x) for x in x_input]
y_score = [model_calculate(model, x) for x in x_input]

y_pred = [1 if y_row > 0.5 else 0 for y_row in y_pred_row]

accuracy = accuracy_score(y_truth, y_pred)
precision = precision_score(y_truth, y_pred)
recall = recall_score(y_truth, y_pred)
f1 = f1_score(y_truth, y_pred)
print(f"{accuracy=}")
print(f"{precision=}")
print(f"{recall=}")
print(f"{f1=}")
# %%
# Output adjusted scores and prediction for each apk

apk_prediction = pl.DataFrame(
    {
        "sha256": dataset.apk_info["sha256"],
        "y_truth": y_truth,
        "y_score": y_score,
        "y_pred": y_pred,
    }
)

# Prepare analysis results
rule_paths = [rule_lib.get(rule) for rule in rules]

weight_dicts = (
    analysis_result_lib.analyze_rules(
        sha256,
        apk_lib.download(sha256, dry_run=True),
        rule_paths,
        dry_run=True
    ) | {"sha256": sha256}
    for sha256 in dataset.apk_info["sha256"]
)
weight_dfs = (
    pl.DataFrame(weight) for weight in weight_dicts
)
weights = pl.concat(weight_dfs, how="vertical")
weights = weights.transpose(include_header=True, column_names="sha256", header_name="rule")

# Prepare adjusts rule scores
# rule_scores = pl.DataFrame(
#     {"rule_score": model.get_rule_scores().cpu().detach().tolist(), "rule": rules}
# ).with_row_index()

rule_scores = dataset.rules.join(
    pl.DataFrame({"rule_score": model.get_rule_scores().cpu().detach().tolist()}).with_row_index(),
    on="index",
    how="left"
)
# %%
# Combine rule_scores and weights
weights_and_rule_scores = rule_scores.join(weights, on="rule", how="left", maintain_order="left")

new_column_names = ["sha256", "y_truth", "y_score", "y_pred"] + weights[
    "rule"
].to_list()

combined = (
    weights_and_rule_scores.transpose(
        include_header=True,
        header_name="sha256",
        column_names="rule",
    )
    .join(apk_prediction, on="sha256", how="full", maintain_order="left")
    .select(new_column_names)
)

combined.write_csv("apk_prediction.csv", include_header=True)
# %%
# Apply Rule scores
# if input("Apply the rule scores? (Y/N)").lower() == "N":
#     sys.exit(0)

# # PATH_TO_RULES = "/Users/shengfeng/codespace/quark-rules/rules"
# rule_index = pl.read_csv(
#     "data/rule_top_1000_index.csv",
#     has_header=True,
#     columns=["index", "rule_path"],
# )

# optimized_rule_score = rule_index.with_row_index()

# combine = rule_index.join(on="index", how="left")

# for row in combine.to_dicts():
#     ruleId, rule_path = row["index"], row["rule_path"]
#     if not os.path.exists(rule_path):
#         continue

#     value = round(row["rule_score"], 2)
#     print(f"更新 {rule_path} 規則分數為 {value}")
#     with open(rule_path, "r") as f:
#         rule_data = json.loads(f.read())
#         rule_data["score"] = value

#     with open(rule_path, "w") as f:
#         json.dump(rule_data, f, indent=4)

# %%
# Apply Rule to Quark Rules
# Load prediction
import polars as pl
apk_prediction = pl.read_csv("/mnt/storage/rule_score_auto_adjust/apk_prediction.csv")
# %%
# Extract rule score
apk_prediction = apk_prediction.transpose(include_header=True, column_names="sha256")
apk_prediction = apk_prediction.select(["column","rule_score"])
apk_prediction = apk_prediction.filter(pl.col("column").str.ends_with(".json"))
# %%
# Apply to Quark rules
from pathlib import Path
import json
path_to_quark_rules = Path("/mnt/storage/quark-rules/rules")

for rule_name, score in apk_prediction.iter_rows():
    rule_path = path_to_quark_rules / rule_name
    assert rule_path.exists(), f"{rule_path} doesn't exists."
    
    round_score = round(score, 2)
    # round_score = score
    print(f"更新 {rule_path} 規則分數為 {round_score}")
    
    with open(rule_path, "r") as f:
        rule_data = json.loads(f.read())
        rule_data["score"] = round_score

    with open(rule_path, "w") as f:
        json.dump(rule_data, f, indent=4)
# %%
