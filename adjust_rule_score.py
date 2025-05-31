# %%
import datetime
from pathlib import Path
import os
import dotenv
import polars

dotenv.load_dotenv()

PATH_TO_DATASET = [
    Path("data/lists/benignAPKs_top_0.4_vt_scan_date.csv"),
    Path("data/lists/maliciousAPKs_top_0.4_vt_scan_date.csv"),
]

PATH_TO_RULE_LIST = [
    "/mnt/storage/rule_score_auto_adjust/data/rule_top_1000.csv"
]

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

@dataclass()
class ApkInfo:
    is_malicious: int = None
    path: Path = None
    analysis_result: dict[str,int] = None
    
    def __init__(self, sha256:str, is_malicious: int):
        self.is_malicious = is_malicious
        self.path = apk_lib.download(sha256, dry_run=True)
        self.analysis_result = analysis_result_lib.analyze_rules(sha256, self.path, rule_paths, dry_run=True)
    
sha256_table = pl.concat(list(map(apk_lib.load_list, PATH_TO_DATASET)))
apk_info = {
    sha256: ApkInfo(sha256, is_malicious)
    for sha256, is_malicious in tqdm.tqdm(sha256_table.rows(), total=len(sha256_table), desc="Preparing APK analysis results")
}

# %%
# Filter out apk not exits
to_drop = [
    sha256
    for sha256, info in apk_info.items()
    if info.path is None or not info.path.exists()
]

print(f"Drop {len(to_drop)} APKs since not exists, see \"to_drop\" for details")

for sha256 in to_drop:
    del apk_info[sha256]

# %%
# Filter out apk have no analysis result
to_drop = [
    sha256
    for sha256, info in apk_info.items()
    if any(v is None for v in info.analysis_result.values())
]

# %%
import polars

__dataframes = [
    polars.read_csv(dataset_path, has_header=True).select(
        ["sha256", "is_malicious"]
    )
    for dataset_path in PATH_TO_DATASET
]
combined = polars.concat(__dataframes)

sha256_list, isMalware = combined
print(f"Load apk list from {PATH_TO_DATASET}")
print(f"Num of apk: {len(sha256_list)}")



# %%
from data_preprocess import analysis_result

import polars as pl
import json

# %%
# Load Model From File
# import torch
# model.load_state_dict(
#     torch.load("manually_saved_model/0429_acc_84", weights_only=True)
# )
# model.eval()

# %%
# Move Model to GPU
import torch

assert torch.cuda.is_available()
device = torch.device("cuda")
model = model.to(device)


# %%
# Sample Dataset
import torch
from data_preprocess import apk, dataset
from tqdm import tqdm

rules = (
    pl.read_csv(
        PATH_TO_RULE_LIST,
        has_header=True,
        columns=["rule"],
        schema={"rule": pl.String()},
    )
    .to_series()
    .to_list()
)

# TODO - Test the new Dataset (Requires to use the filter here)
dataset = dataset.OldApkDataset(PATH_TO_DATASET, rules)
dataset.verify()

print(f"APK distribution: {dataset.apk_info['is_malicious'].value_counts()}")
print(f"Num of rules: {len(dataset.rules)}")

# %%
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=len(dataset), shuffle=True
)

# %%
# Build Model
from model import (
    RuleAdjustmentModel_NoTotalScore_Percentage,
    RuleAdjustmentModel,
)

model = RuleAdjustmentModel_NoTotalScore_Percentage(len(dataset.rules))
print(model)

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
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.1
    )
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
            model_path = "model_logs/model_{}_{}".format(
                timestamp, epoch_number
            )
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    return model_path


# %%
lrs = [1000000] * 1
best_model_param_path = None
# lrs = [1]
for lr in lrs:
    best_model_param_path = run_epochs(lr, epochs=1000)
    if best_model_param_path is not None:
        load_model_from_path(best_model_param_path, model)

print("Down")

# %%
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

apk_prediction = pl.DataFrame(
    {
        "sha256": dataset.apk_info["sha256"],
        "y_truth": y_truth,
        "y_pred_row": y_pred_row,
        "y_pred": y_pred,
    }
)

sha256 = dataset.apk_info["sha256"][0]
weights = dataset.load_cache(sha256).rename({"weights": sha256})

for sha256 in dataset.apk_info["sha256"][1:]:
    next_dataframe = (
        dataset.load_cache(sha256).rename({"weights": sha256}).drop("rule")
    )
    weights = weights.join(
        next_dataframe, on="index", how="left", maintain_order="left"
    )

row_rule_scores = model.get_rule_scores().cpu().detach().tolist()
rule_scores = pl.DataFrame(
    row_rule_scores, schema={"rule_score": pl.Float32}
).with_row_index()
weights_and_rule_scores = weights.join(
    rule_scores,
    on="index",
    how="left",
    maintain_order="left",
)
new_column_names = ["sha256", "y_truth", "y_pred_row", "y_pred"] + weights[
    "rule"
].to_list()

combined = (
    weights_and_rule_scores.drop("rule")
    .transpose(
        include_header=True,
        header_name="sha256",
        column_names=weights["rule"],
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
