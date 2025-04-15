# %%
import datetime
from pathlib import Path
from typing import Callable, Generator
import os
from tqdm.auto import tqdm
import dotenv
import polars

from data_preprocess import rule

dotenv.load_dotenv()

PATH_TO_DATASET = [
    Path(
        "data/dataset/benignAPKs_top_0.4_vt_scan_date_415d3ec3d4e0bf95e16d93649eab516b430abf4953d28617046cc998e391a6da.csv"
    ),
    Path(
        "data/dataset/maliciousAPKs_top_0.4_vt_scan_date_83d140381a28bc311b0423852b44efe3b6b859a8b6be7c6dac21fa9f0a833141.csv"
    ),
]
PATH_TO_RULE_LIST = (
    "/mnt/storage/data/rule_list/selected_rules_result_on_benign.csv"
)
PATH_TO_CONFIDENCES = "/mnt/storage/data/dataset/20250409/confidences"
NUM_OF_RULES = len(
    set(
        polars.read_csv(PATH_TO_RULE_LIST, has_header=True, columns=["rule"])
        .to_series()
        .to_list()
    )
)


# %%
from deprecated import deprecated


@deprecated(reason="Use apk.calculate_sha256 instead")
def calculate_sha256(file_path: str) -> str:
    """計算檔案的 SHA256 雜湊值"""
    import hashlib

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 每次讀取 4KB 以避免大檔案占用過多記憶體
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# %%
import polars

__dataframes = [
    polars.read_csv(dataset_path, has_header=True).select(
        ["sha256", "apk_path", "is_malicious"]
    )
    for dataset_path in PATH_TO_DATASET
]
combined = polars.concat(__dataframes)

sha256_list, apkList, isMalware = combined
print(f"Load apk list from {PATH_TO_DATASET}")
print(f"Num of apk: {len(sha256_list)}")

# %%
# 確認 Quark 分析結果都在
import data_preprocess.analysis_result as analysis_result

apk_infos = {
    sha256: {
        "analysis_result": (
            file
            if (file := analysis_result.get_file(sha256)).exists()
            else None
        )
    }
    for sha256 in sha256_list
}

analysis_report_missing = [
    sha256
    for sha256, info in apk_infos.items()
    if (file := info["analysis_result"]) is None
]

if len(analysis_report_missing) != 0:
    print(
        f"{len(analysis_report_missing)} analysis reports are missing. Check variable `analysis_report_missing` for details."
    )

for sha256 in analysis_report_missing:
    del apk_infos[sha256]

# %%
from data_preprocess import analysis_result

import polars as pl
import json


stage_weight_mapping = {
    0.0: 0.0,
    1.0: 0.0625,
    2.0: 0.125,
    3.0: 0.25,
    4.0: 0.5,
    5.0: 1.0,
}

# %%
# Build Model
from model import ScoringModel

model = ScoringModel(NUM_OF_RULES)
model

# %%
# Load Model From File
# import torch
# model.load_state_dict(torch.load("model_logs/model_20250411_073231_97", weights_only=True))
# model.eval()

# %%
# Sample Dataset
import functools
import torch
from data_preprocess import apk
from tqdm import tqdm

class ApkDataset(torch.utils.data.Dataset):
    def __init__(self, datasets: list[Path], rules: set[str]):

        # sha256, is_malicious, rule1, rule2, ...
        schema = {
            "sha256": polars.String,
            "apk_path": polars.String,
            "is_malicious": polars.Int32,
        }
        __dataset_df = [
            polars.read_csv(
                dataset_path,
                has_header=True,
                columns=["sha256", "apk_path", "is_malicious"],
                schema_overrides=schema,
            )
            for dataset_path in datasets
        ]
        self.apk_info = polars.concat(__dataset_df)

        # Filter out apk not exits
        self.apk_info = combined.filter(
            pl.col("sha256").map_elements(
                lambda x: apk.get(x).exists(), return_dtype=pl.Boolean
            )
        )

        # Filter out analysis result not exits
        self.apk_info = combined.filter(
            pl.col("sha256").map_elements(
                lambda x: analysis_result.get_file(x).exists(),
                return_dtype=pl.Boolean,
            )
        )

        # self.apk_info = sha256_table.join(
        #     combined, on="sha256", how="left"
        # ).to_dicts()

        # Index Rules
        self.rules = pl.DataFrame(
            rules, schema={"rule": pl.String}
        ).unique(["rule"]).with_row_index()

        # self.confidences_folder = confidences
        # # 取得所有 csv 檔案路徑
        # self.confidence_files = [
        #     os.path.join(self.confidences_folder, f"{apk_info['sha256']}.csv")
        #     for apk_info in self.apk_info
        # ]

    def save_cache(self, sha256: str, indexed_result: pl.DataFrame) -> None:
        cache_file = Path(os.getenv("DATASET_FOLDER")) / f"{sha256}.csv"
        indexed_result.write_csv(cache_file, include_header=True)

    def load_cache(self, sha256: str) -> pl.DataFrame | None:
        cache_file = Path(os.getenv("DATASET_FOLDER")) / f"{sha256}.csv"
        if not cache_file.exists():
            return None
        return polars.read_csv(cache_file, has_header=True)

    @functools.cache
    def __getitem__(self, index) -> torch.Tensor:
        sha256 = self.apk_info.item(row=index, column="sha256")

        # cache = self.load_cache(sha256)

        # if cache is not None:
        #     indexed_result = cache
        # else:
        result = (
            analysis_result.load_as_dataframe(sha256)
            .rename(
                {
                    "rule_name": "rule",
                }
            )
            .with_columns(
                pl.col("passing_stage").map_elements(
                    lambda x: stage_weight_mapping.get(float(x), 0.0),
                    return_dtype=pl.Float32,
                ),
                pl.col("rule").map_elements(
                    lambda x: x.split("/")[-1],
                    return_dtype=pl.String,
                )
            )
        )
        
        # drop row if the column "rule" is duplicate
        result = result.unique(subset=["rule"])
        
        indexed_result = self.rules.join(
            result, on="rule", how="left", maintain_order="left"
        )

        self.save_cache(sha256, indexed_result)

        indexed_result_torch = (
            indexed_result.select("passing_stage")
            .transpose()
            .to_torch(dtype=polars.Float32)
        )

        assert (
            indexed_result_torch.numel() != 0
        ), f"Indexed result for {sha256} is empty. {analysis_result.get_file(sha256)}"
        for i in range(indexed_result_torch.shape[1]):
            assert not (
                indexed_result.filter(pl.col("index").eq(i))
            ).is_empty(), f"{sha256} Missing index:{i}"

        assert indexed_result_torch.shape[1] == len(
            self.rules
        ), f"{sha256=}, {indexed_result_torch.shape=}, {len(self.rules)=}"

        expected_score = torch.tensor(
            self.apk_info[index]["is_malicious"], dtype=torch.float32
        )

        return indexed_result_torch, expected_score

    def __len__(self) -> int:
        return len(self.apk_info)

    def verify(self) -> bool:
        [n for n in tqdm(self, desc="Checking Dataset")]

rules = pl.read_csv(PATH_TO_RULE_LIST, has_header=True, columns=["rule"])\
    .to_series().to_list()

dataset = ApkDataset(PATH_TO_DATASET, rules)
dataset.verify()


# %%
dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=True)

# %%
# Loss Function
import torch

# 測試 Loss
loss_fn = torch.nn.HuberLoss()

# 測試數據
y_pred = torch.tensor([0.0, 50.0, 105.5, 160.0, 211.0])  # 模擬不同的預測值
y_exp = torch.tensor([0.0, 50.0, 105.5, 160.0, 0.0])
loss_value = loss_fn(y_pred, y_exp)

print(
    "Loss:", loss_value.item()
)  # 應該在 0 和 211 附近 loss 較小，在 105.5 附近 loss 較大

# my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.05)

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
        if i % 1000 == 999:
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
        print("EPOCH {}:".format(epoch_number + 1))
        print(f"learning rate: {optimizer.param_groups[0]['lr']}")

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
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)

                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

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
            model_path = "model_logs/model_{}_{}".format(
                timestamp, epoch_number
            )
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    tensor = model.pos_rule_scores(model.raw_rule_scores).transpose(-2, 1)[0]
    import polars as pl

    # 將 tensor 轉換為 numpy array
    numpy_array = tensor.detach().numpy()

    # 創建 DataFrame
    optimized_rule_score = pl.DataFrame(numpy_array.round(2)).rename(
        {"column_0": "rule_score"}
    )
    optimized_rule_score.write_csv(
        "optimized_rule_score.csv", include_header=True
    )

    return model_path


# %%
lrs = [5, 5, 1, 1, 1, 0.5]
# lrs = [1]
for lr in lrs:
    best_model_param_path = run_epochs(lr)
    if best_model_param_path is not None:
        load_model_from_path(best_model_param_path, model)

print("Down")

# %%
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def model_inference(model, x):
    with torch.no_grad():
        return model(x).item()


x_input, y_truth = [], []
for x, y in dataloader:
    x_input.append(x)
    y_truth.append(y)

y_pred_row = [model_inference(model, x) for x in x_input]

y_pred = [1 if y_row > 0.5 else 0 for y_row in y_pred_row]

accuracy = accuracy_score(y_truth, y_pred)
print(f"{accuracy=}")

# %%
# Apply Rule scores
if input("Apply the rule scores? (Y/N)").lower() == "N":
    sys.exit(0)

# PATH_TO_RULES = "/Users/shengfeng/codespace/quark-rules/rules"
rule_index = pl.read_csv(
    "data/rule_top_1000_index.csv",
    has_header=True,
    columns=["index", "rule_path"],
)

optimized_rule_score = optimized_rule_score.with_row_index()

combine = optimized_rule_score.join(on="index", how="left")

for row in combine.to_dicts():
    ruleId, rule_path = row["index"], row["rule_path"]
    if not os.path.exists(rule_path):
        continue

    value = round(row["rule_score"], 2)
    print(f"更新 {rule_path} 規則分數為 {value}")
    with open(rule_path, "r") as f:
        rule_data = json.loads(f.read())
        rule_data["score"] = value

    with open(rule_path, "w") as f:
        json.dump(rule_data, f, indent=4)
# %%

# remove 9328A73772559AE20165A0E48A484554F5746BD6505C65C6E7EAE02EAFDE76B7,/mnt/f2ce377e-d586-41c5-bdf7-fad4a20f5b98/generate_rules/data/apks/9328A73772559AE20165A0E48A484554F5746BD6505C65C6E7EAE02EAFDE76B7.apk,1
# remove C687E2F0B4992BD368DF0C24B76943C99AC3EB9E4E8C13422EBF1A872A06070A,/mnt/f2ce377e-d586-41c5-bdf7-fad4a20f5b98/generate_rules/data/apks/C687E2F0B4992BD368DF0C24B76943C99AC3EB9E4E8C13422EBF1A872A06070A.apk,1
# remove 77F7A39A5B5367A5CE12E6B35B999FF1571821FB563764E45615FEB4CEB86FC0,/mnt/f2ce377e-d586-41c5-bdf7-fad4a20f5b98/generate_rules/data/apks/77F7A39A5B5367A5CE12E6B35B999FF1571821FB563764E45615FEB4CEB86FC0.apk,1
# remove 33C9604DB40D4453935A7076BA5018061BC906ACE486BC8C2D7197EC08EB8951,/mnt/f2ce377e-d586-41c5-bdf7-fad4a20f5b98/generate_rules/data/apks/33C9604DB40D4453935A7076BA5018061BC906ACE486BC8C2D7197EC08EB8951.apk,1
