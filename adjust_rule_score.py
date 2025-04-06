# %%
import datetime
from pathlib import Path
from typing import Callable, Generator

PATH_TO_DATASET = [
    "data/dataset/benignAPKs_top_0.4_vt_scan_date_415d3ec3d4e0bf95e16d93649eab516b430abf4953d28617046cc998e391a6da.csv",
    "data/dataset/maliciousAPKs_top_0.4_vt_scan_date_83d140381a28bc311b0423852b44efe3b6b859a8b6be7c6dac21fa9f0a833141.csv"
]
PATH_TO_RULES = "data/rule_top_1000"
PATH_TO_REPORTS = "data/reports"
PATH_TO_CONFIDENCES = "data/confidences"
NUM_OF_RULES = sum(1 for _ in Path(PATH_TO_RULES).glob("[!.]*.json"))
print(f"Load rules from {PATH_TO_RULES}")
print(f"Num of rules: {NUM_OF_RULES}")

# %%
# Build APK List
# import sys
# import os
# import os.path

# apk_paths = sys.argv[1:] if len(sys.argv) > 1 else []

# def has_invalid_apk(apk_files):
#     return [
#         apk
#         for apk in apk_files if (not os.path.exists(apk)) or (os.path.splitext(apk)[-1] != ".apk")
#     ]

# if invalid := has_invalid_apk(apk_paths):
#     print(f"{invalid} is invalid APK.")


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
combined = polars.DataFrame()
for dataset_path in PATH_TO_DATASET:
    combined = combined.vstack(polars.read_csv(dataset_path, has_header=True, columns=["sha256", "apk_path", "is_malicious"]))

sha256_list, apkList, isMalware = combined
print(f"Load apk list from {PATH_TO_DATASET}")
print(f"Num of apk: {len(sha256_list)}")

# %%
# 確認 Quark 分析結果都在
PATH_TO_ANALYSIS_RESULT = Path("data/analysis_results")

get_analysis_report_path = lambda sha256: PATH_TO_ANALYSIS_RESULT / f"{sha256}.apk_progress.json"

get_analysis_report_path_list: Callable[[], Generator[Path]] = lambda: (
    get_analysis_report_path(sha256) for sha256 in sha256_list
)

analysis_report_missing = [
    analysis_report
    for analysis_report in get_analysis_report_path_list()
    if not analysis_report.exists()
]

if len(analysis_report_missing) != 0:
    print(f"{len(analysis_report_missing)} analysis reports are missing. Check variable `analysis_report_missing` for details.")
# %%
apk_not_missing = [
    apk
    for report in analysis_report_missing
    if (apk := Path("/Volumes/SF_Storage/apks") / f"{report.stem.split(".")[0]}.apk").exists()
]

def read_from_json(path: Path):
    with path.open("r") as file:
        content = json.load(file)
    return content

# %%
# Extract confidences


get_analysis_report_jsons_exist = lambda: (
    read_from_json(path)
    for path in get_analysis_report_path_list()
    if path.exists()
)

sha256_list_exist = [
    sha256
    for sha256 in sha256_list
    if get_analysis_report_path(sha256).exists()
]

# %%
# Check Json Format

read_json_error = []
for path in get_analysis_report_path_list():
    if not path.exists():
        continue

    try:
        read_from_json(path)
    except Exception as e:
        read_json_error.append(path)
    
if len(read_json_error) != 0:
    print(f"{len(read_json_error)} analysis report are not in proper JSON format. Check `read_json_error` for more details.")

# %%

def convert_to_confidence(analysis_report_json):
    return {
        items[0]: items[1] 
        for items in analysis_report_json # rule_path, confidence
        if len(items) >= 2
    }

get_confidence_contents = lambda: (
    convert_to_confidence(report)
    for report in get_analysis_report_jsons_exist()
)

import polars as pl
import json


stage_weight_mapping = {
    0.0: 0.0,
    1.0: 0.0625,
    2.0: 0.125,
    3.0: 0.25,
    4.0: 0.5,
    5.0: 1.0
}

get_confidence_files = lambda: (
    Path(PATH_TO_CONFIDENCES) / f"{apk_hash}.csv"
    for apk_hash in sha256_list_exist
)

from itertools import islice

# sha256_list_exist = list(islice(sha256_list_exist, 480))
# confidence_contents = islice(get_confidence_contents(), 480)
# confidence_files = list(islice(confidence_files, 480))

# %%
counter = 0
write_list = []
for sha256, confidence_content, output_file in zip(sha256_list_exist, get_confidence_contents(), get_confidence_files(), strict=True):
    if output_file.exists():
        counter += 1
        if counter > 480:
            break
        write_list.append((sha256, output_file))
        continue
    
    # 將 rule_confidences_mapping 轉換為 DataFrame 並轉置
    confidence_table = polars.DataFrame(confidence_content)
    
    if confidence_table.is_empty():
        continue
    else:
        confidence_table = confidence_table.transpose(include_header=True)
    
    # 重命名列
    confidence_table = confidence_table.rename({
        "column": "rule_path",
        "column_0": "confidence"
    })
    
    rule_index_table = pl.read_csv("data/rule_top_1000_index.csv", has_header=True)
    combined = rule_index_table.join(confidence_table, on="rule_path", how="left")
    combined = combined.with_columns(pl.col("confidence").fill_null(0.0))
    combined = combined.with_columns(pl.col("confidence").map_elements(lambda x: stage_weight_mapping.get(x, 0.0), return_dtype=float).alias("weight"))
    combined = combined.with_columns(pl.col("index").alias("rule_id"))
    combined = combined.rename({
        "confidence": "passing_stage",
        "weight": "confidence"
    })
    combined = combined.sort(pl.col("index"))
    # 寫入 CSV 檔案
    print(f"Write confidence to : {output_file}")
    combined.write_csv(output_file, include_header=True)
    counter += 1
    write_list.append((sha256, output_file))
    if counter > 480:
        break

print("Down")

sha256_list_exist = [sha256 for sha256, _ in write_list]

# %%
# Build Model
import torch
import torch.nn as nn

class ExpLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight= nn.Parameter(torch.zeros(out_features, in_features))
        self.bias= nn.Parameter(torch.zeros(out_features))

    def forward(self, input):
        return nn.functional.linear(input, self.weight.exp(), self.bias.exp())

class ScoringModel(torch.nn.Module):
    def __init__(self, num_of_rules: int) -> None:
        assert num_of_rules > 0

        super(ScoringModel, self).__init__()
        self.raw_rule_scores = torch.nn.Parameter(torch.randn((num_of_rules, 1), dtype=torch.float32))
        self.pos_rule_scores = torch.nn.Softplus()

    def forward(self, confidence):
        return torch.matmul(confidence, self.pos_rule_scores(self.raw_rule_scores))
    torch.nn.Linear

model = ScoringModel(NUM_OF_RULES)
model

# %%
# Sample Dataset
import torch
class ApkDataset(torch.utils.data.Dataset):
    def __init__(self, num_of_rule, sha256_list, confidences = PATH_TO_CONFIDENCES):
        self.num_of_rule = num_of_rule

        schema = {
            "sha256": polars.String,
            "apk_path": polars.String,
            "is_malicious": polars.Int32
        }
        # self.apk_info = polars.read_csv(apk_list, has_header=True, schema=schema).to_dicts()
        combined = polars.DataFrame()
        for dataset_path in PATH_TO_DATASET:
            combined = combined.vstack(polars.read_csv(dataset_path, has_header=True, columns=["sha256", "apk_path", "is_malicious"], schema_overrides=schema))
        
        sha256_table = polars.DataFrame(sha256_list_exist, schema={"sha256":str})

        self.apk_info = sha256_table.join(combined, on="sha256", how="left").to_dicts()

        self.confidences_folder = confidences
        # 取得所有 csv 檔案路徑
        self.confidence_files = [
            os.path.join(self.confidences_folder, f"{apk_info['sha256']}.csv")
            for apk_info in self.apk_info
        ]

    def __getitem__(self, index) -> torch.Tensor:
        selected_file = self.confidence_files[index]
        confidence_table = polars.read_csv(selected_file, has_header=True, columns=["index", "confidence"])
        confidence = confidence_table.select(pl.col("confidence")).transpose().to_torch(dtype=polars.Float32)
        assert (not confidence_table.is_empty()), f"{selected_file=} is empty"
        for i in range(confidence.shape[1]):
            assert not (confidence_table.filter(pl.col("index").eq(i))).is_empty(), f"{selected_file} Missing index:{i}"

        assert confidence.shape[1] == self.num_of_rule, f"{selected_file=}, {confidence.shape=}, {self.num_of_rule=}"
        expected_score = self.num_of_rule * self.apk_info[index]['is_malicious']
        return confidence, torch.tensor(expected_score, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.confidence_files)
    

dataset = ApkDataset(NUM_OF_RULES, sha256_list=sha256_list_exist)

from tqdm import tqdm
[n for n in tqdm(dataset)]

[Path("data/confidences")/f"{sha256}.apk" for sha256 in sha256_list]

# %%
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# %%
# Loss Function
import torch

# 測試 Loss
loss_fn = torch.nn.HuberLoss()

# 測試數據
y_pred = torch.tensor([0.0, 50.0, 105.5, 160.0, 211.0])  # 模擬不同的預測值
y_exp = torch.tensor([0.0, 50.0, 105.5, 160.0, 0.0])
loss_value = loss_fn(y_pred, y_exp)

print("Loss:", loss_value.item())  # 應該在 0 和 211 附近 loss 較小，在 105.5 附近 loss 較大
# %%
# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=5, momentum=0.1)

# my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.05)

# %%
# Train

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

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
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# %%
# Initializing in a separate cell so we can easily add more epochs to the same run
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 100

best_vloss = 1_000_000.

from tqdm import tqdm
for epoch in tqdm(list(range(EPOCHS))):
    print('EPOCH {}:'.format(epoch_number + 1))
    print(f"learning rate: {optimizer.param_groups[0]['lr']}")

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


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
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_logs/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
# %%

tensor = model.pos_rule_scores(model.raw_rule_scores).transpose(-2,1)[0]
import polars as pl

# 將 tensor 轉換為 numpy array
numpy_array = tensor.detach().numpy()

# 創建 DataFrame
optimized_rule_score = pl.DataFrame(numpy_array.round(2)).rename({"column_0":"rule_score"})
optimized_rule_score.write_csv("optimized_rule_score.csv", include_header=True)


# %%
# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.1)

# my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.05)

# %%
# Train

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

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
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# %%
# Initializing in a separate cell so we can easily add more epochs to the same run
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0



best_vloss = 1_000_000.

from tqdm import tqdm
for epoch in tqdm(list(range(EPOCHS))):
    print('EPOCH {}:'.format(epoch_number + 1))
    print(f"learning rate: {optimizer.param_groups[0]['lr']}")

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


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
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_logs/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
# %%

tensor = model.pos_rule_scores(model.raw_rule_scores).transpose(-2,1)[0]
import polars as pl

# 將 tensor 轉換為 numpy array
numpy_array = tensor.detach().numpy()

# 創建 DataFrame
optimized_rule_score = pl.DataFrame(numpy_array.round(2)).rename({"column_0":"rule_score"})
optimized_rule_score.write_csv("optimized_rule_score.csv", include_header=True)


# %%
# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.1)

# my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.05)

# %%
# Train

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

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
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# %%
# Initializing in a separate cell so we can easily add more epochs to the same run
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0


best_vloss = 1_000_000.

from tqdm import tqdm
for epoch in tqdm(list(range(EPOCHS))):
    print('EPOCH {}:'.format(epoch_number + 1))
    print(f"learning rate: {optimizer.param_groups[0]['lr']}")

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


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
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_logs/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
# %%

tensor = model.pos_rule_scores(model.raw_rule_scores).transpose(-2,1)[0]
import polars as pl

# 將 tensor 轉換為 numpy array
numpy_array = tensor.detach().numpy()

# 創建 DataFrame
optimized_rule_score = pl.DataFrame(numpy_array.round(2)).rename({"column_0":"rule_score"})
optimized_rule_score.write_csv("optimized_rule_score.csv", include_header=True)
# %%
# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1)

# my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.05)

# %%
# Train

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

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
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(dataloader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# %%
# Initializing in a separate cell so we can easily add more epochs to the same run
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

best_vloss = 1_000_000.

from tqdm import tqdm
for epoch in tqdm(list(range(EPOCHS))):
    print('EPOCH {}:'.format(epoch_number + 1))
    print(f"learning rate: {optimizer.param_groups[0]['lr']}")

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


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
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_logs/model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1
# %%

tensor = model.pos_rule_scores(model.raw_rule_scores).transpose(-2,1)[0]
import polars as pl

# 將 tensor 轉換為 numpy array
numpy_array = tensor.detach().numpy()

# 創建 DataFrame
optimized_rule_score = pl.DataFrame(numpy_array.round(2)).rename({"column_0":"rule_score"})
optimized_rule_score.write_csv("optimized_rule_score.csv", include_header=True)

# %%
from tqdm import tqdm

with torch.no_grad():
    counter = 0
    hit = 0
    for i, vdata in tqdm(enumerate(dataloader)):
        vinputs, vlabels = vdata
        voutputs = 1 if model(vinputs) > 0.5 else 0
        
        if voutputs == vlabels:
            hit += 1

        counter += 1

accuracy = hit / counter
print(f"{accuracy=}")

# %%
# Apply Rule scores
if input("Apply the rule scores? (Y/N)").lower() == "N":
    sys.exit(0)

# PATH_TO_RULES = "/Users/shengfeng/codespace/quark-rules/rules"
rule_index = pl.read_csv("data/rule_top_1000_index.csv", has_header=True, columns=["index", "rule_path"])

optimized_rule_score = optimized_rule_score.with_row_index()

combine = optimized_rule_score.join(on="index", how="left")
    
for row in combine.to_dicts():
    ruleId, rule_path = row["index"], row["rule_path"]
    if not os.path.exists(rule_path):
        continue
        
    value = round(row["rule_score"], 2)
    print(f"更新 {rule_path} 規則分數為 {value}")
    with open(rule_path, 'r') as f:
        rule_data = json.loads(f.read())
        rule_data['score'] = value
        
    with open(rule_path, 'w') as f:
        json.dump(rule_data, f, indent=4)
# %%

# remove 9328A73772559AE20165A0E48A484554F5746BD6505C65C6E7EAE02EAFDE76B7,/mnt/f2ce377e-d586-41c5-bdf7-fad4a20f5b98/generate_rules/data/apks/9328A73772559AE20165A0E48A484554F5746BD6505C65C6E7EAE02EAFDE76B7.apk,1
# remove C687E2F0B4992BD368DF0C24B76943C99AC3EB9E4E8C13422EBF1A872A06070A,/mnt/f2ce377e-d586-41c5-bdf7-fad4a20f5b98/generate_rules/data/apks/C687E2F0B4992BD368DF0C24B76943C99AC3EB9E4E8C13422EBF1A872A06070A.apk,1
# remove 77F7A39A5B5367A5CE12E6B35B999FF1571821FB563764E45615FEB4CEB86FC0,/mnt/f2ce377e-d586-41c5-bdf7-fad4a20f5b98/generate_rules/data/apks/77F7A39A5B5367A5CE12E6B35B999FF1571821FB563764E45615FEB4CEB86FC0.apk,1
# remove 33C9604DB40D4453935A7076BA5018061BC906ACE486BC8C2D7197EC08EB8951,/mnt/f2ce377e-d586-41c5-bdf7-fad4a20f5b98/generate_rules/data/apks/33C9604DB40D4453935A7076BA5018061BC906ACE486BC8C2D7197EC08EB8951.apk,1






