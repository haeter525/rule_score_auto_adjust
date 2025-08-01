#!/usr/bin/bash
uv run mlflow server --backend-store-uri file:///mnt/storage/data/mlflow/mlruns --serve-artifacts --artifacts-destination file:///mnt/storage/data/mlflow/mlartifacts --port 55688