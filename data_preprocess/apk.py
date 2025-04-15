import dask
import requests
from pathlib import Path
import dotenv
import os

from tqdm import tqdm
dotenv.load_dotenv()

# calculate sha256
def calculate_sha256(file_path: Path) -> str:
    import hashlib
    sha256_hash = hashlib.sha256(usedforsecurity=False)
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().upper()

def get_folder() -> Path:
    return Path(os.getenv("APK_FOLDER"))

def get(sha256: str) -> Path:
    apk_path = get_folder() / f"{sha256}.apk"
    return apk_path.resolve()

def __download(sha256: str, apiKey: str, output_path: Path, force: bool = False) -> Path:
    if not force:
        if output_path.exists() and calculate_sha256(output_path) == sha256.upper():
            return output_path

    url = "https://androzoo.uni.lu/api/download"
    params = {"apikey": apiKey, "sha256": sha256}
    
    try:
        response = requests.get(url, params=params, stream=True)
        response.raise_for_status()
        
        # 获取文件总大小
        total_size = int(response.headers.get("content-length", 0))
        
        # 使用 tqdm 显示下载进度
        with open(output_path, "wb") as f:
            for chunk in tqdm(
                    response.iter_content(chunk_size=1024),
                    desc=sha256,
                    total=total_size / 1024,
                    unit="KB",
                    unit_scale=True,
                    unit_divisor=1024
                    ):
                if chunk:
                    f.write(chunk)
                    
        if calculate_sha256(output_path) != sha256.upper():
            raise ValueError(f"Downloaded file's SHA256 does not match: {output_path}")
        return output_path
        
    except requests.exceptions.RequestException as e:
        if output_path.exists():
            output_path.unlink()
        raise Exception(f"Failed to download apk {sha256}: {str(e)}")
    
@dask.delayed
def download(sha256:str, force: bool=False) -> Path | None:
    api_key = os.getenv("ANDROZOO_API_KEY")
    apk_path = get(sha256=sha256)
    
    try:
        result = __download(sha256, apiKey=api_key, output_path=apk_path, force=force)
        print(f"Downloaded {sha256} to {result}")
        return result
    except Exception as e:
        print(f"Error downloading {sha256}: {e}")
        return None