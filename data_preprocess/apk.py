import polars as pl
import requests
from pathlib import Path
import dotenv
import enum
import os
from diskcache import FanoutCache

dotenv.load_dotenv()

cache = FanoutCache(f"{os.getenv("CACHE_FOLDER")}/apk_download_cache")


class APK_DOWNLOAD_STATUS(enum.Enum):
    SUCCESS: int = 0
    FAILED: int = 1
    NEED_DOWNLOAD: int = 2


APK_SCHEMA = {
    "sha256": pl.String,
    "is_malicious": pl.Int32,
}


def load_list(apk_list: str) -> pl.DataFrame:
    return pl.read_csv(
        apk_list,
        schema_overrides=APK_SCHEMA,
        has_header=True,
        columns=list(APK_SCHEMA.keys()),
    )


def save_list(apk_list: pl.DataFrame, output_path: str) -> Path:
    apk_list.write_csv(output_path, has_header=True)
    return Path(output_path).resolve()


def _calculate_sha256(file_path: Path) -> str:
    import hashlib

    sha256_hash = hashlib.sha256(usedforsecurity=False)
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().upper()


def get_folder() -> Path:
    return Path(os.getenv("APK_FOLDER", "NOT_DEFINED"))


def _get_path(sha256: str) -> Path:
    apk_path = get_folder() / f"{sha256}.apk"
    return apk_path.resolve()


def __download(
    sha256: str,
    apiKey: str,
    output_path: Path,
    force: bool = False,
) -> Path:
    url = f"https://androzoo.uni.lu/api/download?sha256={sha256}&apikey={apiKey}"

    if output_path.exists() and not force:
        return output_path.resolve()

    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path.resolve()
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        raise Exception(f"Failed to download apk {sha256}: {str(e)}")


def download(sha256: str, use_cache: bool = True, dry_run: bool = False) -> Path | None:
    if sha256 not in cache or not use_cache:
        apk_path = _get_path(sha256)
        if apk_path.exists():
            # Migrating: Check if apk exists in the apk folder
            # print(f"Find {sha256} exists. Add it into cache")
            cache.set(sha256, APK_DOWNLOAD_STATUS.SUCCESS)
        elif not dry_run:
            # Download APK
            try:
                # print(f"Downloading {sha256}")
                __download(
                    sha256,
                    apiKey=os.getenv("ANDROZOO_API_KEY", "NOT_DEFINED"),
                    output_path=_get_path(sha256),
                    force=use_cache,
                )
                cache.set(sha256, APK_DOWNLOAD_STATUS.SUCCESS)
            except Exception as e:
                print(f"Error downloading {sha256}: {e}")
                cache.set(sha256, APK_DOWNLOAD_STATUS.FAILED)
    else:
        # print(f"{sha256} is in cache")
        pass

    match cache[sha256]:
        case APK_DOWNLOAD_STATUS.SUCCESS:
            return _get_path(sha256)
        case APK_DOWNLOAD_STATUS.FAILED:
            return None
        case APK_DOWNLOAD_STATUS.NEED_DOWNLOAD:
            return None
