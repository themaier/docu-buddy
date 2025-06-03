import io
import os
import shutil
import zipfile

import requests
from fastapi import HTTPException


def download_github_repo_zip(repo_url: str, dest_folder: str = "./repo"):
    zip_url = f"{repo_url}/archive/refs/heads/main.zip"
    response = requests.get(zip_url)

    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download repository ZIP")

    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(dest_folder)

    inner_folder = os.path.join(dest_folder, os.listdir(dest_folder)[0])

    # Move contents one level up
    for item in os.listdir(inner_folder):
        src = os.path.join(inner_folder, item)
        dst = os.path.join(dest_folder, item)

        if os.path.isdir(src):
            shutil.move(src, dst)
        else:
            shutil.move(src, dst)

    # Remove the now-empty inner folder
    shutil.rmtree(inner_folder)

    return dest_folder


if __name__ == "__main__":
    download_github_repo_zip("https://github.com/openrewrite/rewrite")
