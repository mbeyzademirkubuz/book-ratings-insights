"""Download, load and clean the goodbooks-10k data files."""

import os

import requests

# Pinning a commit of the goodbooks-10k repository, instead of the "master"
# branch, guarantees that the same data are downloaded every time
DATASET_COMMIT = "6dd165b555a7b47b2dd36743a425776e641ff50c"
BASE_URL = f"https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/{DATASET_COMMIT}"

# The data folder is found from the position of this file, so that it works
# whether the code is run from the root, from a notebook or from the app
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DATA_FILES = ["books.csv", "ratings.csv", "tags.csv", "book_tags.csv"]

CHUNK_SIZE = 1024 * 1024  # bytes written to disk at a time (1 MB)
TIMEOUT = 30  # seconds to wait for the server before giving up


def download_file(url, path, timeout=TIMEOUT):
    """
    Download a file from a URL, unless it already exists locally.

    The file is first written to a temporary ``.part`` file, which is renamed
    only when the download is complete: an interrupted download never leaves
    a truncated file that would look like a valid one.

    Parameters
    ----------
    url : str
        Address of the file to download.
    path : str
        Local path where the file is saved.
    timeout : float, optional
        Seconds to wait for the server to respond. Defaults to ``TIMEOUT``.

    Returns
    -------
    str
        The local path of the file.

    Raises
    ------
    RuntimeError
        If the server returns an error, or the network fails.
    """
    if os.path.exists(path):
        return path

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    temp_path = path + ".part"
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            # Turn HTTP errors (e.g. 404 Not Found) into exceptions
            response.raise_for_status()
            # Write the file one chunk at a time, without loading it
            # entirely in memory
            with open(temp_path, "wb") as file:
                file.writelines(response.iter_content(chunk_size=CHUNK_SIZE))
    except (requests.exceptions.RequestException, OSError) as err:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Could not download {url}: {err}") from err

    os.replace(temp_path, path)
    return path


def download_data(data_dir=DATA_DIR, files=DATA_FILES):
    """
    Download the files of the goodbooks-10k dataset that are not yet local.

    Parameters
    ----------
    data_dir : str, optional
        Folder where the files are saved. Defaults to ``DATA_DIR``.
    files : list of str, optional
        Names of the files to download. Defaults to ``DATA_FILES``.

    Returns
    -------
    dict
        Mapping ``file name -> local path``.
    """
    paths = {}
    for name in files:
        url = f"{BASE_URL}/{name}"
        paths[name] = download_file(url, os.path.join(data_dir, name))
    return paths
