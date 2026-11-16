"""Download and load the goodbooks-10k data files."""

import os

import pandas as pd
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

# Columns of books.csv used in the analysis: identifiers, metadata,
# popularity and the distribution of the ratings (from 1 to 5 stars)
BOOK_COLUMNS = [
    "book_id",
    "goodreads_book_id",
    "authors",
    "original_publication_year",
    "title",
    "language_code",
    "average_rating",
    "ratings_count",
    "work_ratings_count",
    "ratings_1",
    "ratings_2",
    "ratings_3",
    "ratings_4",
    "ratings_5",
    "image_url",
]

# The smallest integer types that fit the values: user ids go up to 53,424
# (too many for int16, max 32,767), book ids up to 10,000 and ratings from
# 1 to 5. This reduces the memory of the ratings from ~143 MB to ~42 MB
RATING_DTYPES = {"user_id": "int32", "book_id": "int16", "rating": "int8"}
BOOK_TAG_DTYPES = {"goodreads_book_id": "int32", "tag_id": "int32", "count": "int32"}


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


def read_data_file(name, data_dir=DATA_DIR, **kwargs):
    """
    Read a CSV file of the dataset into a DataFrame.

    Parameters
    ----------
    name : str
        Name of the file, e.g. ``"books.csv"``.
    data_dir : str, optional
        Folder that contains the file. Defaults to ``DATA_DIR``.
    **kwargs
        Further arguments passed to ``pandas.read_csv``, e.g. ``usecols``
        or ``dtype``.

    Returns
    -------
    pandas.DataFrame
        The content of the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist, i.e. the data have not been downloaded.
    ValueError
        If the file is empty or is not a valid CSV file.
    """
    path = os.path.join(data_dir, name)
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"{path} not found: run download_data() to download the dataset"
        ) from err
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as err:
        raise ValueError(f"{path} is not a valid CSV file: {err}") from err


def load_books(data_dir=DATA_DIR):
    """
    Load the metadata of the books, keeping only the columns in ``BOOK_COLUMNS``.

    Parameters
    ----------
    data_dir : str, optional
        Folder that contains the data. Defaults to ``DATA_DIR``.

    Returns
    -------
    pandas.DataFrame
        One row per book.
    """
    return read_data_file("books.csv", data_dir, usecols=BOOK_COLUMNS)


def load_ratings(data_dir=DATA_DIR):
    """
    Load the ratings given by the users to the books.

    Parameters
    ----------
    data_dir : str, optional
        Folder that contains the data. Defaults to ``DATA_DIR``.

    Returns
    -------
    pandas.DataFrame
        One row per rating, with the columns ``user_id``, ``book_id`` and
        ``rating``, stored with the compact types of ``RATING_DTYPES``.
    """
    return read_data_file("ratings.csv", data_dir, dtype=RATING_DTYPES)


def load_tags(data_dir=DATA_DIR):
    """
    Load the names of the tags.

    Parameters
    ----------
    data_dir : str, optional
        Folder that contains the data. Defaults to ``DATA_DIR``.

    Returns
    -------
    pandas.DataFrame
        One row per tag, with the columns ``tag_id`` and ``tag_name``.
    """
    return read_data_file("tags.csv", data_dir)


def load_book_tags(data_dir=DATA_DIR):
    """
    Load the tags assigned by the users to each book.

    Parameters
    ----------
    data_dir : str, optional
        Folder that contains the data. Defaults to ``DATA_DIR``.

    Returns
    -------
    pandas.DataFrame
        One row per (book, tag) pair, with the columns ``goodreads_book_id``,
        ``tag_id`` and ``count``, i.e. how many users assigned the tag.
    """
    return read_data_file("book_tags.csv", data_dir, dtype=BOOK_TAG_DTYPES)
