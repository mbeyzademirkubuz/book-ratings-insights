"""Clean the raw goodbooks-10k tables and derive the columns of the analysis."""

import pandas as pd

# Tags that describe a genre, mapped to the genre they belong to. Synonyms
# (e.g. "sci-fi" and "science-fiction") are merged into a single genre. Very
# generic tags, like "fiction", "non-fiction" or "contemporary", are left out:
# almost every book has them, so they would not distinguish between books.
# "young-adult" and "children" describe an audience rather than a genre, but
# they are kept, since Goodreads and its readers use them as genres
GENRE_TAGS = {
    "fantasy": "fantasy",
    "science-fiction": "science-fiction",
    "sci-fi": "science-fiction",
    "scifi": "science-fiction",
    "romance": "romance",
    "mystery": "mystery",
    "mysteries": "mystery",
    "crime": "mystery",
    "thriller": "thriller",
    "thrillers": "thriller",
    "horror": "horror",
    "historical-fiction": "historical-fiction",
    "historical": "historical-fiction",
    "young-adult": "young-adult",
    "ya": "young-adult",
    "children": "children",
    "childrens": "children",
    "children-s": "children",
    "kids": "children",
    "classics": "classics",
    "classic": "classics",
    "literary-fiction": "literary-fiction",
    "poetry": "poetry",
    "comics": "comics",
    "graphic-novels": "comics",
    "graphic-novel": "comics",
    "manga": "comics",
    "biography": "biography",
    "autobiography": "biography",
    "memoir": "biography",
    "memoirs": "biography",
    "history": "history",
    "self-help": "self-help",
    "philosophy": "philosophy",
    "humor": "humor",
    "humour": "humor",
}

# Edges and labels of the publication eras: the first era collects all the
# books published before 1800, ancient ones (negative years) included
ERA_BINS = [float("-inf"), 1800, 1900, 1950, 2000, 2010, float("inf")]
ERA_LABELS = [
    "before 1800",
    "1800-1899",
    "1900-1949",
    "1950-1999",
    "2000-2009",
    "2010 or later",
]

# Series information at the end of a title, e.g. "(Harry Potter, #1)" or
# "(Discworld, #1; Rincewind #1)": only the first series is kept. Ranges of
# numbers, like "#1-7" in box sets, do not match, since they are not a
# single book of the series
SERIES_PATTERN = (
    r"\((?P<series>[^()#]+?),?\s*#(?P<series_number>\d+(?:\.\d+)?)(?:;[^()]*)?\)\s*$"
)


def clean_book_tags(book_tags):
    """
    Remove the invalid rows of the book tags.

    Rows with a negative count are dropped, since a tag cannot be assigned
    by a negative number of users. When a (book, tag) pair appears more than
    once, only the row with the highest count is kept.

    Parameters
    ----------
    book_tags : pandas.DataFrame
        Columns ``goodreads_book_id``, ``tag_id`` and ``count``.

    Returns
    -------
    pandas.DataFrame
        The valid rows, one per (book, tag) pair.
    """
    valid = book_tags[book_tags["count"] >= 0]
    return valid.sort_values("count", ascending=False).drop_duplicates(
        ["goodreads_book_id", "tag_id"]
    )


def assign_primary_genre(book_tags, tags, genre_tags=GENRE_TAGS):
    """
    Find the primary genre of each book, i.e. its most used genre tag.

    The counts of synonym tags are summed, so that e.g. "sci-fi" and
    "science-fiction" both contribute to the genre "science-fiction".

    Parameters
    ----------
    book_tags : pandas.DataFrame
        Clean book tags, with the columns ``goodreads_book_id``, ``tag_id``
        and ``count``.
    tags : pandas.DataFrame
        Columns ``tag_id`` and ``tag_name``.
    genre_tags : dict, optional
        Mapping ``tag name -> genre``. Defaults to ``GENRE_TAGS``.

    Returns
    -------
    pandas.Series
        The primary genre, indexed by ``goodreads_book_id``. Books without
        any genre tag are not included.
    """
    named = book_tags.merge(tags, on="tag_id")
    named["genre"] = named["tag_name"].map(genre_tags)
    genre_counts = (
        named.dropna(subset=["genre"])
        .groupby(["goodreads_book_id", "genre"], as_index=False)["count"]
        .sum()
    )
    # After sorting by count, the first row of each book is its primary genre
    primary = genre_counts.sort_values("count", ascending=False).drop_duplicates(
        "goodreads_book_id"
    )
    return primary.set_index("goodreads_book_id")["genre"]


def extract_main_author(authors):
    """
    Extract the main author from the list of authors of each book.

    The first name is the author; the following ones are usually
    illustrators, translators or narrators.

    Parameters
    ----------
    authors : pandas.Series
        Comma-separated names, e.g. ``"J.K. Rowling, Mary GrandPré"``.

    Returns
    -------
    pandas.Series
        The first name of each list.
    """
    return authors.str.split(", ").str[0]


def extract_series(titles):
    """
    Extract the series and the position in the series from the titles.

    Parameters
    ----------
    titles : pandas.Series
        Book titles, e.g. ``"The Hunger Games (The Hunger Games, #1)"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``series`` (str) and ``series_number`` (float, since some
        books are in between, e.g. ``#1.5``). Both are missing for the books
        that are not part of a series.
    """
    series = titles.str.extract(SERIES_PATTERN)
    series["series"] = series["series"].str.strip()
    series["series_number"] = series["series_number"].astype(float)
    return series


def add_publication_era(years):
    """
    Group the publication years into eras.

    Parameters
    ----------
    years : pandas.Series
        Publication years; negative values are years before Christ.

    Returns
    -------
    pandas.Series
        Ordered categories, with the labels of ``ERA_LABELS``. Missing years
        stay missing.
    """
    return pd.cut(years, bins=ERA_BINS, labels=ERA_LABELS, right=False)


def clean_books(books, book_tags, tags):
    """
    Build the table of the books used in the analysis.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``load_books``.
    book_tags : pandas.DataFrame
        Output of ``load_book_tags``.
    tags : pandas.DataFrame
        Output of ``load_tags``.

    Returns
    -------
    pandas.DataFrame
        One row per book, with the columns of ``books`` and the derived
        columns ``main_author``, ``publication_year`` (nullable integer),
        ``era``, ``series``, ``series_number`` and ``genre``.
    """
    genres = assign_primary_genre(clean_book_tags(book_tags), tags)

    clean = books.rename(columns={"original_publication_year": "publication_year"})
    clean["publication_year"] = clean["publication_year"].astype("Int64")
    clean["era"] = add_publication_era(clean["publication_year"])
    clean["main_author"] = extract_main_author(clean["authors"])
    clean[["series", "series_number"]] = extract_series(clean["title"])
    clean["genre"] = clean["goodreads_book_id"].map(genres)
    return clean
