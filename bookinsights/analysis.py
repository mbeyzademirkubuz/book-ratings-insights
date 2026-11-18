"""Statistics on books, genres, authors and ratings."""

import pandas as pd
from scipy import stats


def summarize_by_group(books, column, sort_by_median=True):
    """
    Summarize the average ratings of the books of each group, e.g. each genre.

    Each book counts once, whatever its number of ratings. The weighted mean,
    where each book counts as many times as its ratings, is added as a check:
    it shows the point of view of the single ratings, where popular books
    weigh more.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``, with the columns ``average_rating``,
        ``work_ratings_count`` and ``column``. Books where ``column`` is
        missing are left out.
    column : str
        Column that defines the groups, e.g. ``"genre"`` or ``"era"``.
    sort_by_median : bool, optional
        If True (default), the groups are sorted from the highest to the
        lowest median; otherwise they keep their own order, e.g. the
        chronological order of the eras.

    Returns
    -------
    pandas.DataFrame
        One row per group, with the columns ``n_books``, ``mean``,
        ``median``, ``std`` and ``weighted_mean``.
    """
    with_group = books.dropna(subset=[column])
    # observed=True: with categorical columns (like the eras), only the
    # categories that actually appear in the data become groups
    grouped = with_group.groupby(column, observed=True)
    summary = grouped["average_rating"].agg(
        n_books="count", mean="mean", median="median", std="std"
    )

    # Weighted mean = sum(rating * n. of ratings) / total n. of ratings
    rating_sum = with_group["average_rating"] * with_group["work_ratings_count"]
    weighted = (
        with_group.assign(rating_sum=rating_sum)
        .groupby(column, observed=True)[["rating_sum", "work_ratings_count"]]
        .sum()
    )
    summary["weighted_mean"] = weighted["rating_sum"] / weighted["work_ratings_count"]

    if sort_by_median:
        summary = summary.sort_values("median", ascending=False)
    return summary


def test_group_differences(books, column):
    """
    Test whether the average ratings of the books differ between groups.

    The Kruskal-Wallis test is used instead of the one-way ANOVA, since the
    ratings are skewed and their spread is not the same in all the groups,
    while the ANOVA assumes normal groups with equal variances. The test
    compares the ranks of the values instead of the values themselves.

    With thousands of books even a tiny difference is significant, so the
    p-value is complemented by an effect size, the epsilon squared:
    ``(H - k + 1) / (n - k)``, where ``k`` is the number of groups and ``n``
    the number of books. It goes from 0 (no difference) to 1, and is the
    share of the variability of the ranks explained by the groups.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``, with the columns ``average_rating`` and
        ``column``. Books where ``column`` is missing are left out.
    column : str
        Column that defines the groups, e.g. ``"genre"`` or ``"era"``.

    Returns
    -------
    dict
        ``statistic`` (the H statistic), ``p_value``, ``epsilon_squared``,
        ``n_books`` and ``n_groups``.
    """
    with_group = books.dropna(subset=[column])
    groups = [
        ratings.to_numpy()
        for _, ratings in with_group.groupby(column, observed=True)["average_rating"]
    ]
    statistic, p_value = stats.kruskal(*groups)

    n_books, n_groups = len(with_group), len(groups)
    epsilon_squared = (statistic - n_groups + 1) / (n_books - n_groups)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "epsilon_squared": float(epsilon_squared),
        "n_books": n_books,
        "n_groups": n_groups,
    }


def correlate_with_rating(books, column):
    """
    Measure how strongly a column is related to the average rating.

    The Spearman correlation is used instead of the Pearson one: it compares
    the ranks of the values, so it captures any monotonic relation (not only
    a linear one) and it is not dominated by a few extreme values.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``, with the columns ``average_rating`` and
        ``column``. Books where ``column`` is missing are left out.
    column : str
        Numeric column, e.g. ``"publication_year"``.

    Returns
    -------
    dict
        ``rho`` (from -1 to 1, 0 means no monotonic relation), ``p_value``
        and ``n_books``.
    """
    valid = books.dropna(subset=[column])
    rho, p_value = stats.spearmanr(valid[column].astype(float), valid["average_rating"])
    return {"rho": float(rho), "p_value": float(p_value), "n_books": len(valid)}


def genre_shares(books, column):
    """
    Compute the share of each genre within each group, e.g. each era.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``, with the columns ``genre`` and
        ``column``. Books where either is missing are left out.
    column : str
        Column that defines the groups, e.g. ``"era"``.

    Returns
    -------
    pandas.DataFrame
        One row per group and one column per genre, from the most to the
        least common genre overall. Each row sums to 1.
    """
    valid = books.dropna(subset=["genre", column])
    shares = pd.crosstab(valid[column], valid["genre"], normalize="index")
    order = valid["genre"].value_counts().index
    return shares[order]


def summarize_by_popularity(books, n_groups=10):
    """
    Summarize the average ratings of the books by popularity group.

    The books are split into groups of the same size by their number of
    ratings (``work_ratings_count``), e.g. ten groups (deciles): group 1
    has the 10% least rated books, group 10 the 10% most rated ones.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``, with the columns ``average_rating`` and
        ``work_ratings_count``.
    n_groups : int, optional
        Number of groups. Defaults to 10.

    Returns
    -------
    pandas.DataFrame
        One row per group, from the least to the most popular, with the
        columns ``n_books``, ``min_ratings_count``, ``median_ratings_count``,
        ``median`` (rating), ``min`` (lowest rating) and ``std``.
    """
    group = pd.qcut(
        books["work_ratings_count"], n_groups, labels=range(1, n_groups + 1)
    ).rename("popularity_group")
    grouped = books.groupby(group, observed=True)
    summary = grouped["work_ratings_count"].agg(
        n_books="count", min_ratings_count="min", median_ratings_count="median"
    )
    ratings = grouped["average_rating"].agg(median="median", min="min", std="std")
    return summary.join(ratings)


def compare_series_to_first(books, min_books=3):
    """
    Compare each book of a series with the first book of the same series.

    A series is identified by its name and its main author, since different
    authors may use the same series name. When the same position appears
    more than once (e.g. a graphic novel edition of the first book), only
    the most rated book is kept. Only the series whose first book is in the
    data and with at least ``min_books`` main books are kept.

    Each book gets a ``kind``:

    - ``"first"``: the first book of the series;
    - ``"main"``: the following books with an integer number (#2, #3, ...);
    - ``"in-between"``: novellas and prequels, with a fractional number or
      0 (e.g. #1.5 or #0). They are kept apart, since they are a different
      kind of book, read by far fewer readers.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``, with the columns ``title``,
        ``main_author``, ``series``, ``series_number``, ``average_rating``
        and ``work_ratings_count``.
    min_books : int, optional
        Minimum number of main books of a series, the first one included.
        Defaults to 3.

    Returns
    -------
    pandas.DataFrame
        One row per book of the selected series, with the columns ``title``,
        ``main_author``, ``series``, ``series_number``, ``kind``,
        ``average_rating``, ``work_ratings_count``, ``rating_diff`` (the
        rating minus the rating of the first book) and ``ratings_ratio``
        (the number of ratings divided by that of the first book).
    """
    key = ["main_author", "series"]
    in_series = (
        books.dropna(subset=["series"])
        .sort_values("work_ratings_count", ascending=False)
        .drop_duplicates(key + ["series_number"])
    )

    number = in_series["series_number"]
    kind = pd.Series("in-between", index=in_series.index)
    kind[(number >= 1) & (number % 1 == 0)] = "main"
    kind[number == 1] = "first"
    in_series = in_series.assign(kind=kind)

    # Keep the series with the first book and enough main books
    n_main = in_series[in_series["kind"] != "in-between"].groupby(key).size()
    has_first = in_series[in_series["kind"] == "first"].set_index(key).index
    selected = n_main[(n_main >= min_books) & n_main.index.isin(has_first)].index
    in_series = in_series[in_series.set_index(key).index.isin(selected)]

    first = in_series[in_series["kind"] == "first"].set_index(key)
    first = first[["average_rating", "work_ratings_count"]].add_suffix("_first")
    comparison = in_series.join(first, on=key)
    comparison["rating_diff"] = (
        comparison["average_rating"] - comparison["average_rating_first"]
    )
    comparison["ratings_ratio"] = (
        comparison["work_ratings_count"] / comparison["work_ratings_count_first"]
    )

    columns = [
        "title",
        "main_author",
        "series",
        "series_number",
        "kind",
        "average_rating",
        "work_ratings_count",
        "rating_diff",
        "ratings_ratio",
    ]
    return comparison[columns].sort_values(key + ["series_number"])


def summarize_series_progression(comparison, max_position=10):
    """
    Summarize how the main books of the series compare with the first one.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Output of ``compare_series_to_first``.
    max_position : int, optional
        Last position in the series to summarize: later positions have few
        series. Defaults to 10.

    Returns
    -------
    pandas.DataFrame
        One row per position, from 1 to ``max_position``, with the columns
        ``n_series``, ``median_rating_diff``, ``share_higher`` (share of
        books rated higher than the first one) and ``median_ratings_ratio``.
    """
    main = comparison[comparison["kind"].isin(["first", "main"])]
    main = main[main["series_number"] <= max_position]
    summary = main.groupby(main["series_number"].astype(int).rename("position")).agg(
        n_series=("rating_diff", "size"),
        median_rating_diff=("rating_diff", "median"),
        share_higher=("rating_diff", lambda diff: (diff > 0).mean()),
        median_ratings_ratio=("ratings_ratio", "median"),
    )
    return summary


def summarize_series_kinds(comparison):
    """
    Compare the main books and the in-between books with the first books.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Output of ``compare_series_to_first``.

    Returns
    -------
    pandas.DataFrame
        One row per kind (``"main"`` and ``"in-between"``), with the columns
        ``n_books``, ``n_series``, ``median_rating_diff``, ``share_higher``
        and ``median_ratings_ratio``.
    """
    later = comparison[comparison["kind"] != "first"]
    series_id = later["main_author"] + " / " + later["series"]
    return (
        later.assign(series_id=series_id)
        .groupby("kind")
        .agg(
            n_books=("rating_diff", "size"),
            n_series=("series_id", "nunique"),
            median_rating_diff=("rating_diff", "median"),
            share_higher=("rating_diff", lambda diff: (diff > 0).mean()),
            median_ratings_ratio=("ratings_ratio", "median"),
        )
        .sort_values("n_books", ascending=False)
    )


def last_main_books(comparison):
    """
    Select the last main book of each series.

    Parameters
    ----------
    comparison : pandas.DataFrame
        Output of ``compare_series_to_first``.

    Returns
    -------
    pandas.DataFrame
        One row per series: the main book with the highest number.
    """
    main = comparison[comparison["kind"] == "main"]
    return main.sort_values("series_number").groupby(["main_author", "series"]).tail(1)


def test_difference_from_first(differences):
    """
    Test whether books are rated differently than the first book of their series.

    The Wilcoxon signed-rank test is used: the data are paired (each book is
    compared with the first book of its own series) and the differences are
    not assumed to be normal. Books rated exactly like the first one carry
    no information on the direction, so they are left out by the test.

    Parameters
    ----------
    differences : pandas.Series
        Rating of each book minus the rating of the first book of its
        series, e.g. the ``rating_diff`` column of ``last_main_books``.

    Returns
    -------
    dict
        ``n_books``, ``median_diff``, ``share_higher``, ``statistic`` and
        ``p_value``.
    """
    statistic, p_value = stats.wilcoxon(differences)
    return {
        "n_books": len(differences),
        "median_diff": float(differences.median()),
        "share_higher": float((differences > 0).mean()),
        "statistic": float(statistic),
        "p_value": float(p_value),
    }
