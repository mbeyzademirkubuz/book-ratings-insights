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
