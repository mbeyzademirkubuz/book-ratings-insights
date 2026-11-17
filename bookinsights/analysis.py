"""Statistics on books, genres, authors and ratings."""

from scipy import stats


def summarize_by_genre(books):
    """
    Summarize the average ratings of the books of each genre.

    Each book counts once, whatever its number of ratings. The weighted mean,
    where each book counts as many times as its ratings, is added as a check:
    it shows the point of view of the single ratings, where popular books
    weigh more.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``, with the columns ``genre``,
        ``average_rating`` and ``work_ratings_count``. Books without a genre
        are left out.

    Returns
    -------
    pandas.DataFrame
        One row per genre, sorted from the highest to the lowest median, with
        the columns ``n_books``, ``mean``, ``median``, ``std`` and
        ``weighted_mean``.
    """
    with_genre = books.dropna(subset=["genre"])
    summary = with_genre.groupby("genre")["average_rating"].agg(
        n_books="count", mean="mean", median="median", std="std"
    )

    # Weighted mean = sum(rating * n. of ratings) / total n. of ratings
    weighted = (
        with_genre.assign(
            rating_sum=with_genre["average_rating"] * with_genre["work_ratings_count"]
        )
        .groupby("genre")[["rating_sum", "work_ratings_count"]]
        .sum()
    )
    summary["weighted_mean"] = weighted["rating_sum"] / weighted["work_ratings_count"]

    return summary.sort_values("median", ascending=False)


def test_genre_differences(books):
    """
    Test whether the average ratings of the books differ between genres.

    The Kruskal-Wallis test is used instead of the one-way ANOVA, since the
    ratings are skewed and their spread is not the same in all the genres,
    while the ANOVA assumes normal groups with equal variances. The test
    compares the ranks of the values instead of the values themselves.

    With thousands of books even a tiny difference is significant, so the
    p-value is complemented by an effect size, the epsilon squared:
    ``(H - k + 1) / (n - k)``, where ``k`` is the number of genres and ``n``
    the number of books. It goes from 0 (no difference) to 1, and is the
    share of the variability of the ranks explained by the genre.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``, with the columns ``genre`` and
        ``average_rating``. Books without a genre are left out.

    Returns
    -------
    dict
        ``statistic`` (the H statistic), ``p_value``, ``epsilon_squared``,
        ``n_books`` and ``n_genres``.
    """
    with_genre = books.dropna(subset=["genre"])
    groups = [
        ratings.to_numpy()
        for _, ratings in with_genre.groupby("genre")["average_rating"]
    ]
    statistic, p_value = stats.kruskal(*groups)

    n_books, n_genres = len(with_genre), len(groups)
    epsilon_squared = (statistic - n_genres + 1) / (n_books - n_genres)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "epsilon_squared": float(epsilon_squared),
        "n_books": n_books,
        "n_genres": n_genres,
    }
