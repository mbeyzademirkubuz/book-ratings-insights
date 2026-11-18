"""Charts of the analysis: each function returns a matplotlib Figure."""

import matplotlib.pyplot as plt
import numpy as np

from bookinsights.analysis import summarize_by_group, summarize_by_popularity

# Colors shared by all the charts
MAIN_COLOR = "#2a78d6"
LIGHT_COLOR = "#c9dcf3"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#d9d8d4"


def _style_axes(ax):
    """Make the frame of the chart recessive, so that the data stand out."""
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR)


def plot_rating_distribution(books):
    """
    Draw the histogram of the average ratings of the books.

    The average ratings have two decimals, so each bar is 0.05 wide and
    centered on a multiple of 0.05: it always collects five possible values.
    The dashed line is the median.

    Parameters
    ----------
    books : pandas.DataFrame
        Must have the column ``average_rating``.

    Returns
    -------
    matplotlib.figure.Figure
        The chart.
    """
    ratings = books["average_rating"]
    # Edges halfway between two multiples of 0.05, e.g. 3.975 and 4.025
    lowest = np.floor(ratings.min() * 20) / 20
    highest = np.ceil(ratings.max() * 20) / 20
    bins = np.arange(lowest - 0.025, highest + 0.05, 0.05)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ratings, bins=bins, color=MAIN_COLOR, edgecolor="white", linewidth=1)
    ax.axvline(
        ratings.median(),
        color=TEXT_COLOR,
        linestyle="--",
        linewidth=1,
        label=f"median ({ratings.median():.2f})",
    )

    _style_axes(ax)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average rating of the book", color=TEXT_COLOR)
    ax.set_ylabel("Number of books", color=TEXT_COLOR)
    ax.set_title("Distribution of the average ratings", loc="left")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def plot_ratings_by_group(books, column, sort_by_median=True, title=None):
    """
    Draw the distribution of the average ratings of the books of each group.

    One horizontal box per group, in the order of ``summarize_by_group``
    from top to bottom. The box spans the middle half of the books, the line
    inside it is the median and the dots are the outliers. The dashed line
    is the median of all the books.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``. Books where ``column`` is missing are
        left out.
    column : str
        Column that defines the groups, e.g. ``"genre"`` or ``"era"``.
    sort_by_median : bool, optional
        If True (default), the best rated group is at the top; otherwise
        the groups keep their own order, e.g. the chronological one.
    title : str, optional
        Title of the chart. Defaults to "Average rating of the books by
        <column>".

    Returns
    -------
    matplotlib.figure.Figure
        The chart.
    """
    summary = summarize_by_group(books, column, sort_by_median)
    # matplotlib draws the first box at the bottom, so the order is reversed
    order = summary.index[::-1]
    with_group = books.dropna(subset=[column])
    grouped = with_group.groupby(column, observed=True)["average_rating"]
    values = [grouped.get_group(group) for group in order]
    labels = [f"{group} ({summary.loc[group, 'n_books']})" for group in order]

    fig, ax = plt.subplots(figsize=(8, 1.5 + 0.3 * len(order)))
    ax.boxplot(
        values,
        orientation="horizontal",
        tick_labels=labels,
        widths=0.6,
        patch_artist=True,
        boxprops={"facecolor": LIGHT_COLOR, "edgecolor": MAIN_COLOR},
        medianprops={"color": MAIN_COLOR, "linewidth": 2},
        whiskerprops={"color": MAIN_COLOR},
        capprops={"color": MAIN_COLOR},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": TEXT_COLOR,
            "markeredgecolor": "none",
            "alpha": 0.4,
        },
    )
    ax.axvline(
        with_group["average_rating"].median(),
        color=TEXT_COLOR,
        linestyle="--",
        linewidth=1,
        label="median of all books",
    )

    _style_axes(ax)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average rating of the book", color=TEXT_COLOR)
    if title is None:
        title = f"Average rating of the books by {column}"
    ax.set_title(title, loc="left")
    # Above the plot, on the right of the title, so that it never hides data
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), frameon=False, borderaxespad=0)
    fig.tight_layout()
    return fig


def plot_popularity_vs_rating(books, n_groups=10, n_labels=3):
    """
    Draw the average rating of the books against their number of ratings.

    Each dot is a book. The number of ratings goes from a few thousand to
    millions, so the horizontal axis is logarithmic. The line joins the
    median rating of each popularity group (see
    ``summarize_by_popularity``), and the most rated books are labeled.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``, with the columns ``title``,
        ``average_rating`` and ``work_ratings_count``.
    n_groups : int, optional
        Number of popularity groups of the line. Defaults to 10.
    n_labels : int, optional
        Number of most rated books to label. Defaults to 3.

    Returns
    -------
    matplotlib.figure.Figure
        The chart.
    """
    summary = summarize_by_popularity(books, n_groups)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        books["work_ratings_count"],
        books["average_rating"],
        s=6,
        color=MAIN_COLOR,
        alpha=0.2,
        linewidths=0,
    )
    ax.plot(
        summary["median_ratings_count"],
        summary["median"],
        color=TEXT_COLOR,
        linewidth=2,
        marker="o",
        markersize=5,
        label=f"median rating of each of {n_groups} popularity groups",
    )

    # The labeled books are drawn as solid dots, so that they stand out
    top = books.nlargest(n_labels, "work_ratings_count")
    ax.scatter(top["work_ratings_count"], top["average_rating"], s=20, color=MAIN_COLOR)
    # Short titles, without the series, e.g. "Twilight" instead of
    # "Twilight (Twilight, #1)"
    for _, book in top.iterrows():
        title = book["title"].split(" (")[0]
        ax.annotate(
            title,
            (book["work_ratings_count"], book["average_rating"]),
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color=TEXT_COLOR,
        )

    ax.set_xscale("log")
    _style_axes(ax)
    ax.grid(color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Number of ratings (log scale)", color=TEXT_COLOR)
    ax.set_ylabel("Average rating", color=TEXT_COLOR)
    ax.set_title("Popularity and rating of the books", loc="left")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_series_progression(progression):
    """
    Draw how the books of the series compare with the first one.

    Two panels with the same horizontal axis, the position in the series:
    on the left, the median difference in rating from the first book; on
    the right, the median number of ratings as a share of those of the first
    book. Two panels are used instead of one chart with two vertical axes,
    which would make the two measures look comparable.

    Parameters
    ----------
    progression : pandas.DataFrame
        Output of ``summarize_series_progression``.

    Returns
    -------
    matplotlib.figure.Figure
        The chart.
    """
    positions = progression.index

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    left.axhline(0, color=TEXT_COLOR, linewidth=1)
    left.plot(
        positions,
        progression["median_rating_diff"],
        color=MAIN_COLOR,
        linewidth=2,
        marker="o",
        markersize=5,
    )
    left.set_ylabel("Rating minus rating of book #1", color=TEXT_COLOR)
    left.set_title("Later books are rated higher...", loc="left")

    right.plot(
        positions,
        progression["median_ratings_ratio"] * 100,
        color=MAIN_COLOR,
        linewidth=2,
        marker="o",
        markersize=5,
    )
    right.set_ylim(0, 105)
    right.set_ylabel("Ratings, % of those of book #1", color=TEXT_COLOR)
    right.set_title("...but by fewer and fewer readers", loc="left")

    for ax in (left, right):
        _style_axes(ax)
        ax.grid(color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(positions)
        ax.set_xlabel("Position in the series", color=TEXT_COLOR)

    fig.suptitle(
        f"Median over the series (from {progression['n_series'].iloc[0]} series "
        f"at book #1 to {progression['n_series'].iloc[-1]} at book "
        f"#{positions[-1]})",
        x=0.01,
        ha="left",
        color=TEXT_COLOR,
        fontsize=10,
    )
    fig.tight_layout()
    return fig
