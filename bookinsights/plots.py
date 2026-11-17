"""Charts of the analysis: each function returns a matplotlib Figure."""

import matplotlib.pyplot as plt
import numpy as np

from bookinsights.analysis import summarize_by_group

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
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig
