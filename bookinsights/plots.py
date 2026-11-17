"""Charts of the analysis: each function returns a matplotlib Figure."""

import matplotlib.pyplot as plt

from bookinsights.analysis import summarize_by_genre

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


def plot_genre_ratings(books):
    """
    Draw the distribution of the average ratings of the books of each genre.

    One horizontal box per genre, sorted by median so that the best rated
    genre is at the top. The box spans the middle half of the books, the
    line inside it is the median and the dots are the outliers. The dashed
    line is the median of all the books.

    Parameters
    ----------
    books : pandas.DataFrame
        Output of ``clean_books``. Books without a genre are left out.

    Returns
    -------
    matplotlib.figure.Figure
        The chart.
    """
    summary = summarize_by_genre(books)
    # matplotlib draws the first box at the bottom, so the order is reversed
    order = summary.index[::-1]
    with_genre = books.dropna(subset=["genre"])
    values = [
        with_genre.loc[with_genre["genre"] == genre, "average_rating"]
        for genre in order
    ]
    labels = [f"{genre} ({summary.loc[genre, 'n_books']})" for genre in order]

    fig, ax = plt.subplots(figsize=(8, 7))
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
        with_genre["average_rating"].median(),
        color=TEXT_COLOR,
        linestyle="--",
        linewidth=1,
        label="median of all books",
    )

    _style_axes(ax)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Average rating of the book", color=TEXT_COLOR)
    ax.set_title("Average rating of the books by genre", loc="left")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return fig
