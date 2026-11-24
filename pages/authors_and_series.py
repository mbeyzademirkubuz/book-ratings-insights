"""Authors and series page: sequels and careers (questions 4 and 5)."""

import streamlit as st

from app_data import format_p_value, get_books, get_series, show_figure
from bookinsights.analysis import (
    author_careers,
    author_spread,
    last_main_books,
    summarize_careers,
    summarize_series_progression,
    test_author_trends,
    test_difference_from_first,
    test_group_differences,
)
from bookinsights.plots import (
    plot_author_careers,
    plot_book_ratings,
    plot_series_progression,
)


@st.cache_data(show_spinner="Following the careers of the authors...")
def get_careers(min_books):
    """Return the careers of the authors with at least ``min_books`` books, and their tests."""
    books = get_books()
    careers = author_careers(books, min_books=min_books)
    standalone = author_careers(books, min_books=min_books, standalone_only=True)
    return {
        "careers": careers,
        "summary_all": summarize_careers(careers),
        "summary_standalone": summarize_careers(standalone),
        "consistency": test_group_differences(careers, "main_author"),
        "spread": author_spread(careers),
        "trend_all": test_author_trends(careers),
        "trend_standalone": test_author_trends(standalone),
    }


series = get_series()

st.title("Authors and series")

# ---------------------------------------------------------------------- series
st.header("Series")
st.markdown(
    "Each book of a series is compared with the **first book of the same series**: "
    "the difference in rating, and its number of ratings as a share of those of the "
    "first book. Only the series with their first book and at least three main "
    "books in the data are kept."
)
max_position = st.slider(
    "Show the books up to position", min_value=3, max_value=15, value=10
)
show_figure(plot_series_progression(summarize_series_progression(series, max_position)))

test = test_difference_from_first(last_main_books(series)["rating_diff"])
columns = st.columns(3)
columns[0].metric("Series", f"{test['n_books']:,}")
columns[1].metric(
    "Last book vs first",
    f"{test['median_diff']:+.2f} stars",
    help="Median difference between the last main book and the first one.",
)
columns[2].metric("Rated higher than the first", f"{test['share_higher']:.0%}")
st.markdown(
    f"The difference is significant (Wilcoxon signed-rank test, p "
    f"{format_p_value(test['p_value'])}). The later books are rated higher, but by "
    "fewer and fewer readers: probably only the readers who liked the first book "
    "keep reading, and they rate the sequels generously."
)

st.subheader("Explore a series")
# A series is identified by its name and its author, since different authors use
# the same series name
labels = series["series"] + " (" + series["main_author"] + ")"
options = sorted(labels.unique())
default = "Harry Potter (J.K. Rowling)"
choice = st.selectbox(
    "Series", options, index=options.index(default) if default in options else 0
)
books_in_series = series[labels == choice].sort_values("series_number")

show_figure(
    plot_book_ratings(
        books_in_series, "series_number", "Position in the series", choice
    )
)
st.dataframe(
    books_in_series[
        [
            "series_number",
            "title",
            "kind",
            "average_rating",
            "rating_diff",
            "ratings_ratio",
        ]
    ],
    hide_index=True,
    column_config={
        "series_number": st.column_config.NumberColumn("#", format="%g"),
        "title": "Title",
        "kind": "Kind",
        "average_rating": st.column_config.NumberColumn("Rating", format="%.2f"),
        "rating_diff": st.column_config.NumberColumn("vs first book", format="%+.2f"),
        "ratings_ratio": st.column_config.NumberColumn(
            "Ratings vs first book",
            format="percent",
            help="Number of ratings as a share of those of the first book.",
        ),
    },
)

# --------------------------------------------------------------------- authors
st.header("Authors")
st.markdown(
    "The books of each author are ordered by publication year. The data contain "
    "only popular books, so the first book of an author here is not necessarily "
    "their debut."
)
min_books = st.slider(
    "Minimum number of books per author", min_value=3, max_value=10, value=5
)
results = get_careers(min_books)

consistency = results["consistency"]
spread = results["spread"]
columns = st.columns(3)
columns[0].metric("Authors", f"{consistency['n_groups']:,}")
columns[1].metric(
    "Explained by the author",
    f"{consistency['epsilon_squared']:.0%}",
    help="Effect size ε² of the Kruskal-Wallis test with the author as group.",
)
columns[2].metric(
    "Spread within an author",
    f"{spread['median_author_std']:.2f} stars",
    help=f"Median standard deviation of the ratings of an author "
    f"(all the books: {spread['overall_std']:.2f}).",
)
st.markdown(
    "Authors are very consistent: to guess the rating of a book, its author is far "
    "more informative than its genre."
)

show_figure(plot_author_careers(results["summary_all"], results["summary_standalone"]))
columns = st.columns(2)
for column, key, label in [
    (columns[0], "trend_all", "All books"),
    (columns[1], "trend_standalone", "Standalone books only"),
]:
    trend = results[key]
    column.metric(
        f"{label}: median trend ρ",
        f"{trend['median_rho']:+.2f}",
        help="Median over the authors of the Spearman correlation between "
        "publication year and rating.",
    )
    column.caption(
        f"{trend['share_positive']:.0%} of {trend['n_authors']} authors have a "
        f"positive trend, "
        f"p {format_p_value(trend['p_value'])}"
    )
st.markdown(
    "With all the books there is no clear trend, but the standalone books tend to "
    "be rated lower along a career: the sequels of a series, rated higher, hide "
    "the decline."
)

st.subheader("Explore an author")
careers = results["careers"]
authors = sorted(careers["main_author"].unique())
default = "Stephen King"
author = st.selectbox(
    "Author", authors, index=authors.index(default) if default in authors else 0
)
author_books = careers[careers["main_author"] == author]

show_figure(
    plot_book_ratings(author_books, "publication_year", "Publication year", author)
)
st.dataframe(
    author_books[
        [
            "position",
            "publication_year",
            "title",
            "series",
            "average_rating",
            "rating_deviation",
        ]
    ],
    hide_index=True,
    column_config={
        "position": "#",
        "publication_year": st.column_config.NumberColumn("Year", format="%d"),
        "title": "Title",
        "series": "Series",
        "average_rating": st.column_config.NumberColumn("Rating", format="%.2f"),
        "rating_deviation": st.column_config.NumberColumn(
            "vs author mean",
            format="%+.2f",
            help="Rating minus the mean rating of the author.",
        ),
    },
)
