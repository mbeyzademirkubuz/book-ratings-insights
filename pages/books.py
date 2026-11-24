"""Books page: genres, publication years and popularity (questions 1 to 3)."""

import streamlit as st

from app_data import format_p_value, get_books, show_figure
from bookinsights.analysis import (
    correlate_with_rating,
    summarize_by_group,
    summarize_by_popularity,
    test_group_differences,
)
from bookinsights.plots import plot_popularity_vs_rating, plot_ratings_by_group

BOOK_COLUMNS = {
    "title": "Title",
    "main_author": "Author",
    "publication_year": st.column_config.NumberColumn("Year", format="%d"),
    "average_rating": st.column_config.NumberColumn("Rating", format="%.2f"),
    "work_ratings_count": st.column_config.NumberColumn("Ratings", format="localized"),
}

books = get_books()

st.title("Books")
st.markdown(
    "Do the genre, the publication year and the popularity of a book tell us "
    "something about its rating? Each book is one observation, described by its "
    "average rating on Goodreads, from 1 to 5 stars."
)

# ---------------------------------------------------------------- genres and eras
st.header("Genres and publication eras")
grouping = st.segmented_control(
    "Compare the books by", ["Genre", "Publication era"], default="Genre", required=True
)
column = "genre" if grouping == "Genre" else "era"
selected = books
if column == "era" and st.checkbox(
    "Leave out the classics",
    help="Most of the old books in the data are classics, a genre rated below average.",
):
    selected = books[books["genre"] != "classics"]

# Genres have no natural order, so they are sorted by median; eras keep their order
sort_by_median = column == "genre"
show_figure(plot_ratings_by_group(selected, column, sort_by_median=sort_by_median))

test = test_group_differences(selected, column)
columns = st.columns(3)
columns[0].metric("Kruskal-Wallis H", f"{test['statistic']:,.0f}")
columns[1].metric("p-value", format_p_value(test["p_value"]))
columns[2].metric(
    "Effect size ε²",
    f"{test['epsilon_squared']:.3f}",
    help="Share of the variability of the ranks of the ratings explained by the groups.",
)
st.markdown(
    f"The {grouping.lower()} explains about **{test['epsilon_squared']:.0%}** of the "
    f"variability of the ratings ({test['n_books']:,} books in "
    f"{test['n_groups']} groups). With so many books even small differences are "
    "significant, so the effect size tells more than the p-value."
)
with st.expander("Summary table"):
    st.dataframe(
        summarize_by_group(selected, column, sort_by_median=sort_by_median).round(3)
    )

# ------------------------------------------------------------------ one genre
st.header("Explore a genre")
genres = sorted(books["genre"].dropna().unique())
genre = st.selectbox("Genre", genres, index=genres.index("fantasy"))
order = st.segmented_control(
    "Show the books",
    ["Highest rated", "Most rated"],
    default="Highest rated",
    required=True,
)

in_genre = books[books["genre"] == genre]
columns = st.columns(3)
columns[0].metric("Books", f"{len(in_genre):,}")
columns[1].metric("Median rating", f"{in_genre['average_rating'].median():.2f}")
columns[2].metric(
    "Median number of ratings", f"{in_genre['work_ratings_count'].median():,.0f}"
)

sort_column = "average_rating" if order == "Highest rated" else "work_ratings_count"
st.dataframe(
    in_genre.nlargest(10, sort_column)[list(BOOK_COLUMNS)],
    hide_index=True,
    column_config=BOOK_COLUMNS,
)

# ------------------------------------------------------------------ popularity
st.header("Popularity")
st.markdown(
    "Popularity is measured by the number of ratings received by all the editions "
    "of a book. The books are split into groups of the same size, from the least "
    "to the most rated, and the line follows the median rating of each group."
)
n_groups = st.slider("Number of popularity groups", min_value=4, max_value=20, value=10)
show_figure(plot_popularity_vs_rating(books, n_groups=n_groups))

correlation = correlate_with_rating(books, "work_ratings_count")
columns = st.columns(2)
columns[0].metric("Spearman ρ", f"{correlation['rho']:.2f}")
columns[1].metric("p-value", format_p_value(correlation["p_value"]))
st.markdown(
    "The line of the medians is almost flat: the most rated books are not rated "
    "higher. They only avoid the lowest ratings, probably because a book that "
    "readers dislike stops being recommended and read."
)
with st.expander("Summary table"):
    st.dataframe(summarize_by_popularity(books, n_groups=n_groups).round(3))
