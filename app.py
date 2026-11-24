"""Streamlit app: an interactive tour of the goodbooks-10k ratings.

Run it from the project folder with ``streamlit run app.py``.
"""

import streamlit as st

from app_data import get_books, get_ratings, show_figure
from bookinsights.plots import plot_rating_distribution

st.set_page_config(page_title="Book Ratings Insights", page_icon="📚")


def home():
    """Draw the home page: the project, the data and the main answers."""
    books = get_books()
    ratings = get_ratings()

    st.title("📚 Book Ratings Insights")
    st.markdown(
        "What makes readers like a book? This app explores the "
        "[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset: the "
        "10,000 most popular books on Goodreads and about 6 million ratings given "
        "by their readers. All the analysis comes from the `bookinsights` package; "
        "the pages let you explore it."
    )

    columns = st.columns(4)
    columns[0].metric("Books", f"{len(books):,}")
    columns[1].metric("Ratings", f"{len(ratings):,}")
    columns[2].metric("Readers", f"{ratings['user_id'].nunique():,}")
    columns[3].metric("Genres", books["genre"].nunique())

    st.header("How are the books rated?")
    show_figure(plot_rating_distribution(books))
    st.markdown(
        "The average ratings are concentrated in a narrow range: most books are "
        "rated between 3.5 and 4.5 stars, so even a difference of 0.1 stars between "
        "two groups of books is not negligible."
    )

    st.header("Six questions, six short answers")
    st.markdown(
        """
1. **Genres: yes.** Comics, poetry and children's books are rated higher,
   literary fiction, humor and horror lower. The genre explains about 13% of the
   variability of the ratings: a moderate effect.
2. **Publication year: not really.** The era explains only about 1% of the
   variability and there is no trend over the years. The slightly lower ratings of
   the oldest books are due to the fact that most of them are classics.
3. **Popularity: no.** The number of ratings is almost unrelated to the rating.
   The most popular books only avoid the lowest ratings.
4. **Series: rated higher, but by fewer readers.** The last main book of a series
   is rated about 0.12 stars higher than the first one, but fewer and fewer
   readers rate the books as the series goes on.
5. **Authors: very stable.** The author explains about 58% of the variability of
   the ratings, much more than the genre. Standalone books tend to be rated
   slightly lower along a career.
6. **Readers: generous.** About 70% of the ratings are 4 or 5 stars. Users who
   rate many books are slightly harsher, but not more consistent.

A common thread runs through these answers: **self-selection**. Readers choose
what they read, and they rate what they chose.
"""
    )

    st.header("Explore")
    st.page_link(books_page, label="Books: genres, publication years and popularity")
    st.page_link(authors_page, label="Authors and series: careers and sequels")
    st.page_link(readers_page, label="Readers: how people rate, and a reader lookup")


home_page = st.Page(home, title="Home", icon="📚", default=True)
books_page = st.Page("pages/books.py", title="Books", icon="📖")
authors_page = st.Page(
    "pages/authors_and_series.py", title="Authors and series", icon="✍️"
)
readers_page = st.Page("pages/readers.py", title="Readers", icon="⭐")

st.navigation([home_page, books_page, authors_page, readers_page]).run()
