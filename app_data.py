"""Data and small helpers shared by the pages of the Streamlit app.

The loading functions are cached, so the data are downloaded, read and cleaned
only once, and not every time the user moves a slider or changes page.
"""

import matplotlib.pyplot as plt
import streamlit as st

from bookinsights.analysis import (
    compare_series_to_first,
    rating_shares,
    summarize_users,
)
from bookinsights.cleaning import clean_books
from bookinsights.data import (
    download_data,
    load_book_tags,
    load_books,
    load_ratings,
    load_tags,
)


@st.cache_resource(show_spinner="Downloading the data (only the first time)...")
def get_data_files():
    """Download the data files that are not yet local and return their paths."""
    return download_data()


@st.cache_data(show_spinner="Loading and cleaning the books...")
def get_books():
    """Return the clean table of the books, one row per book."""
    get_data_files()
    return clean_books(load_books(), load_book_tags(), load_tags())


# The ratings are only read, never modified, so they are cached as a shared
# resource: cache_data would copy the 6 million rows at every access
@st.cache_resource(show_spinner="Loading the ratings...")
def get_ratings():
    """Return the table of the ratings, one row per rating."""
    get_data_files()
    return load_ratings()


@st.cache_data(show_spinner="Describing the users...")
def get_users():
    """Return how each user rates, one row per user."""
    return summarize_users(get_ratings(), get_books())


@st.cache_data
def get_rating_shares():
    """Return the share of the ratings with each number of stars."""
    return rating_shares(get_ratings())


@st.cache_data
def get_series():
    """Return the comparison of each book of a series with the first book."""
    return compare_series_to_first(get_books())


def show_figure(fig):
    """Show a matplotlib figure in the page and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


def format_p_value(p_value):
    """Write a p-value in a readable way, e.g. ``"< 0.001"`` for tiny values."""
    return "< 0.001" if p_value < 0.001 else f"{p_value:.3f}"
