"""Readers page: how people rate, and a lookup of single readers (question 6)."""

import streamlit as st

from app_data import (
    get_books,
    get_rating_shares,
    get_ratings,
    get_users,
    show_figure,
)
from bookinsights.analysis import correlate_with_activity, summarize_by_activity
from bookinsights.plots import plot_activity_profiles, plot_rating_shares

MEASURES = {
    "n_ratings": ("Ratings", "{:,.0f}"),
    "mean_rating": ("Mean rating", "{:.2f}"),
    "share_five": ("5-star ratings", "{:.0%}"),
    "relative_harshness": ("Relative harshness", "{:+.2f}"),
    "std_rating": ("Spread", "{:.2f}"),
}


@st.cache_data
def get_correlations():
    """Return the Spearman correlations between activity and each measure."""
    return correlate_with_activity(get_users())


users = get_users()
shares = get_rating_shares()

st.title("Readers")
st.markdown(
    "So far each book was described by its average rating. Here we use the single "
    "ratings, from 1 to 5 stars, given by the users of the dataset."
)

# ------------------------------------------------------------------ star ratings
st.header("How do readers rate?")
show_figure(plot_rating_shares(shares))
st.markdown(
    f"Readers are generous: {shares.loc[[4, 5]].sum():.0%} of the ratings are 4 or "
    f"5 stars, and only {shares.loc[[1, 2]].sum():.0%} are 1 or 2 stars. People "
    "mostly read books they expect to like."
)

# ------------------------------------------------------------------- activity
st.header("Users who rate many books vs the others")
st.info(
    f"Every user in the dataset has rated at least {users['n_ratings'].min()} books "
    f"(the median is {users['n_ratings'].median():.0f}), so we compare active and "
    "very active readers, not occasional ones."
)
st.markdown(
    "- **Relative harshness**: the mean difference between the ratings of a user "
    "and the average ratings of the same books. A negative value means that the "
    "user rates lower than the other readers of the same books.\n"
    "- **Spread**: the standard deviation of the ratings of a user.\n"
    "- **Distance from others**: the mean absolute difference from the average "
    "ratings of the books."
)
n_groups = st.slider("Number of activity groups", min_value=3, max_value=10, value=5)
summary = summarize_by_activity(users, n_groups=n_groups)
show_figure(plot_activity_profiles(summary))
st.markdown(
    "Users who rate more books are slightly harsher: they rate the same books lower "
    "than the others. But they are not more consistent: the spread of their ratings "
    "and their distance from the other readers do not change."
)
with st.expander("Summary table and correlations"):
    st.dataframe(summary.round(3))
    st.dataframe(get_correlations().round(4))

# --------------------------------------------------------------- one reader
st.header("Look up a reader")
st.markdown(
    "Pick a user id to see how this reader rates, compared with the median user."
)
user_id = st.number_input(
    "User id",
    min_value=int(users.index.min()),
    max_value=int(users.index.max()),
    value=1,
    step=1,
)

if user_id not in users.index:
    st.warning("There is no user with this id.")
    st.stop()

profile = users.loc[user_id]
median = users.median()
columns = st.columns(len(MEASURES))
for column, (measure, (label, fmt)) in zip(columns, MEASURES.items()):
    # The difference from the median always shows its sign, e.g. "+0.12"
    delta_fmt = fmt if "+" in fmt else fmt.replace("{:", "{:+")
    column.metric(
        label,
        fmt.format(profile[measure]),
        delta=f"{delta_fmt.format(profile[measure] - median[measure])} vs median",
        delta_color="off",
    )

ratings = get_ratings()
books = get_books()
user_ratings = ratings[ratings["user_id"] == user_id].merge(
    books[["book_id", "title", "main_author", "average_rating"]], on="book_id"
)
user_shares = (
    user_ratings["rating"]
    .value_counts(normalize=True)
    .reindex(range(1, 6), fill_value=0)
)
show_figure(plot_rating_shares(user_shares))

st.subheader("Where this reader disagrees most with the others")
user_ratings["difference"] = user_ratings["rating"] - user_ratings["average_rating"]
disagreements = user_ratings.loc[
    user_ratings["difference"].abs().sort_values(ascending=False).index[:5]
]
st.dataframe(
    disagreements[["title", "main_author", "rating", "average_rating", "difference"]],
    hide_index=True,
    column_config={
        "title": "Title",
        "main_author": "Author",
        "rating": st.column_config.NumberColumn("Their rating", format="%d ⭐"),
        "average_rating": st.column_config.NumberColumn(
            "Average rating", format="%.2f"
        ),
        "difference": st.column_config.NumberColumn("Difference", format="%+.2f"),
    },
)
