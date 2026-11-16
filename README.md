# Book Ratings Insights

What makes a book loved by readers? This project explores the
[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset (10,000 books
and about 6 million user ratings from Goodreads) to study how genre, publication
year, author and popularity relate to the appreciation of a book.

Project for the Python module of *Coding for Data Science and Data Management*,
Università degli Studi di Milano, A.Y. 2026/27.

## Research Questions

**Books**

1. Are some genres consistently rated higher than others?
2. Are older books rated differently than recent ones?
3. Is popularity the same as appreciation? Do the most read books get higher
   ratings?

**Authors and series**

4. How do the ratings of an author evolve across their works?
5. What happens as a series goes on? Are later books rated higher or lower than
   the first one?

**Readers**

6. How do readers rate? Are users who rate many books harsher, or more
   consistent, than occasional readers?

## Data

| File | Content |
|---|---|
| `books.csv` | metadata of the 10,000 books (authors, title, year, average rating, ...) |
| `ratings.csv` | user ratings, from 1 to 5 (~6 million rows, ~70 MB) |
| `tags.csv` | tag id → tag name |
| `book_tags.csv` | tags (user shelves/genres) assigned to each book, with counts |

Ratings are linked to books through `book_id`, while tags are linked through
`goodreads_book_id`.

The data files are not stored in this repository: they are downloaded into
`data/` by the code (see *Usage*).

## Project Structure

```
book-ratings-insights/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # downloaded data (not tracked by Git)
├── bookinsights/          # the package: all the logic of the project
│   ├── __init__.py
│   ├── data.py            # download and load the data
│   ├── cleaning.py        # clean the data and derive genres, eras, series
│   ├── analysis.py        # statistics
│   └── plots.py           # charts
└── notebooks/             # presentation of the analysis
```

## Setup

Requires Python 3.12 or later.

```bash
git clone <repository-url>
cd book-ratings-insights
python3 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

<!-- TODO: how to download the data and run the notebook -->

## Libraries

- `pandas`: data loading and manipulation
- `numpy`, `scipy`: numerical and statistical computations
- `matplotlib`: visualisation
- `requests`: download of the data files
- `jupyterlab`: presentation notebook
- `streamlit`: interactive web application
