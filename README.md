# Book Ratings Insights

What makes a book loved by readers? This project explores the
[goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) dataset (10,000 books
and about 6 million user ratings from Goodreads) to study how genre, publication
year, author and popularity relate to the appreciation of a book.

Project for the Python module of *Coding for Data Science and Data Management*,
Università degli Studi di Milano, A.Y. 2026/27.

## Research Questions

<!-- TODO: refine once the data has been explored -->
1. Are some genres consistently rated higher than others?
2. Are older books rated differently than recent ones?
3. How do the ratings of an author evolve across their works?

## Data

| File | Content |
|---|---|
| `books.csv` | metadata of the 10,000 books (authors, title, year, average rating, ...) |
| `ratings.csv` | user ratings, from 1 to 5 (~6 million rows, ~70 MB) |
| `tags.csv` | tag id → tag name |
| `book_tags.csv` | tags (user shelves/genres) assigned to each book, with counts |

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
│   ├── data.py            # download, load and clean the data
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
