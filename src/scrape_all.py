import json
import time
from pathlib import Path

from collect_urls import collect_article_urls
from scraper import scrape_article


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MAX_ARTICLES = 100

OUTPUT_DIR = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "articles.json"


# ---------------------------------------------------------
# SCRAPE ALL ARTICLES
# ---------------------------------------------------------

def scrape_all_articles():

    print("\n" + "=" * 80)
    print("STEP 1: COLLECTING ARTICLE URLS")
    print("=" * 80)

    # Use the pagination-enabled collector
    articles = collect_article_urls(
        max_articles=MAX_ARTICLES
    )

    print("\n" + "=" * 80)
    print("FOUND ARTICLES:", len(articles))
    print("=" * 80)

    scraped_articles = []
    failed_articles = []

    # -----------------------------------------------------
    # Scrape each article
    # -----------------------------------------------------

    for number, (url, search_title) in enumerate(
        articles.items(),
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"ARTICLE {number}/{len(articles)}"
        )

        print(
            "Search title:",
            search_title
        )

        print(
            "URL:",
            url
        )

        print("-" * 80)

        try:

            article = scrape_article(url)

            # Make sure article exists
            if article is None:

                print(
                    "✗ Failed: scraper returned None"
                )

                failed_articles.append(url)

                continue

            # Make sure content exists
            content = article.get(
                "content",
                ""
            )

            if not content.strip():

                print(
                    "✗ Failed: article content is empty"
                )

                failed_articles.append(url)

                continue

            # Add successfully scraped article
            scraped_articles.append(article)

            print(
                "✓ Successfully scraped"
            )

            print(
                "Characters:",
                len(content)
            )

        except Exception as e:

            print(
                "✗ Failed:"
            )

            print(e)

            failed_articles.append(url)

        # -------------------------------------------------
        # Delay between requests
        # -------------------------------------------------

        time.sleep(2)

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SCRAPING SUMMARY")
    print("=" * 80)

    print(
        "URLs collected:",
        len(articles)
    )

    print(
        "Successfully scraped:",
        len(scraped_articles)
    )

    print(
        "Failed:",
        len(failed_articles)
    )

    if failed_articles:

        print("\nFAILED URLS:")

        for url in failed_articles:

            print(url)

    print("=" * 80)

    return scraped_articles


# ---------------------------------------------------------
# SAVE JSON
# ---------------------------------------------------------

def save_articles(articles):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            articles,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 80)

    print(
        "DATASET SAVED"
    )

    print(
        "File:",
        OUTPUT_FILE
    )

    print(
        "Number of articles:",
        len(articles)
    )

    print("=" * 80)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    articles = scrape_all_articles()

    save_articles(
        articles
    )