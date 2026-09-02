from playwright.sync_api import sync_playwright
from urllib.parse import quote
import re


SEARCH_QUERY = "Johor Polls"

SEARCH_URL = (
    "https://www.thestar.com.my/search?query="
    + quote(SEARCH_QUERY)
)


def is_article_url(url):

    pattern = (
        r"^https://www\.thestar\.com\.my/"
        r"news/.+/"
        r"\d{4}/\d{2}/\d{2}/"
        r".+"
    )

    return re.match(pattern, url) is not None


def collect_article_urls(max_articles=100):

    print("Opening browser...")
    print("Opening The Star search page...")

    with sync_playwright() as p:

        browser = p.chromium.launch(headless=True)

        page = browser.new_page()

        page.goto(
            SEARCH_URL,
            wait_until="domcontentloaded",
            timeout=60000
        )

        page.wait_for_timeout(5000)

        print("\nPage title:")
        print(page.title())

        print("\nSearch URL:")
        print(page.url)

        articles = {}

        page_number = 1

        while len(articles) < max_articles:

            print("\n" + "=" * 80)
            print(f"PROCESSING SEARCH PAGE {page_number}")
            print("=" * 80)

            # -------------------------------------------------
            # GET CURRENT PAGE LINKS
            # -------------------------------------------------

            links = page.locator("a").evaluate_all("""
                elements => elements.map(a => ({
                    text: a.innerText.trim(),
                    href: a.href
                }))
            """)

            print("Total links found:", len(links))

            # -------------------------------------------------
            # COLLECT ARTICLE URLS
            # -------------------------------------------------

            before_count = len(articles)

            for link in links:

                title = link["text"]
                url = link["href"]

                if not title:
                    continue

                # Ignore sidebar "Read" links
                if title.lower() == "read":
                    continue

                if is_article_url(url):

                    articles[url] = title

                    if len(articles) >= max_articles:
                        break

            new_articles = len(articles) - before_count

            print(f"New articles found: {new_articles}")
            print(f"Total unique articles: {len(articles)}")

            # -------------------------------------------------
            # STOP WHEN WE HAVE ENOUGH
            # -------------------------------------------------

            if len(articles) >= max_articles:
                break

            # -------------------------------------------------
            # FIND NEXT PAGE BUTTON
            # -------------------------------------------------

            next_button = page.locator(
                "a.next_btn",
                has_text="Next Page"
            )

            if next_button.count() == 0:

                print("\nNo Next Page button found.")
                break

            print("\nNext Page button found.")

            # -------------------------------------------------
            # TRIGGER THE BUTTON USING JAVASCRIPT
            # -------------------------------------------------

            print("Loading next page...")

            try:

                # The Star uses:
                # searchPage.turnpage(20)
                #
                # Instead of physical mouse click,
                # trigger the DOM click using JavaScript.

                page.evaluate("""
                    () => {
                        const button = document.querySelector(
                            'a.next_btn'
                        );

                        if (button) {
                            button.click();
                        }
                    }
                """)

                # Wait for AJAX / JavaScript results
                page.wait_for_timeout(4000)

                print("Next page loaded.")

            except Exception as e:

                print("\nCould not load next page.")
                print("Error:", e)

                break

            page_number += 1

        # -----------------------------------------------------
        # DISPLAY FINAL RESULTS
        # -----------------------------------------------------

        print("\n" + "=" * 80)
        print("FINAL ARTICLE SEARCH RESULTS")
        print("=" * 80)

        for i, (url, title) in enumerate(
            articles.items(),
            start=1
        ):

            print(f"\n{i}. {title}")
            print(url)

        print("\n" + "=" * 80)
        print("FINAL UNIQUE ARTICLE URLS:", len(articles))
        print("=" * 80)

        browser.close()

        return articles


if __name__ == "__main__":

    collect_article_urls(max_articles=100)