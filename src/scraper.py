import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta


# ================================================================
# REQUEST HEADERS
# ================================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}


# ================================================================
# DATE CONVERSION
# ================================================================

def convert_to_malaysia_date(date_string):
    """
    Convert UTC datetime to Malaysia local date (UTC+8).

    Example:
        2026-07-09T16:00:00.000Z
        ->
        2026-07-10
    """

    if not date_string:
        return None

    try:

        dt = datetime.fromisoformat(
            date_string.replace("Z", "+00:00")
        )

        malaysia_tz = timezone(
            timedelta(hours=8)
        )

        dt_malaysia = dt.astimezone(
            malaysia_tz
        )

        return dt_malaysia.strftime(
            "%Y-%m-%d"
        )

    except Exception:

        return date_string


# ================================================================
# META TAG HELPER
# ================================================================

def get_meta_content(
    soup,
    name=None,
    property_name=None
):
    """
    Extract content from an HTML meta tag.

    Supports:

        <meta name="description" content="...">

    and:

        <meta property="og:title" content="...">
    """

    if name:

        tag = soup.find(
            "meta",
            attrs={
                "name": name
            }
        )

    elif property_name:

        tag = soup.find(
            "meta",
            attrs={
                "property": property_name
            }
        )

    else:

        return None

    if tag:

        return tag.get("content")

    return None


# ================================================================
# SCRAPE ARTICLE
# ================================================================

def scrape_article(url):

    print("\n" + "=" * 80)
    print("SCRAPING:")
    print(url)
    print("=" * 80)

    # ============================================================
    # 1. REQUEST ARTICLE
    # ============================================================

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        print(
            "Status:",
            response.status_code
        )

        if response.status_code != 200:

            print(
                "Failed to retrieve article."
            )

            return None

        soup = BeautifulSoup(
            response.text,
            "lxml"
        )

    except Exception as e:

        print(
            "ERROR while requesting article:"
        )

        print(e)

        return None


    # ============================================================
    # 2. INITIALIZE METADATA
    # ============================================================

    title = None

    description = None

    date_published = None

    authors = []

    category = None

    keywords = []


    # ============================================================
    # 3. EXTRACT JSON-LD METADATA
    # ============================================================

    json_ld_scripts = soup.find_all(
        "script",
        type="application/ld+json"
    )

    json_ld_success = False


    for script in json_ld_scripts:

        try:

            # ----------------------------------------------------
            # Some script tags may not contain text
            # ----------------------------------------------------

            if not script.string:

                continue


            data = json.loads(
                script.string
            )


            # ----------------------------------------------------
            # JSON-LD can sometimes be a list
            # ----------------------------------------------------

            if isinstance(data, list):

                for item in data:

                    if (
                        isinstance(item, dict)
                        and item.get("@type") == "NewsArticle"
                    ):

                        data = item

                        break


            # ----------------------------------------------------
            # JSON-LD can sometimes use @graph
            # ----------------------------------------------------

            elif (
                isinstance(data, dict)
                and "@graph" in data
            ):

                for item in data["@graph"]:

                    if (
                        isinstance(item, dict)
                        and item.get("@type") == "NewsArticle"
                    ):

                        data = item

                        break


            # ----------------------------------------------------
            # Extract NewsArticle
            # ----------------------------------------------------

            if (
                isinstance(data, dict)
                and data.get("@type") == "NewsArticle"
            ):

                json_ld_success = True


                # =================================================
                # TITLE
                # =================================================

                title = data.get(
                    "headline"
                )


                # =================================================
                # DESCRIPTION
                # =================================================

                description = data.get(
                    "description"
                )


                # =================================================
                # DATE
                # =================================================

                date_published = (
                    convert_to_malaysia_date(
                        data.get(
                            "datePublished"
                        )
                    )
                )


                # =================================================
                # CATEGORY
                # =================================================

                category = data.get(
                    "articleSection"
                )


                # =================================================
                # AUTHORS
                # =================================================

                author_data = data.get(
                    "author",
                    []
                )


                if isinstance(
                    author_data,
                    dict
                ):

                    author_data = [
                        author_data
                    ]


                for author in author_data:

                    if isinstance(
                        author,
                        dict
                    ):

                        name = author.get(
                            "name"
                        )

                        if name:

                            authors.append(
                                name
                            )

                    elif isinstance(
                        author,
                        str
                    ):

                        authors.append(
                            author
                        )


                # =================================================
                # KEYWORDS
                # =================================================

                keyword_data = data.get(
                    "keywords",
                    []
                )


                if isinstance(
                    keyword_data,
                    str
                ):

                    keywords = [
                        keyword.strip()
                        for keyword in keyword_data.split(",")
                        if keyword.strip()
                    ]


                elif isinstance(
                    keyword_data,
                    list
                ):

                    keywords = keyword_data


                # We found the NewsArticle JSON-LD.
                break


        except Exception:

            print(
                "WARNING: JSON-LD could not be parsed."
            )

            continue


    # ============================================================
    # 4. HTML METADATA FALLBACK
    # ============================================================
    #
    # Some The Star pages contain malformed JSON-LD.
    #
    # Therefore, if JSON-LD did not provide a field,
    # use standard HTML metadata.
    #
    # ============================================================


    # ------------------------------------------------------------
    # TITLE FALLBACK
    # ------------------------------------------------------------

    if not title:

        title = get_meta_content(
            soup,
            property_name="og:title"
        )


    if not title:

        h1 = soup.find("h1")

        if h1:

            title = h1.get_text(
                " ",
                strip=True
            )


    # ------------------------------------------------------------
    # DESCRIPTION FALLBACK
    # ------------------------------------------------------------

    if not description:

        description = get_meta_content(
            soup,
            property_name="og:description"
        )


    if not description:

        description = get_meta_content(
            soup,
            name="description"
        )


    # ------------------------------------------------------------
    # DATE FALLBACK
    # ------------------------------------------------------------

    if not date_published:

        published_time = get_meta_content(
            soup,
            property_name="article:published_time"
        )

        date_published = (
            convert_to_malaysia_date(
                published_time
            )
        )


    # ------------------------------------------------------------
    # CATEGORY FALLBACK
    # ------------------------------------------------------------

    if not category:

        category = get_meta_content(
            soup,
            property_name="article:section"
        )


    # ------------------------------------------------------------
    # KEYWORDS FALLBACK
    # ------------------------------------------------------------

    if not keywords:

        raw_keywords = get_meta_content(
            soup,
            name="keywords"
        )

        if raw_keywords:

            keywords = [
                keyword.strip()
                for keyword in raw_keywords.split(",")
                if keyword.strip()
            ]


    # ============================================================
    # 5. REMOVE DUPLICATE AUTHORS
    # ============================================================

    authors = list(
        dict.fromkeys(
            authors
        )
    )


    # ============================================================
    # 6. EXTRACT ARTICLE BODY
    # ============================================================

    content = []


    # ------------------------------------------------------------
    # The Star article body
    # ------------------------------------------------------------

    story_body = soup.find(
        "div",
        id="story-body"
    )


    if story_body:

        paragraphs = story_body.find_all(
            "p"
        )


        for paragraph in paragraphs:

            text = paragraph.get_text(
                " ",
                strip=True
            )


            if not text:

                continue


            # ----------------------------------------------------
            # Remove obvious footer / navigation content
            # ----------------------------------------------------

            if text.startswith(
                "For the latest updates"
            ):

                continue


            if text.startswith(
                "Copyright ©"
            ):

                continue


            if text.startswith(
                "Best viewed on"
            ):

                continue


            content.append(
                text
            )


    # ============================================================
    # 7. FALLBACK ARTICLE BODY EXTRACTION
    # ============================================================

    if not content:

        print(
            "WARNING: story-body not found or empty."
        )


        paragraphs = soup.find_all(
            "p"
        )


        excluded_phrases = [
            "Stay signed in to save",
            "Copyright ©",
            "Best viewed on",
            "Thank you for your report",
            "For the latest updates"
        ]


        for paragraph in paragraphs:

            text = paragraph.get_text(
                " ",
                strip=True
            )


            if not text:

                continue


            # ----------------------------------------------------
            # Remove navigation/footer text
            # ----------------------------------------------------

            if any(
                phrase.lower() in text.lower()
                for phrase in excluded_phrases
            ):

                continue


            content.append(
                text
            )


    # ============================================================
    # 8. COMBINE ARTICLE CONTENT
    # ============================================================

    article_content = "\n\n".join(
        content
    )


    # ============================================================
    # 9. CREATE ARTICLE RECORD
    # ============================================================

    article = {

        "title": title,

        "authors": authors,

        "date_published": date_published,

        "category": category,

        "description": description,

        "keywords": keywords,

        "url": url,

        "content": article_content

    }


    # ============================================================
    # 10. PRINT ARTICLE SUMMARY
    # ============================================================

    print("\nARTICLE INFORMATION")
    print("-" * 80)


    print(
        "Title:",
        title
    )


    print(
        "Authors:",
        ", ".join(authors)
    )


    print(
        "Date:",
        date_published
    )


    print(
        "Category:",
        category
    )


    print(
        "Keywords:",
        keywords
    )


    print(
        "Content paragraphs:",
        len(content)
    )


    print(
        "Content characters:",
        len(article_content)
    )


    print("\nFirst 500 characters:")
    print("-" * 80)


    print(
        article_content[:500]
    )


    return article


# ================================================================
# TEST
# ================================================================

if __name__ == "__main__":

    test_urls = [

        # --------------------------------------------------------
        # Normal article
        # --------------------------------------------------------

        "https://www.thestar.com.my/news/nation/2026/07/10/johor-polls-no-pas-as-appointed-representative",


        # --------------------------------------------------------
        # Normal article
        # --------------------------------------------------------

        "https://www.thestar.com.my/news/nation/2026/06/25/johor-polls-bersatu-announces-candidates-to-contest-16-seats",


        # --------------------------------------------------------
        # Normal article
        # --------------------------------------------------------

        "https://www.thestar.com.my/news/nation/2026/06/25/johor-polls-mipp-to-contest-five-seats-under-perikatan",


        # --------------------------------------------------------
        # Normal article
        # --------------------------------------------------------

        "https://www.thestar.com.my/news/nation/2026/06/26/wawasan-to-skip-johor-polls",


        # --------------------------------------------------------
        # Normal article
        # --------------------------------------------------------

        "https://www.thestar.com.my/news/nation/2026/06/27/johor-polls-fine-weather-sets-tone-for-nomination-day",


        # --------------------------------------------------------
        # Article with malformed JSON-LD
        # --------------------------------------------------------

        "https://www.thestar.com.my/news/nation/2026/07/09/johor-polls-039free-najib039-banners-meant-to-smear-yong-peng-candidate-says-mca-youth"

    ]


    successful = 0

    failed = 0


    for i, url in enumerate(
        test_urls,
        start=1
    ):

        print("\n")
        print(
            "#" * 100
        )

        print(
            f"TEST ARTICLE {i}/{len(test_urls)}"
        )

        print(
            "#" * 100
        )


        article = scrape_article(
            url
        )


        if (
            article
            and article["content"]
        ):

            successful += 1

        else:

            failed += 1


    # ============================================================
    # TEST SUMMARY
    # ============================================================

    print("\n")
    print(
        "=" * 100
    )

    print(
        "TEST SUMMARY"
    )

    print(
        "=" * 100
    )


    print(
        "Successful:",
        successful
    )


    print(
        "Failed:",
        failed
    )