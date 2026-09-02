import json
from pathlib import Path
from collections import Counter


INPUT_FILE = Path("data/articles.json")


with open(INPUT_FILE, "r", encoding="utf-8") as file:
    articles = json.load(file)


print("=" * 80)
print("DATASET QUALITY CHECK")
print("=" * 80)

print("\nTotal articles:", len(articles))


# ============================================================
# 1. MISSING VALUES
# ============================================================

required_fields = [
    "title",
    "authors",
    "date_published",
    "category",
    "description",
    "keywords",
    "url",
    "content"
]

print("\n")
print("=" * 80)
print("MISSING VALUES")
print("=" * 80)


for field in required_fields:

    missing_articles = []

    for article in articles:

        value = article.get(field)

        is_missing = (
            value is None
            or (isinstance(value, str) and not value.strip())
            or (isinstance(value, list) and len(value) == 0)
        )

        if is_missing:
            missing_articles.append(article)

    print(f"{field}: {len(missing_articles)} missing")

    if missing_articles:

        for article in missing_articles:
            print(
                f"   - URL: {article.get('url')}"
            )


# ============================================================
# 2. DUPLICATE URL CHECK
# ============================================================

print("\n")
print("=" * 80)
print("DUPLICATE URL CHECK")
print("=" * 80)


urls = [article.get("url") for article in articles]

duplicate_urls = [
    url
    for url, count in Counter(urls).items()
    if count > 1
]

print("Duplicate URLs:", len(duplicate_urls))

for url in duplicate_urls:
    print("-", url)


# ============================================================
# 3. DUPLICATE TITLE CHECK
# ============================================================

print("\n")
print("=" * 80)
print("DUPLICATE TITLE CHECK")
print("=" * 80)


titles = [article.get("title") for article in articles]

duplicate_titles = [
    title
    for title, count in Counter(titles).items()
    if count > 1
]


print("Duplicate titles:", len(duplicate_titles))


for title in duplicate_titles:

    print("\nTitle:", title)

    for article in articles:

        if article.get("title") == title:

            print("URL:", article.get("url"))
            print("Date:", article.get("date_published"))


# ============================================================
# 4. CONTENT LENGTH
# ============================================================

content_lengths = []

for article in articles:

    content = article.get("content", "")

    content_lengths.append(len(content))


print("\n")
print("=" * 80)
print("CONTENT LENGTH")
print("=" * 80)


if content_lengths:

    print(
        "Minimum:",
        min(content_lengths),
        "characters"
    )

    print(
        "Maximum:",
        max(content_lengths),
        "characters"
    )

    print(
        "Average:",
        round(
            sum(content_lengths) / len(content_lengths),
            2
        ),
        "characters"
    )


# ============================================================
# 5. VERY SHORT ARTICLES
# ============================================================

print("\n")
print("=" * 80)
print("VERY SHORT ARTICLES")
print("=" * 80)


short_articles = []


for article in articles:

    content = article.get("content", "")

    if len(content) < 500:

        short_articles.append(article)


print(
    "Articles below 500 characters:",
    len(short_articles)
)


for article in short_articles:

    print("\nTitle:")
    print(article.get("title"))

    print("Characters:")
    print(len(article.get("content", "")))

    print("URL:")
    print(article.get("url"))

    print("Content preview:")
    print(
        article.get("content", "")[:300]
    )


# ============================================================
# 6. DATE RANGE
# ============================================================

print("\n")
print("=" * 80)
print("DATE RANGE")
print("=" * 80)


dates = []

for article in articles:

    date = article.get("date_published")

    if date:
        dates.append(date)


if dates:

    dates.sort()

    print("Earliest:", dates[0])
    print("Latest:", dates[-1])


# ============================================================
# 7. CATEGORIES
# ============================================================

print("\n")
print("=" * 80)
print("CATEGORIES")
print("=" * 80)


categories = Counter()


for article in articles:

    category = article.get("category")

    if category:

        categories[category] += 1


for category, count in categories.items():

    print(
        f"{category}: {count}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("QUALITY CHECK COMPLETE")
print("=" * 80)