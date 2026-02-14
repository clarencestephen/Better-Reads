#!/usr/bin/env python3
"""
Better-Reads: LLM-Powered Book Recommendations (Collaborative Filtering Approach)

Uses an LLM to simulate collaborative filtering — recommending books based on
reader preference patterns (what readers of similar books also enjoy), inspired
by the ALS-based recommendation system trained on 26M+ Amazon/Goodreads reviews.

Usage:
    python recommend.py "The Great Gatsby"
    python recommend.py "Pride and Prejudice" --n 5
"""

import argparse
import json
import re
import sys

try:
    from openai import OpenAI
except ImportError:
    print("Install openai: pip install openai")
    sys.exit(1)


def get_llm_client():
    """Connect to available LLM backend (vLLM → Ollama → OpenAI)."""
    # Try vLLM
    try:
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
        client.models.list()
        return client, "vLLM"
    except Exception:
        pass

    # Try Ollama
    try:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        client.models.list()
        return client, "Ollama"
    except Exception:
        pass

    # Try OpenAI
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key), "OpenAI"

    print("No LLM backend found. Start vLLM, Ollama, or set OPENAI_API_KEY.")
    sys.exit(1)


def get_recommendations(book_title, n=10):
    """
    Get book recommendations based on collaborative filtering patterns.

    Leverages LLM knowledge of reader review patterns — what Amazon/Goodreads
    readers of similar books frequently recommend together.

    Args:
        book_title: Title of the book to get recommendations for.
        n: Number of recommendations to return.

    Returns:
        list[dict]: Recommendations as [{"title": ..., "author": ...}, ...]
    """
    client, backend = get_llm_client()
    print(f"Using {backend} backend")

    prompt = (
        f'Based on your knowledge of reader preferences and book review patterns, '
        f'what books do readers who enjoyed "{book_title}" also tend to love? Consider '
        f'what Amazon and Goodreads reviewers of similar books frequently recommend '
        f'together — collaborative filtering based on real reading behavior.\n\n'
        f'Recommend {n} books that readers of "{book_title}" would enjoy, based on '
        f'collaborative reading patterns rather than just thematic similarity.\n\n'
        f'Return ONLY a JSON array of objects with "title" and "author" keys. '
        f'No explanations, no markdown, no extra text. Example:\n'
        f'[{{"title": "Example Book", "author": "Example Author"}}]'
    )

    models = client.models.list()
    model = models.data[0].id if models.data else "gpt-3.5-turbo"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a book recommendation engine. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=800,
    )

    text = response.choices[0].message.content.strip()

    # Parse JSON from response
    cleaned = re.sub(r'```(?:json)?\s*', '', text)
    cleaned = re.sub(r'```', '', cleaned)
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group())
        return [
            {"title": item["title"].strip(), "author": item.get("author", "Unknown").strip()}
            for item in items
            if isinstance(item, dict) and "title" in item
        ][:n]
    except (json.JSONDecodeError, TypeError):
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Better-Reads: LLM-powered book recommendations (collaborative filtering)"
    )
    parser.add_argument("book_title", help="Title of the book to get recommendations for")
    parser.add_argument("--n", type=int, default=10, help="Number of recommendations (default: 10)")
    args = parser.parse_args()

    print(f'\nFinding books similar to "{args.book_title}"...\n')
    recs = get_recommendations(args.book_title, args.n)

    if not recs:
        print("No recommendations found.")
        return

    print(f"Recommendations for readers of \"{args.book_title}\":\n")
    for i, rec in enumerate(recs, 1):
        print(f"  {i:2d}. {rec['title']} — {rec['author']}")
    print()


if __name__ == "__main__":
    main()
