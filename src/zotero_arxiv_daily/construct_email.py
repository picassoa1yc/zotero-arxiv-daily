from __future__ import annotations

import math
from html import escape

from .protocol import Paper


framework = """
<html>
  <body style="font-family: Arial, sans-serif; line-height: 1.55;">
    <h2>Daily Paper Recommendation</h2>

    __CONTENT__

    <hr>
    <p style="color: #777; font-size: 12px;">
      To unsubscribe, remove your email in your GitHub Action settings.
    </p>
  </body>
</html>
"""


def _safe(value: object | None) -> str:
    if value is None:
        return ""
    return escape(str(value))


def _format_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown Authors"

    if len(authors) <= 5:
        return ", ".join(authors)

    return ", ".join(authors[:3] + ["..."] + authors[-2:])


def _format_list(items: list[str] | None, max_items: int = 5, empty: str = "Not available") -> str:
    if not items:
        return empty

    shown = items[:max_items]
    result = ", ".join(shown)

    if len(items) > max_items:
        result += ", ..."

    return result


def get_empty_html() -> str:
    return """
    <p>No Papers Today. Take a Rest!</p>
    """


def get_stars(score: float) -> str:
    full_star = "⭐"
    half_star = "⭐"
    low = 6
    high = 8

    if score <= low:
        return ""

    if score >= high:
        return full_star * 5

    interval = (high - low) / 10
    star_num = math.ceil((score - low) / interval)
    full_star_num = int(star_num / 2)
    half_star_num = star_num - full_star_num * 2

    return full_star * full_star_num + half_star * half_star_num


def _link_html(label: str, url: str | None) -> str:
    if not url:
        return ""

    return f'<a href="{_safe(url)}">{_safe(label)}</a>'


def get_block_html(paper: Paper) -> str:
    rate = round(paper.score, 1) if paper.score is not None else "Unknown"
    stars = get_stars(paper.score) if paper.score is not None else ""

    authors = _format_authors(paper.authors)
    affiliations = _format_list(paper.affiliations, max_items=5, empty="Not available")
    topics = _format_list(paper.topics, max_items=5, empty="Not available")

    journal = paper.journal or "Preprint / Unknown Journal"
    publication_date = paper.publication_date or "Unknown date"
    doi = paper.doi or "Not available"

    tldr = paper.tldr or paper.abstract or "No TLDR available."

    links = []

    if paper.url:
        links.append(_link_html("Landing Page", paper.url))

    if paper.doi:
        links.append(_link_html("DOI", paper.doi))

    if paper.pdf_url:
        links.append(_link_html("PDF", paper.pdf_url))

    links_html = " | ".join([link for link in links if link])

    if not links_html:
        links_html = "No link available"

    return f"""
    <div style="
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 14px 16px;
      margin: 16px 0;
      background: #fff;
    ">
      <h3 style="margin: 0 0 8px 0;">
        {_safe(paper.title)}
      </h3>

      <p style="margin: 4px 0;">
        <b>Authors:</b> {_safe(authors)}
      </p>

      <p style="margin: 4px 0;">
        <b>Journal:</b> {_safe(journal)}
      </p>

      <p style="margin: 4px 0;">
        <b>Published:</b> {_safe(publication_date)}
      </p>

      <p style="margin: 4px 0;">
        <b>DOI:</b> {_safe(doi)}
      </p>

      <p style="margin: 4px 0;">
        <b>Affiliations:</b> {_safe(affiliations)}
      </p>

      <p style="margin: 4px 0;">
        <b>Topics:</b> {_safe(topics)}
      </p>

      <p style="margin: 4px 0;">
        <b>Source:</b> {_safe(paper.source)}
      </p>

      <p style="margin: 4px 0;">
        <b>Relevance:</b> {_safe(rate)} {_safe(stars)}
      </p>

      <p style="margin: 8px 0;">
        <b>TLDR:</b> {_safe(tldr)}
      </p>

      <p style="margin: 8px 0;">
        <b>Links:</b> {links_html}
      </p>
    </div>
    """


def render_email(papers: list[Paper]) -> str:
    if len(papers) == 0:
        return framework.replace("__CONTENT__", get_empty_html())

    parts = [get_block_html(p) for p in papers]
    content = "\n".join(parts)

    return framework.replace("__CONTENT__", content)
