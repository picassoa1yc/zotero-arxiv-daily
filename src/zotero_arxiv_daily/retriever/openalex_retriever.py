from __future__ import annotations

from datetime import datetime, timedelta, timezone
from time import sleep
from typing import Any

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


OPENALEX_WORKS_API = "https://api.openalex.org/works"
REQUEST_TIMEOUT = (10, 60)


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default

    try:
        return config.get(key, default)
    except AttributeError:
        return getattr(config, key, default)


def _cfg_list(config: Any, key: str, default: list[Any] | None = None) -> list[Any]:
    value = _cfg_get(config, key, default or [])

    if value is None:
        return []

    # OmegaConf ListConfig or normal list/tuple
    if not isinstance(value, str):
        try:
            return list(value)
        except TypeError:
            pass

    # A single string value
    return [value]


def restore_openalex_abstract(abstract_inverted_index: dict[str, list[int]] | None) -> str:
    """
    OpenAlex stores abstracts as an inverted index:
    {
        "This": [0],
        "paper": [1],
        ...
    }

    This function restores it into plain text.
    """
    if not abstract_inverted_index:
        return ""

    positioned_words: list[tuple[int, str]] = []

    for word, positions in abstract_inverted_index.items():
        for position in positions:
            positioned_words.append((position, word))

    positioned_words.sort(key=lambda x: x[0])
    return " ".join(word for _, word in positioned_words)


def _unique_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        item = item.strip()

        if not item:
            continue

        key = item.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def extract_authors(work: dict[str, Any]) -> list[str]:
    authors = []

    for authorship in work.get("authorships", []) or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")

        if name:
            authors.append(name)

    return _unique_keep_order(authors)


def extract_institutions(work: dict[str, Any]) -> list[str]:
    institutions = []

    for authorship in work.get("authorships", []) or []:
        for institution in authorship.get("institutions", []) or []:
            name = institution.get("display_name")

            if name:
                institutions.append(name)

    return _unique_keep_order(institutions)


def extract_journal(work: dict[str, Any]) -> str | None:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}

    journal = source.get("display_name")

    if journal:
        return journal

    for location in work.get("locations", []) or []:
        source = (location or {}).get("source") or {}
        journal = source.get("display_name")

        if journal:
            return journal

    return None


def extract_publisher(work: dict[str, Any]) -> str | None:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}

    publisher = source.get("host_organization_name")

    if publisher:
        return publisher

    return None


def extract_pdf_url(work: dict[str, Any]) -> str | None:
    primary_location = work.get("primary_location") or {}

    pdf_url = primary_location.get("pdf_url")

    if pdf_url:
        return pdf_url

    open_access = work.get("open_access") or {}
    oa_url = open_access.get("oa_url")

    if oa_url and str(oa_url).lower().endswith(".pdf"):
        return oa_url

    return None


def extract_landing_url(work: dict[str, Any]) -> str:
    doi = work.get("doi")

    if doi:
        return doi

    primary_location = work.get("primary_location") or {}
    landing_page_url = primary_location.get("landing_page_url")

    if landing_page_url:
        return landing_page_url

    return work.get("id") or ""


def extract_topics(work: dict[str, Any]) -> list[str]:
    topics = []

    for topic in work.get("topics", []) or []:
        name = topic.get("display_name")

        if name:
            topics.append(name)

    return _unique_keep_order(topics)


def normalize_identifier(work: dict[str, Any]) -> str:
    doi = work.get("doi")

    if doi:
        return str(doi).lower().strip()

    openalex_id = work.get("id")

    if openalex_id:
        return str(openalex_id).lower().strip()

    title = work.get("display_name") or ""
    return title.lower().strip()


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)
    
def get_primary_location(work: dict) -> dict:
    return work.get("primary_location") or {}


def get_primary_source(work: dict) -> dict:
    primary_location = get_primary_location(work)
    return primary_location.get("source") or {}


def is_journal_article(work: dict) -> bool:
    primary_location = get_primary_location(work)
    primary_source = get_primary_source(work)

    source_type = primary_source.get("type")
    version = primary_location.get("version")

    if source_type != "journal":
        return False

    if version == "submittedVersion":
        return False

    return True

def normalize_journal_name(name: str | None) -> str:
    if not name:
        return ""
    return name.lower().replace("&", "and").strip()


def get_journal_tier(journal: str | None, tier1: list[str], tier2: list[str], tier3: list[str]) -> str:
    journal_key = normalize_journal_name(journal)

    tier1_set = {normalize_journal_name(j) for j in tier1}
    tier2_set = {normalize_journal_name(j) for j in tier2}
    tier3_set = {normalize_journal_name(j) for j in tier3}

    if journal_key in tier1_set:
        return "tier1"

    if journal_key in tier2_set:
        return "tier2"

    if journal_key in tier3_set:
        return "tier3"

    return "unknown"

@register_retriever("openalex")
class OpenAlexRetriever(BaseRetriever):
    """
    Retrieve recent journal articles from OpenAlex.

    This retriever is designed for geography / ecology / hydrology / remote sensing
    paper recommendation, where arXiv coverage is often insufficient.
    """

    def __init__(self, config):
        super().__init__(config)

        self.days_back = int(_cfg_get(self.retriever_config, "days_back", 7))
        self.per_query = int(_cfg_get(self.retriever_config, "per_query", 50))
        self.max_results = int(_cfg_get(self.retriever_config, "max_results", 200))
        self.mailto = _cfg_get(self.retriever_config, "mailto", None)

        self.queries = [str(q) for q in _cfg_list(self.retriever_config, "queries")]
        self.required_keywords = [
            str(k) for k in _cfg_list(self.retriever_config, "required_keywords")
        ]
        self.exclude_keywords = [
            str(k) for k in _cfg_list(self.retriever_config, "exclude_keywords")
        ]
        self.include_journals = [
            str(j).lower()
            for j in _cfg_list(self.retriever_config, "include_journals")
        ]

        if not self.queries:
            raise ValueError("source.openalex.queries must contain at least one query.")

    def _date_window(self) -> tuple[str, str]:
        today = datetime.now(timezone.utc).date()
        from_date = today - timedelta(days=self.days_back)
        return from_date.isoformat(), today.isoformat()

    def _build_filters(self, from_date: str, to_date: str) -> str:
        filters = [
            f"from_publication_date:{from_date}",
            f"to_publication_date:{to_date}",
            "type:article",
            "has_abstract:true",
            "is_retracted:false",
        ]

        return ",".join(filters)

    def _request_openalex(self, query: str, from_date: str, to_date: str) -> list[dict[str, Any]]:
        params = {
            "search": query,
            "filter": self._build_filters(from_date, to_date),
            "per-page": min(self.per_query, 200),
            "sort": "publication_date:desc",
        }

        if self.mailto:
            params["mailto"] = self.mailto

        retry_num = 5
        delay_time = 10

        for i in range(retry_num):
            try:
                response = requests.get(
                    OPENALEX_WORKS_API,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("results", []) or []
            except Exception as exc:
                if i == retry_num - 1:
                    raise exc

                logger.warning(
                    f"Failed to retrieve OpenAlex papers for query={query!r}: {exc}. "
                    f"Retry in {delay_time} seconds."
                )
                sleep(delay_time)

        return []

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        from_date, to_date = self._date_window()

        logger.info(
            f"Retrieving OpenAlex journal articles from {from_date} to {to_date} "
            f"with {len(self.queries)} queries."
        )

        queries = self.queries

        if self.config.executor.debug:
            queries = queries[:3]

        raw_papers: list[dict[str, Any]] = []
        seen_ids = set()

        for query in queries:
            works = self._request_openalex(query, from_date, to_date)

            logger.info(f"OpenAlex query={query!r}: retrieved {len(works)} works.")

            for work in works:
                identifier = normalize_identifier(work)

                if not identifier or identifier in seen_ids:
                    continue

                seen_ids.add(identifier)
                raw_papers.append(work)

                if len(raw_papers) >= self.max_results:
                    break

            if len(raw_papers) >= self.max_results:
                break

            sleep(1)

        if self.config.executor.debug:
            raw_papers = raw_papers[:30]

        logger.info(f"Retrieved {len(raw_papers)} unique OpenAlex works.")
        return raw_papers
        
        if not is_journal_article(raw_paper):
            return None
    
    def convert_to_paper(self, raw_paper: dict[str, Any]) -> Paper | None:
        title = raw_paper.get("display_name") or ""
        abstract = restore_openalex_abstract(raw_paper.get("abstract_inverted_index"))

        if not title or not abstract:
            return None

        text_for_filter = f"{title}\n{abstract}"

        if self.required_keywords and not contains_any_keyword(
            text_for_filter,
            self.required_keywords,
        ):
            return None
            

        if self.exclude_keywords and contains_any_keyword(
            text_for_filter,
            self.exclude_keywords,
        ):
            return None

        journal = extract_journal(raw_paper)

        journal_tier = get_journal_tier(
            journal,
            self.tier1_journals,
            self.tier2_journals,
            self.tier3_journals,
        )
        
        if self.exclude_unknown_journals and journal_tier == "unknown":
            return None

        authors = extract_authors(raw_paper)
        affiliations = extract_institutions(raw_paper)
        topics = extract_topics(raw_paper)

        doi = raw_paper.get("doi")
        url = extract_landing_url(raw_paper)
        pdf_url = extract_pdf_url(raw_paper)

        publication_date = raw_paper.get("publication_date")

        if not publication_date and raw_paper.get("publication_year"):
            publication_date = str(raw_paper.get("publication_year"))

        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            full_text=None,
            affiliations=affiliations,
            journal=journal,
            doi=doi,
            publication_date=publication_date,
            publisher=extract_publisher(raw_paper),
            openalex_id=raw_paper.get("id"),
            topics=topics,
        )
