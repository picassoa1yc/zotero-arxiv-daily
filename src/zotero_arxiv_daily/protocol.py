from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import json

import tiktoken
from openai import OpenAI
from loguru import logger


RawPaperItem = TypeVar("RawPaperItem")


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str

    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    # Journal / metadata fields for OpenAlex and other journal sources
    journal: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[str] = None
    publisher: Optional[str] = None
    openalex_id: Optional[str] = None
    topics: Optional[list[str]] = None

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: dict) -> str:
        lang = llm_params.get("language", "English")
        prompt = (
            f"Given the following information of a paper, generate a one-sentence "
            f"TLDR summary in {lang}:\n\n"
        )

        if self.title:
            prompt += f"Title:\n{self.title}\n\n"

        if self.journal:
            prompt += f"Journal:\n{self.journal}\n\n"

        if self.publication_date:
            prompt += f"Publication date:\n{self.publication_date}\n\n"

        if self.abstract:
            prompt += f"Abstract:\n{self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n{self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided."

        # Use gpt-4o tokenizer for token estimation.
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant who perfectly summarizes scientific "
                        f"papers and gives the core idea of the paper to the user. "
                        f"Your answer should be in {lang}."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get("generation_kwargs", {}),
        )

        return response.choices[0].message.content

    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate TLDR of {self.url}: {e}")
            self.tldr = self.abstract
            return self.tldr

    def _generate_affiliations_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        # OpenAlex may already provide affiliations. Do not overwrite them.
        if self.affiliations:
            return self.affiliations

        # If no full text is available, there is nothing reliable for the LLM to extract.
        if self.full_text is None:
            return self.affiliations

        prompt = (
            "Given the beginning of a paper, extract the affiliations of the authors "
            "in a python list format, which is sorted by the author order. "
            "If there is no affiliation found, return an empty list '[]':\n\n"
            f"{self.full_text}"
        )

        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:2000]
        prompt = enc.decode(prompt_tokens)

        affiliations_response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant who perfectly extracts affiliations of "
                        "authors from a paper. You should return a python list of "
                        "affiliations sorted by the author order, like "
                        '["Tsinghua University", "Peking University"]. If an '
                        "affiliation consists of multi-level affiliations, like "
                        "'Department of Computer Science, Tsinghua University', "
                        "you should return the top-level affiliation "
                        "'Tsinghua University' only. Do not contain duplicated "
                        "affiliations. If there is no affiliation found, you should "
                        "return an empty list []. You should only return the final "
                        "list of affiliations, and do not return any intermediate results."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get("generation_kwargs", {}),
        )

        affiliations_text = affiliations_response.choices[0].message.content
        match = re.search(r"\[.*?\]", affiliations_text, flags=re.DOTALL)

        if match is None:
            return self.affiliations

        affiliations = json.loads(match.group(0))
        affiliations = list(set(affiliations))
        affiliations = [str(a) for a in affiliations]

        return affiliations

    def generate_affiliations(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client, llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            return self.affiliations


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
