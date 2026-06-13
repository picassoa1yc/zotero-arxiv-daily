from .base import get_retriever_cls

# Import retrievers here so that @register_retriever decorators are executed.
from . import arxiv_retriever
from . import biorxiv_retriever
from . import medrxiv_retriever
from . import openalex_retriever
self.tier1_journals = [str(j) for j in _cfg_list(self.retriever_config, "tier1_journals")]
self.tier2_journals = [str(j) for j in _cfg_list(self.retriever_config, "tier2_journals")]
self.tier3_journals = [str(j) for j in _cfg_list(self.retriever_config, "tier3_journals")]
self.exclude_unknown_journals = bool(_cfg_get(self.retriever_config, "exclude_unknown_journals", True))
