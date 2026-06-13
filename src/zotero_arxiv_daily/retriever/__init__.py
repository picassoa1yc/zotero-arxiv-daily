from .base import get_retriever_cls

# Import retrievers here so that @register_retriever decorators are executed.
from . import arxiv_retriever
from . import biorxiv_retriever
from . import medrxiv_retriever
from . import openalex_retriever
