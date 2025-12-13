"""
Service Layer - IndexingService, SearchService, EvaluationService, and ServicesContainer.
"""

from aci.services.container import ServicesContainer, create_services
from aci.services.evaluation_service import (
    EvaluationResult,
    EvaluationService,
)
from aci.services.indexing_service import (
    IndexingResult,
    IndexingService,
)
from aci.services.reranker import (
    OpenAICompatibleReranker,
    SimpleReranker,
)
from aci.services.repository_resolver import (
    RepositoryResolution,
    resolve_repository,
)
from aci.services.search_service import SearchService
from aci.services.search_types import RerankerInterface, SearchMode

__all__ = [
    # Container and factory
    "ServicesContainer",
    "create_services",
    # Repository resolution
    "RepositoryResolution",
    "resolve_repository",
    # Services
    "IndexingService",
    "IndexingResult",
    "SearchService",
    "SearchMode",
    "RerankerInterface",
    "OpenAICompatibleReranker",
    "SimpleReranker",
    "EvaluationService",
    "EvaluationResult",
]
