"""
Headless integration test skeleton for the Globule orchestration engine.

This test will eventually verify the end-to-end processing of a Globule
through the decoupled engine, using dummy or mocked providers initially.
"""
import pytest
from uuid import uuid4

from globule.core.models import GlobuleV1, ProcessedGlobuleV1
from globule.core.interfaces import IOrchestrationEngine, IParserProvider, IEmbeddingProvider, IStorageManager
from globule.core.errors import ParserError, EmbeddingError, StorageError

# Dummy implementations for the integration test skeleton
class NoOpParser(IParserProvider):
    def parse(self, text: str) -> dict:
        return {"status": "parsed_by_noop", "original_length": len(text)}

class NoOpEmbedder(IEmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        return [0.0] * 10  # Return a dummy embedding

class InMemoryStorage(IStorageManager):
    _store = {}

    def save(self, globule: ProcessedGlobuleV1) -> None:
        self._store[globule.globule_id] = globule

    def get(self, globule_id: UUID) -> ProcessedGlobuleV1:
        if globule_id not in self._store:
            raise StorageError(f"Globule {globule_id} not found in in-memory store.")
        return self._store[globule_id]

class NoOpOrchestrationEngine(IOrchestrationEngine):
    def __init__(
        self,
        parser: IParserProvider,
        embedder: IEmbeddingProvider,
        storage: IStorageManager
    ):
        self.parser = parser
        self.embedder = embedder
        self.storage = storage

    def process(self, globule: GlobuleV1) -> ProcessedGlobuleV1:
        # Simulate processing steps
        parsed_data = self.parser.parse(globule.raw_text)
        embedding = self.embedder.embed(globule.raw_text)

        processed_globule = ProcessedGlobuleV1(
            globule_id=globule.globule_id,
            original_globule=globule,
            embedding=embedding,
            parsed_data=parsed_data,
            processing_time_ms=1.0, # Dummy time
            provider_metadata={
                "parser": "noop",
                "embedder": "noop",
                "storage": "in_memory"
            }
        )
        self.storage.save(processed_globule)
        return processed_globule

@pytest.fixture
def noop_engine():
    parser = NoOpParser()
    embedder = NoOpEmbedder()
    storage = InMemoryStorage()
    return NoOpOrchestrationEngine(parser, embedder, storage)

def test_headless_processing_flow(noop_engine: IOrchestrationEngine):
    """Tests the basic end-to-end processing flow of a Globule through the headless engine."""
    raw_globule = GlobuleV1(raw_text="This is a test sentence.", source="cli")
    
    processed_globule = noop_engine.process(raw_globule)
    
    assert isinstance(processed_globule, ProcessedGlobuleV1)
    assert processed_globule.globule_id == raw_globule.globule_id
    assert processed_globule.original_globule == raw_globule
    assert processed_globule.parsed_data == {"status": "parsed_by_noop", "original_length": len(raw_globule.raw_text)}
    assert processed_globule.embedding == [0.0] * 10
    assert processed_globule.provider_metadata["parser"] == "noop"

    # Verify it was saved
    retrieved_globule = noop_engine.storage.get(raw_globule.globule_id)
    assert retrieved_globule == processed_globule
