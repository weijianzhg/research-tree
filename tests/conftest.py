from __future__ import annotations

import pytest

from research_tree.render import write_overview
from research_tree.store import GraphStore


@pytest.fixture
def store(tmp_path):
    graph, _ = GraphStore.create(
        tmp_path / "research",
        "What makes sparse mixture-of-experts models efficient?",
        title="Mixture-of-experts",
    )
    write_overview(graph)
    return graph
