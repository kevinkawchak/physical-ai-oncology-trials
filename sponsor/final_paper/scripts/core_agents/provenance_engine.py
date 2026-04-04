"""Provenance engine: W3C PROV-DM provenance model implementation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class PROVType(Enum):
    ENTITY = "prov:Entity"
    ACTIVITY = "prov:Activity"
    AGENT = "prov:Agent"


class PROVRelation(Enum):
    WAS_GENERATED_BY = "prov:wasGeneratedBy"
    USED = "prov:used"
    WAS_DERIVED_FROM = "prov:wasDerivedFrom"
    WAS_ATTRIBUTED_TO = "prov:wasAttributedTo"
    WAS_ASSOCIATED_WITH = "prov:wasAssociatedWith"
    ACTED_ON_BEHALF_OF = "prov:actedOnBehalfOf"
    WAS_INFORMED_BY = "prov:wasInformedBy"


@dataclass
class PROVNode:
    node_id: str
    prov_type: PROVType
    label: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PROVEdge:
    source_id: str
    target_id: str
    relation: PROVRelation
    timestamp: datetime = field(default_factory=datetime.utcnow)
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class ProvenanceBundle:
    bundle_id: str
    nodes: list[PROVNode] = field(default_factory=list)
    edges: list[PROVEdge] = field(default_factory=list)
    sealed: bool = False
    seal_hash: str = ""


class ProvenanceEngine:
    """Implements W3C PROV-DM provenance tracking for trial artifacts."""

    def __init__(self) -> None:
        self.bundles: dict[str, ProvenanceBundle] = {}

    def create_bundle(self, bundle_id: str) -> ProvenanceBundle:
        bundle = ProvenanceBundle(bundle_id=bundle_id)
        self.bundles[bundle_id] = bundle
        return bundle

    def add_entity(
        self, bundle_id: str, entity_id: str, label: str, attributes: dict[str, str] | None = None
    ) -> PROVNode:
        node = PROVNode(node_id=entity_id, prov_type=PROVType.ENTITY, label=label, attributes=attributes or {})
        self.bundles[bundle_id].nodes.append(node)
        return node

    def add_activity(
        self, bundle_id: str, activity_id: str, label: str, attributes: dict[str, str] | None = None
    ) -> PROVNode:
        node = PROVNode(node_id=activity_id, prov_type=PROVType.ACTIVITY, label=label, attributes=attributes or {})
        self.bundles[bundle_id].nodes.append(node)
        return node

    def add_agent(
        self, bundle_id: str, agent_id: str, label: str, attributes: dict[str, str] | None = None
    ) -> PROVNode:
        node = PROVNode(node_id=agent_id, prov_type=PROVType.AGENT, label=label, attributes=attributes or {})
        self.bundles[bundle_id].nodes.append(node)
        return node

    def add_relation(self, bundle_id: str, source: str, target: str, relation: PROVRelation) -> PROVEdge:
        edge = PROVEdge(source_id=source, target_id=target, relation=relation)
        self.bundles[bundle_id].edges.append(edge)
        return edge

    def seal_bundle(self, bundle_id: str) -> str:
        bundle = self.bundles[bundle_id]
        content = json.dumps(
            {
                "nodes": [{"id": n.node_id, "type": n.prov_type.value} for n in bundle.nodes],
                "edges": [{"src": e.source_id, "tgt": e.target_id, "rel": e.relation.value} for e in bundle.edges],
            },
            sort_keys=True,
        )
        bundle.seal_hash = hashlib.sha256(content.encode()).hexdigest()[:24]
        bundle.sealed = True
        return bundle.seal_hash

    def export_provn(self, bundle_id: str) -> str:
        bundle = self.bundles[bundle_id]
        lines = [f"bundle {bundle_id}"]
        for node in bundle.nodes:
            lines.append(f"  {node.prov_type.value.split(':')[1].lower()}({node.node_id}, [{node.label}])")
        for edge in bundle.edges:
            rel_name = edge.relation.value.split(":")[1]
            lines.append(f"  {rel_name}({edge.source_id}, {edge.target_id})")
        lines.append("endBundle")
        return "\n".join(lines)

    def lineage(self, bundle_id: str, entity_id: str) -> list[str]:
        bundle = self.bundles[bundle_id]
        visited: set[str] = set()
        queue = [entity_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            for edge in bundle.edges:
                if edge.source_id == current:
                    queue.append(edge.target_id)
        return sorted(visited)

    def summary(self) -> dict[str, object]:
        return {
            "bundles": len(self.bundles),
            "total_nodes": sum(len(b.nodes) for b in self.bundles.values()),
            "total_edges": sum(len(b.edges) for b in self.bundles.values()),
            "sealed": sum(1 for b in self.bundles.values() if b.sealed),
        }


def main() -> None:
    engine = ProvenanceEngine()
    engine.create_bundle("protocol-gen")
    engine.add_entity("protocol-gen", "proto:v1", "Protocol v1.0", {"version": "1.0"})
    engine.add_entity("protocol-gen", "proto:v2", "Protocol v2.0", {"version": "2.0"})
    engine.add_activity("protocol-gen", "act:amend", "Protocol Amendment")
    engine.add_agent("protocol-gen", "agent:medical_writer", "Medical Writer")
    engine.add_relation("protocol-gen", "proto:v2", "act:amend", PROVRelation.WAS_GENERATED_BY)
    engine.add_relation("protocol-gen", "act:amend", "proto:v1", PROVRelation.USED)
    engine.add_relation("protocol-gen", "act:amend", "agent:medical_writer", PROVRelation.WAS_ASSOCIATED_WITH)
    engine.add_relation("protocol-gen", "proto:v2", "proto:v1", PROVRelation.WAS_DERIVED_FROM)
    seal = engine.seal_bundle("protocol-gen")
    print(f"PROV-N:\n{engine.export_provn('protocol-gen')}")
    print(f"Seal: {seal}")
    print(f"Lineage of proto:v2: {engine.lineage('protocol-gen', 'proto:v2')}")
    print(f"Summary: {engine.summary()}")


if __name__ == "__main__":
    main()
