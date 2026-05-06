"""Topic definitions for targeted PubMed ingestion.

Domains:
  - Autoimmune / SLE
  - Muscle physiology (hypertrophy, protein synthesis)
"""

from dataclasses import dataclass, field


@dataclass
class IngestionTopic:
    name: str
    mesh_query: str
    description: str
    priority: int = 1  # 1=highest


TOPICS: list[IngestionTopic] = [
    IngestionTopic(
        name="sle_core",
        mesh_query='"Lupus Erythematosus, Systemic"[MeSH]',
        description="Systemic Lupus Erythematosus — core literature",
        priority=1,
    ),
    IngestionTopic(
        name="lupus_nephritis",
        mesh_query='"Lupus Nephritis"[MeSH]',
        description="Lupus nephritis",
        priority=1,
    ),
    IngestionTopic(
        name="cutaneous_lupus",
        mesh_query='"Lupus Erythematosus, Cutaneous"[MeSH]',
        description="Cutaneous lupus erythematosus",
        priority=2,
    ),
    IngestionTopic(
        name="antiphospholipid",
        mesh_query='"Antiphospholipid Syndrome"[MeSH]',
        description="Antiphospholipid syndrome (commonly co-occurs with SLE)",
        priority=2,
    ),
    IngestionTopic(
        name="sjogrens",
        mesh_query='"Sjogren\'s Syndrome"[MeSH]',
        description="Sjögren's syndrome",
        priority=2,
    ),
    IngestionTopic(
        name="complement_system",
        mesh_query='"Complement System Proteins"[MeSH] AND ("lupus"[Title/Abstract] OR "autoimmune"[Title/Abstract])',
        description="Complement system in autoimmune disease",
        priority=2,
    ),
    IngestionTopic(
        name="autoimmune_broad",
        mesh_query='"Autoimmune Diseases"[MeSH] AND ("lupus"[Title/Abstract] OR "SLE"[Title/Abstract])',
        description="Broad autoimmune disease literature mentioning lupus/SLE",
        priority=3,
    ),

    # ── Muscle physiology ─────────────────────────────────────────────────────

    IngestionTopic(
        name="muscle_hypertrophy",
        mesh_query='"Muscle Development"[MeSH] OR "Hypertrophy"[MeSH] AND "Muscle, Skeletal"[MeSH]',
        description="Skeletal muscle hypertrophy — growth, adaptation, signalling",
        priority=1,
    ),
    IngestionTopic(
        name="protein_synthesis",
        mesh_query='"Protein Biosynthesis"[MeSH] AND ("muscle"[Title/Abstract] OR "skeletal"[Title/Abstract] OR "exercise"[Title/Abstract])',
        description="Muscle protein synthesis — mTOR, ribosomal biogenesis, amino acid signalling",
        priority=1,
    ),
    IngestionTopic(
        name="resistance_training",
        mesh_query='"Resistance Training"[MeSH] AND ("hypertrophy"[Title/Abstract] OR "protein synthesis"[Title/Abstract] OR "muscle mass"[Title/Abstract])',
        description="Resistance training effects on muscle mass and protein turnover",
        priority=2,
    ),
    IngestionTopic(
        name="mtor_signaling",
        mesh_query='"TOR Serine-Threonine Kinases"[MeSH] AND ("muscle"[Title/Abstract] OR "hypertrophy"[Title/Abstract] OR "anabolism"[Title/Abstract])',
        description="mTORC1 pathway in muscle anabolism",
        priority=2,
    ),
    IngestionTopic(
        name="amino_acid_muscle",
        mesh_query='"Amino Acids"[MeSH] AND "Muscle, Skeletal"[MeSH] AND ("protein synthesis"[Title/Abstract] OR "anabolic"[Title/Abstract])',
        description="Amino acid regulation of muscle protein synthesis (leucine, EAAs, whey)",
        priority=2,
    ),
]


def get_topics_by_priority(max_priority: int = 1) -> list[IngestionTopic]:
    return [t for t in TOPICS if t.priority <= max_priority]


def get_topic_by_name(name: str) -> IngestionTopic | None:
    return next((t for t in TOPICS if t.name == name), None)
