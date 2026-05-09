from __future__ import annotations

"""Topic definitions for targeted PubMed ingestion.

Design principle: first-principles, mechanism-first. Topics are organized into
three independent layers so the system discovers relationships from evidence
rather than encoding assumptions (e.g. "creatine improves muscle"):

  Layer 1 — Disease / condition mechanisms (what is broken and how)
  Layer 2 — Compound / intervention pharmacology (what does X do, independent of disease)
  Layer 3 — Clinical evidence (what happened when X was tried in condition Y — broadly)

Domains:
  - Autoimmune / SLE
  - Muscle physiology (hypertrophy, protein synthesis)
  - Pharmacology & interventions (mechanism-only, no assumed efficacy)
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

    # ── Layer 1: Disease / condition mechanisms ───────────────────────────────
    # What is the condition at a molecular and cellular level.
    # No treatment assumptions — pure mechanism and pathophysiology.

    IngestionTopic(
        name="sle_pathophysiology",
        mesh_query='"Lupus Erythematosus, Systemic"[MeSH] AND ("pathophysiology"[Subheading] OR "immunology"[Subheading] OR "etiology"[Subheading])',
        description="SLE pathophysiology — autoantibodies, immune dysregulation, complement, interferon",
        priority=1,
    ),
    IngestionTopic(
        name="inflammatory_pathways",
        mesh_query='"NF-kappa B"[MeSH] OR "JAK-STAT Signaling Pathway"[MeSH] OR "Interleukin-6"[MeSH] OR "Tumor Necrosis Factor-alpha"[MeSH] OR "Interferons"[MeSH]',
        description="Core inflammatory signalling pathways — NF-kB, JAK-STAT, IL-6, TNF-α, interferons",
        priority=1,
    ),
    IngestionTopic(
        name="autoimmunity_mechanisms",
        mesh_query='"Autoimmunity"[MeSH] AND ("B-Lymphocytes"[MeSH] OR "T-Lymphocytes"[MeSH] OR "Dendritic Cells"[MeSH] OR "Complement System Proteins"[MeSH])',
        description="Cellular mechanisms of autoimmunity — B/T cell dysregulation, dendritic cells, complement",
        priority=1,
    ),
    IngestionTopic(
        name="muscle_hypertrophy",
        mesh_query='"Muscle Development"[MeSH] OR ("Hypertrophy"[MeSH] AND "Muscle, Skeletal"[MeSH])',
        description="Skeletal muscle hypertrophy mechanisms — what happens at cellular and molecular level",
        priority=1,
    ),
    IngestionTopic(
        name="muscle_cell_biology",
        mesh_query='"Satellite Cells, Skeletal Muscle"[MeSH] OR "Myoblasts"[MeSH] OR ("Muscle Fibers, Skeletal"[MeSH] AND ("cell proliferation"[MeSH] OR "cell differentiation"[MeSH]))',
        description="Muscle cell biology — satellite cells, myoblast differentiation, fiber-type composition",
        priority=1,
    ),
    IngestionTopic(
        name="protein_synthesis_mechanisms",
        mesh_query='"Protein Biosynthesis"[MeSH] AND "Muscle, Skeletal"[MeSH]',
        description="Muscle protein synthesis mechanisms — ribosomal biogenesis, translation, turnover",
        priority=1,
    ),
    IngestionTopic(
        name="protein_degradation_mechanisms",
        mesh_query='("Proteolysis"[MeSH] OR "Ubiquitin-Proteasome System"[MeSH] OR "Autophagy"[MeSH]) AND "Muscle, Skeletal"[MeSH]',
        description="Protein degradation in muscle — proteasome, autophagy, atrophy pathways",
        priority=1,
    ),
    IngestionTopic(
        name="mtor_signaling",
        mesh_query='"TOR Serine-Threonine Kinases"[MeSH]',
        description="mTOR signalling — full mechanistic literature, not limited to any disease or outcome",
        priority=1,
    ),

    # ── Layer 2: Compound pharmacology (mechanism-only, no disease assumption) ─
    # What does a compound do biochemically — fetched independently of any
    # disease or expected outcome. The system discovers connections, not us.

    IngestionTopic(
        name="pharmacology_autoimmune_drugs",
        mesh_query='"Hydroxychloroquine"[MeSH] OR "Methotrexate"[MeSH] OR "Mycophenolic Acid"[MeSH] OR "Azathioprine"[MeSH] OR "Cyclophosphamide"[MeSH] OR "Belimumab"[MeSH]',
        description="Pharmacology of compounds studied in autoimmune conditions — mechanism, kinetics, effects",
        priority=2,
    ),
    IngestionTopic(
        name="pharmacology_biologics",
        mesh_query='"Janus Kinase Inhibitors"[MeSH] OR "Interleukin Inhibitors"[MeSH] OR "Tumor Necrosis Factor Inhibitors"[MeSH] OR ("Antibodies, Monoclonal"[MeSH] AND "Immunology"[MeSH])',
        description="Biologic and small-molecule immunomodulator classes — mechanism, immune effects",
        priority=2,
    ),
    IngestionTopic(
        name="pharmacology_amino_acids",
        mesh_query='"Leucine"[MeSH] OR "Essential Amino Acids"[MeSH] OR "Glutamine"[MeSH] OR "Arginine"[MeSH]',
        description="Amino acid biochemistry and signalling — metabolic roles independent of any disease or outcome",
        priority=2,
    ),
    IngestionTopic(
        name="pharmacology_nutritional_compounds",
        mesh_query='"Creatine"[MeSH] OR "Carnitine"[MeSH] OR "Vitamin D"[MeSH] OR "Fatty Acids, Omega-3"[MeSH] OR "Magnesium"[MeSH] OR "Zinc"[MeSH] OR "Coenzyme A"[MeSH]',
        description="Biochemistry of nutritional compounds — what they do, not what they treat",
        priority=2,
    ),
    IngestionTopic(
        name="pharmacology_hormones",
        mesh_query='"Insulin-Like Growth Factor I"[MeSH] OR "Growth Hormone"[MeSH] OR "Testosterone"[MeSH] OR "Cortisol"[MeSH] OR "Estrogens"[MeSH] OR "Insulin"[MeSH]',
        description="Hormone mechanisms — tissue growth, immune modulation, metabolism",
        priority=2,
    ),

    # ── Layer 3: Clinical evidence (broad — what was tried, not what works) ───
    # All interventional studies in these conditions without pre-filtering by compound.
    # The system reasons over what the evidence shows, not what we assume.

    IngestionTopic(
        name="clinical_trials_sle",
        mesh_query='"Lupus Erythematosus, Systemic"[MeSH] AND ("Randomized Controlled Trial"[Publication Type] OR "Clinical Trial"[Publication Type])',
        description="All clinical trials in SLE — any intervention, any outcome",
        priority=1,
    ),
    IngestionTopic(
        name="clinical_trials_muscle",
        mesh_query='("Muscle, Skeletal"[MeSH] OR "Muscle Strength"[MeSH] OR "Muscle Development"[MeSH]) AND ("Randomized Controlled Trial"[Publication Type] OR "Clinical Trial"[Publication Type])',
        description="All clinical trials with skeletal muscle outcomes — any intervention",
        priority=1,
    ),
    IngestionTopic(
        name="systematic_reviews_autoimmune",
        mesh_query='"Autoimmune Diseases"[MeSH] AND ("Meta-Analysis"[Publication Type] OR "Systematic Review"[Publication Type])',
        description="Meta-analyses and systematic reviews across autoimmune conditions",
        priority=1,
    ),
    IngestionTopic(
        name="systematic_reviews_muscle",
        mesh_query='("Muscle, Skeletal"[MeSH] OR "Resistance Training"[MeSH]) AND ("Meta-Analysis"[Publication Type] OR "Systematic Review"[Publication Type])',
        description="Meta-analyses and systematic reviews on muscle physiology and exercise",
        priority=1,
    ),
    IngestionTopic(
        name="biomarkers_sle",
        mesh_query='"Biomarkers"[MeSH] AND "Lupus Erythematosus, Systemic"[MeSH]',
        description="SLE biomarkers — disease activity, flare prediction, treatment response",
        priority=2,
    ),
    IngestionTopic(
        name="biomarkers_muscle",
        mesh_query='"Biomarkers"[MeSH] AND ("Muscle, Skeletal"[MeSH] OR "Muscle Development"[MeSH])',
        description="Muscle biomarkers — anabolic/catabolic state, recovery, performance",
        priority=2,
    ),
    IngestionTopic(
        name="resistance_training",
        mesh_query='"Resistance Training"[MeSH]',
        description="All resistance training literature — mechanisms, adaptations, outcomes, any population",
        priority=2,
    ),

    # ── Ichthyosis (genetic skin barrier disorders) ─────────────────────────
    # Heterogeneous group of inherited cornification disorders. Worth keeping
    # broad — there are many subtypes (lamellar, harlequin, X-linked, EHK, etc.)
    # each with its own genetics and management literature.

    IngestionTopic(
        name="ichthyosis_core",
        mesh_query='"Ichthyosis"[MeSH]',
        description="Ichthyosis — broad MeSH term covering inherited cornification disorders",
        priority=1,
    ),
    IngestionTopic(
        name="ichthyosis_lamellar",
        mesh_query='"Ichthyosis, Lamellar"[MeSH]',
        description="Lamellar ichthyosis — TGM1 and related autosomal recessive forms",
        priority=1,
    ),
    IngestionTopic(
        name="ichthyosis_x_linked",
        mesh_query='"Ichthyosis, X-Linked"[MeSH]',
        description="X-linked ichthyosis — STS deficiency, steroid sulfatase",
        priority=1,
    ),
    IngestionTopic(
        name="ichthyosis_harlequin",
        mesh_query='"Hyperkeratosis, Epidermolytic"[MeSH] OR "harlequin ichthyosis"[Title/Abstract] OR "ABCA12"[Title/Abstract]',
        description="Severe forms — harlequin and epidermolytic hyperkeratosis (ABCA12, KRT1/KRT10)",
        priority=2,
    ),
    IngestionTopic(
        name="ichthyosis_genetics",
        mesh_query='"Ichthyosis"[MeSH] AND ("Genetics"[Subheading] OR "genetics"[MeSH Subheading])',
        description="Genetic basis of ichthyosis — gene panels, inheritance, molecular diagnostics",
        priority=1,
    ),
    IngestionTopic(
        name="ichthyosis_treatment",
        mesh_query='"Ichthyosis"[MeSH] AND ("therapy"[Subheading] OR "drug therapy"[Subheading] OR "Randomized Controlled Trial"[Publication Type] OR "Clinical Trial"[Publication Type])',
        description="Ichthyosis treatments — retinoids, emollients, gene therapy, clinical trials",
        priority=1,
    ),
    # ── Creatine (mechanism + clinical) ─────────────────────────────────────
    # Pulled out of the broader pharmacology_nutritional_compounds bucket so
    # the corpus has dedicated coverage. Two angles:
    #   1. Biochemistry / pharmacology — what creatine does at the cellular
    #      level (PCr buffer, mTOR signalling, satellite-cell biology).
    #   2. Clinical evidence — RCTs and human trials across populations
    #      (athletic, sarcopenia, neuro, cardiac).

    IngestionTopic(
        name="creatine_pharmacology",
        mesh_query='"Creatine"[MeSH] OR "Phosphocreatine"[MeSH] OR "Creatine Kinase"[MeSH]',
        description="Creatine pharmacology and biochemistry — phosphocreatine, creatine kinase, energetics",
        priority=1,
    ),
    IngestionTopic(
        name="creatine_clinical",
        mesh_query='"Creatine"[MeSH] AND ("Randomized Controlled Trial"[Publication Type] OR "Clinical Trial"[Publication Type] OR "Meta-Analysis"[Publication Type] OR "Systematic Review"[Publication Type])',
        description="Creatine in human studies — RCTs, meta-analyses, systematic reviews",
        priority=1,
    ),

    IngestionTopic(
        name="skin_barrier_function",
        mesh_query='("Filaggrin Proteins"[MeSH] OR "Loricrin"[Title/Abstract] OR "Transglutaminases"[MeSH] OR "Stratum Corneum"[Title/Abstract]) AND ("skin"[MeSH] OR "epidermis"[MeSH])',
        description="Skin-barrier biology — filaggrin, loricrin, transglutaminase, stratum corneum (mechanism layer for ichthyosis)",
        priority=2,
    ),
]


def get_topics_by_priority(max_priority: int = 1) -> list[IngestionTopic]:
    return [t for t in TOPICS if t.priority <= max_priority]


def get_topic_by_name(name: str) -> IngestionTopic | None:
    return next((t for t in TOPICS if t.name == name), None)
