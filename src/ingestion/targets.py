"""Molecular targets relevant to the conditions we study.

These targets define the biology — they are NOT assumptions about what treats
what. We query ChEMBL for all compounds active against these targets,
then let the system discover compound→pathway→condition connections from evidence.

Sources for target selection:
  - SLE: known dysregulated pathways from GWAS, proteomics, literature
  - Muscle: established signalling nodes in hypertrophy/atrophy biology
"""

from dataclasses import dataclass, field


@dataclass
class MolecularTarget:
    gene_name: str              # HGNC gene symbol
    uniprot_id: str | None      # for cross-referencing with UniProt/Reactome
    description: str            # what this protein does
    biology: list[str]          # which biological domains this belongs to (documentation only)
    chembl_target_id: str | None = None  # filled in after ChEMBL lookup


TARGETS: list[MolecularTarget] = [

    # ── Innate immune signalling ──────────────────────────────────────────────

    MolecularTarget(
        gene_name="TLR7",
        uniprot_id="Q9NYK1",
        description="Toll-like receptor 7 — senses single-stranded RNA, activates innate immune response",
        biology=["innate_immunity", "sle"],
    ),
    MolecularTarget(
        gene_name="TLR9",
        uniprot_id="Q9NR96",
        description="Toll-like receptor 9 — senses unmethylated CpG DNA, key in type I interferon production",
        biology=["innate_immunity", "sle"],
    ),
    MolecularTarget(
        gene_name="IRAK4",
        uniprot_id="Q9NWZ3",
        description="Interleukin-1 receptor-associated kinase 4 — downstream of TLR/IL-1R signalling",
        biology=["innate_immunity", "nf_kb"],
    ),
    MolecularTarget(
        gene_name="MYD88",
        uniprot_id="Q99836",
        description="Myeloid differentiation primary response protein — adaptor for TLR/IL-1R → NF-kB",
        biology=["innate_immunity", "nf_kb"],
    ),

    # ── JAK-STAT pathway ─────────────────────────────────────────────────────

    MolecularTarget(
        gene_name="JAK1",
        uniprot_id="P23458",
        description="Janus kinase 1 — cytokine receptor signalling, type I/II interferon, IL-6",
        biology=["jak_stat", "sle", "inflammation"],
    ),
    MolecularTarget(
        gene_name="JAK2",
        uniprot_id="O60674",
        description="Janus kinase 2 — EPO, TPO, GM-CSF receptor signalling",
        biology=["jak_stat", "hematopoiesis"],
    ),
    MolecularTarget(
        gene_name="TYK2",
        uniprot_id="P29597",
        description="Tyrosine kinase 2 — IFN-α/β, IL-12, IL-23 signalling; SLE GWAS hit",
        biology=["jak_stat", "sle", "interferon"],
    ),
    MolecularTarget(
        gene_name="STAT3",
        uniprot_id="P40763",
        description="Signal transducer and activator of transcription 3 — IL-6, oncostatin M, leptin",
        biology=["jak_stat", "inflammation"],
    ),

    # ── B cell / antibody biology ─────────────────────────────────────────────

    MolecularTarget(
        gene_name="BTK",
        uniprot_id="Q06187",
        description="Bruton tyrosine kinase — B cell receptor signalling, B cell development",
        biology=["b_cell", "sle"],
    ),
    MolecularTarget(
        gene_name="TNFSF13B",
        uniprot_id="Q9Y275",
        description="BLyS/BAFF — B lymphocyte survival and maturation factor",
        biology=["b_cell", "sle"],
    ),
    MolecularTarget(
        gene_name="SYK",
        uniprot_id="P43405",
        description="Spleen tyrosine kinase — B/T cell receptor and Fc receptor signalling",
        biology=["b_cell", "t_cell", "sle"],
    ),

    # ── Complement system ─────────────────────────────────────────────────────

    MolecularTarget(
        gene_name="C3",
        uniprot_id="P01024",
        description="Complement component 3 — central hub of complement activation",
        biology=["complement", "sle"],
    ),
    MolecularTarget(
        gene_name="C5",
        uniprot_id="P01031",
        description="Complement component 5 — terminal complement pathway, membrane attack complex",
        biology=["complement", "sle"],
    ),

    # ── NF-kB / inflammatory cytokines ───────────────────────────────────────

    MolecularTarget(
        gene_name="NFKB1",
        uniprot_id="P19838",
        description="Nuclear factor kappa B — master regulator of inflammatory gene expression",
        biology=["nf_kb", "inflammation"],
    ),
    MolecularTarget(
        gene_name="IL6",
        uniprot_id="P05231",
        description="Interleukin-6 — pleiotropic cytokine, acute phase response, B cell differentiation",
        biology=["inflammation", "sle", "muscle"],
    ),
    MolecularTarget(
        gene_name="IL6R",
        uniprot_id="P08887",
        description="IL-6 receptor — transmits IL-6 signalling via JAK-STAT",
        biology=["inflammation", "sle"],
    ),
    MolecularTarget(
        gene_name="TNF",
        uniprot_id="P01375",
        description="Tumour necrosis factor alpha — pro-inflammatory cytokine, NF-kB activator",
        biology=["inflammation", "nf_kb"],
    ),
    MolecularTarget(
        gene_name="IFNAR1",
        uniprot_id="P17181",
        description="Interferon alpha/beta receptor 1 — type I interferon signalling, elevated in SLE",
        biology=["interferon", "sle"],
    ),

    # ── mTOR / protein synthesis (muscle and immune) ─────────────────────────

    MolecularTarget(
        gene_name="MTOR",
        uniprot_id="P42345",
        description="Mechanistic target of rapamycin — master regulator of cell growth, protein synthesis, autophagy",
        biology=["mtor", "muscle", "immunity"],
    ),
    MolecularTarget(
        gene_name="RPS6KB1",
        uniprot_id="P23443",
        description="S6 kinase 1 — mTORC1 substrate, ribosomal protein S6 phosphorylation, protein synthesis",
        biology=["mtor", "muscle"],
    ),
    MolecularTarget(
        gene_name="EIF4EBP1",
        uniprot_id="Q13541",
        description="4E-BP1 — mTORC1 substrate, translation initiation regulator",
        biology=["mtor", "muscle"],
    ),
    MolecularTarget(
        gene_name="PIK3CA",
        uniprot_id="P42336",
        description="PI3-kinase catalytic subunit alpha — upstream of AKT/mTOR",
        biology=["pi3k_akt", "muscle", "immunity"],
    ),
    MolecularTarget(
        gene_name="AKT1",
        uniprot_id="P31749",
        description="AKT serine/threonine kinase 1 — survival, growth, mTOR activation",
        biology=["pi3k_akt", "muscle"],
    ),
    MolecularTarget(
        gene_name="PRKAA1",
        uniprot_id="Q13131",
        description="AMPK alpha-1 — energy sensor, opposes mTOR, activates catabolic pathways",
        biology=["ampk", "muscle", "metabolism"],
    ),

    # ── Muscle-specific biology ───────────────────────────────────────────────

    MolecularTarget(
        gene_name="MSTN",
        uniprot_id="O14793",
        description="Myostatin (GDF-8) — TGF-β family member, negative regulator of muscle mass",
        biology=["muscle", "tgf_beta"],
    ),
    MolecularTarget(
        gene_name="IGF1R",
        uniprot_id="P08069",
        description="IGF-1 receptor — anabolic signalling → PI3K/AKT/mTOR axis",
        biology=["muscle", "pi3k_akt"],
    ),
    MolecularTarget(
        gene_name="ACVR2B",
        uniprot_id="Q13705",
        description="Activin receptor 2B — binds myostatin and activins, regulates muscle mass",
        biology=["muscle", "tgf_beta"],
    ),
    MolecularTarget(
        gene_name="FOXO3",
        uniprot_id="O43524",
        description="FOXO3 transcription factor — muscle atrophy (atrogin-1, MuRF1), inhibited by AKT",
        biology=["muscle", "autophagy"],
    ),
    MolecularTarget(
        gene_name="FBXO32",
        uniprot_id="Q969P5",
        description="Atrogin-1 (MAFbx) — E3 ubiquitin ligase, muscle-specific atrophy marker",
        biology=["muscle", "protein_degradation"],
    ),

    # ── Metabolism / nutrient sensing ────────────────────────────────────────

    MolecularTarget(
        gene_name="CKM",
        uniprot_id="P06732",
        description="Muscle creatine kinase — phosphocreatine ↔ ATP interconversion in muscle",
        biology=["muscle", "energy_metabolism"],
    ),
    MolecularTarget(
        gene_name="PPARGC1A",
        uniprot_id="Q9UBK2",
        description="PGC-1α — mitochondrial biogenesis, oxidative metabolism, anti-atrophy",
        biology=["muscle", "mitochondria", "metabolism"],
    ),
]


def get_targets_by_biology(domain: str) -> list[MolecularTarget]:
    return [t for t in TARGETS if domain in t.biology]


def get_all_gene_names() -> list[str]:
    return [t.gene_name for t in TARGETS]
