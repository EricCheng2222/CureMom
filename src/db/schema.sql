-- CureMom medical literature knowledge base schema
-- PostgreSQL 16 + pgvector

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ─── Journals ───────────────────────────────────────────────────────────────

CREATE TABLE journals (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    abbreviation VARCHAR(255),
    issn        VARCHAR(20),
    eissn       VARCHAR(20),
    nlm_id      VARCHAR(20),
    UNIQUE (nlm_id),
    UNIQUE (issn)
);

-- ─── Papers (core) ──────────────────────────────────────────────────────────

CREATE TABLE papers (
    id                  BIGSERIAL PRIMARY KEY,
    pmid                VARCHAR(20) UNIQUE NOT NULL,
    pmcid               VARCHAR(20),
    doi                 VARCHAR(255),
    title               TEXT NOT NULL,
    abstract            TEXT,
    abstract_json       JSONB,          -- {background, methods, results, conclusions, ...}
    pub_year            SMALLINT,
    pub_date            DATE,
    journal_id          INT REFERENCES journals(id),
    publication_types   TEXT[],         -- ['Randomized Controlled Trial', ...]
    language            CHAR(3) DEFAULT 'eng',
    has_full_text       BOOLEAN DEFAULT FALSE,
    license             VARCHAR(100),   -- 'CC BY', 'CC BY-NC', etc.
    grant_agencies      TEXT[],
    ingested_at         TIMESTAMPTZ DEFAULT NOW(),
    last_updated        TIMESTAMPTZ DEFAULT NOW(),
    -- PostgreSQL FTS (fallback search)
    search_vector       TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
    ) STORED
);

CREATE INDEX idx_papers_pmid ON papers (pmid);
CREATE INDEX idx_papers_pmcid ON papers (pmcid) WHERE pmcid IS NOT NULL;
CREATE INDEX idx_papers_pub_year ON papers (pub_year);
CREATE INDEX idx_papers_pub_types ON papers USING GIN (publication_types);
CREATE INDEX idx_papers_search_vector ON papers USING GIN (search_vector);

-- ─── Authors ────────────────────────────────────────────────────────────────

CREATE TABLE authors (
    id          BIGSERIAL PRIMARY KEY,
    last_name   TEXT,
    fore_name   TEXT,
    initials    VARCHAR(20),
    orcid       VARCHAR(30),
    UNIQUE (last_name, fore_name, orcid)
);

CREATE TABLE paper_authors (
    paper_id            BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    author_id           BIGINT NOT NULL REFERENCES authors(id),
    position            SMALLINT NOT NULL,
    is_corresponding    BOOLEAN DEFAULT FALSE,
    affiliations        TEXT[],
    PRIMARY KEY (paper_id, author_id)
);

CREATE INDEX idx_paper_authors_author_id ON paper_authors (author_id);

-- ─── MeSH ───────────────────────────────────────────────────────────────────

CREATE TABLE mesh_terms (
    id              SERIAL PRIMARY KEY,
    descriptor_ui   VARCHAR(20) UNIQUE NOT NULL,   -- e.g. D002318
    descriptor_name TEXT NOT NULL,
    tree_numbers    TEXT[],                         -- e.g. {'C14.280.647'}
    synonyms        TEXT[]
);

CREATE INDEX idx_mesh_descriptor_ui ON mesh_terms (descriptor_ui);
CREATE INDEX idx_mesh_tree_numbers ON mesh_terms USING GIN (tree_numbers);

CREATE TABLE paper_mesh (
    paper_id        BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    mesh_id         INT NOT NULL REFERENCES mesh_terms(id),
    qualifier_name  TEXT NOT NULL DEFAULT '',
    is_major_topic  BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (paper_id, mesh_id, qualifier_name)
);

CREATE INDEX idx_paper_mesh_mesh_id ON paper_mesh (mesh_id);

-- ─── Citations (reference graph) ────────────────────────────────────────────

CREATE TABLE citations (
    citing_paper_id BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    cited_paper_id  BIGINT REFERENCES papers(id) ON DELETE SET NULL,
    cited_pmid_raw  VARCHAR(20) NOT NULL,
    PRIMARY KEY (citing_paper_id, cited_pmid_raw)
);

CREATE INDEX idx_citations_cited_paper ON citations (cited_paper_id);
CREATE INDEX idx_citations_cited_pmid_raw ON citations (cited_pmid_raw);

-- ─── Full-text sections ─────────────────────────────────────────────────────

CREATE TABLE paper_sections (
    id              BIGSERIAL PRIMARY KEY,
    paper_id        BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_type    TEXT NOT NULL,  -- 'abstract', 'introduction', 'methods', 'results', 'discussion', 'conclusion', 'other'
    section_order   SMALLINT,
    title           TEXT,
    content         TEXT NOT NULL
);

CREATE INDEX idx_paper_sections_paper_id ON paper_sections (paper_id);
CREATE INDEX idx_paper_sections_type ON paper_sections (section_type);

-- ─── Chunks for vector search ───────────────────────────────────────────────

CREATE TABLE paper_chunks (
    id              BIGSERIAL PRIMARY KEY,
    paper_id        BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    section_id      BIGINT REFERENCES paper_sections(id) ON DELETE SET NULL,
    chunk_index     INT NOT NULL,
    chunk_text      TEXT NOT NULL,
    source_type     TEXT NOT NULL,  -- 'abstract', 'introduction', 'methods', 'results', 'discussion'
    start_char      INT,
    end_char        INT,
    paragraph_index INT,
    token_count     INT,
    embedding       VECTOR(768),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_chunks_paper_id ON paper_chunks (paper_id);
CREATE INDEX idx_chunks_source_type ON paper_chunks (source_type);
-- HNSW index created after initial bulk load for performance:
-- CREATE INDEX idx_chunks_embedding ON paper_chunks
--     USING hnsw (embedding vector_cosine_ops)
--     WITH (m = 16, ef_construction = 64);

-- ─── Named entities (Phase 2) ───────────────────────────────────────────────

CREATE TABLE paper_entities (
    id              BIGSERIAL PRIMARY KEY,
    paper_id        BIGINT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_id        BIGINT REFERENCES paper_chunks(id) ON DELETE SET NULL,
    entity_text     TEXT NOT NULL,
    entity_type     TEXT NOT NULL,  -- DISEASE, CHEMICAL, GENE_OR_GENE_PRODUCT, ORGANISM, CELL_TYPE
    start_char      INT,
    end_char        INT,
    kb_id           TEXT            -- UMLS CUI or MeSH descriptor_ui
);

CREATE INDEX idx_entities_paper_id ON paper_entities (paper_id);
CREATE INDEX idx_entities_type_text ON paper_entities (entity_type, entity_text);
CREATE INDEX idx_entities_kb_id ON paper_entities (kb_id) WHERE kb_id IS NOT NULL;

-- ─── ChEMBL compound-target layer ──────────────────────────────────────────
-- Compounds discovered via target biology, not pre-assumed disease relevance.

CREATE TABLE compounds (
    id                  BIGSERIAL PRIMARY KEY,
    chembl_id           VARCHAR(20) UNIQUE NOT NULL,
    name                TEXT,
    synonyms            TEXT[],
    max_phase           SMALLINT,       -- 0=preclinical, 4=approved drug
    molecule_type       TEXT,           -- Small molecule, Protein, Antibody, etc.
    mechanism_of_action TEXT,
    molecular_formula   TEXT,
    molecular_weight    NUMERIC(10,3),
    inchi_key           TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_compounds_chembl_id ON compounds (chembl_id);
CREATE INDEX idx_compounds_name ON compounds USING GIN (to_tsvector('english', coalesce(name, '')));

CREATE TABLE molecular_targets (
    id              SERIAL PRIMARY KEY,
    chembl_target_id VARCHAR(20) UNIQUE NOT NULL,
    gene_name       TEXT NOT NULL,
    pref_name       TEXT,
    target_type     TEXT,
    organism        TEXT DEFAULT 'Homo sapiens',
    biology         TEXT[]          -- which biological domains (documentation)
);

CREATE INDEX idx_targets_gene ON molecular_targets (gene_name);

CREATE TABLE compound_targets (
    id              BIGSERIAL PRIMARY KEY,
    compound_id     BIGINT NOT NULL REFERENCES compounds(id) ON DELETE CASCADE,
    target_id       INT NOT NULL REFERENCES molecular_targets(id),
    action_type     TEXT,           -- INHIBITOR, AGONIST, ANTAGONIST, MODULATOR, etc.
    activity_type   TEXT,           -- IC50, Ki, Kd, EC50, etc.
    activity_value  NUMERIC,        -- in nM
    assay_type      TEXT,           -- B=binding, F=functional
    confidence_score SMALLINT,      -- ChEMBL 0-9
    document_year   INT,
    UNIQUE (compound_id, target_id, activity_type)
);

CREATE INDEX idx_compound_targets_compound ON compound_targets (compound_id);
CREATE INDEX idx_compound_targets_target ON compound_targets (target_id);

-- Links compounds discovered via ChEMBL to PubMed papers fetched for them
CREATE TABLE compound_papers (
    compound_id     BIGINT NOT NULL REFERENCES compounds(id) ON DELETE CASCADE,
    pmid            VARCHAR(20) NOT NULL,
    PRIMARY KEY (compound_id, pmid)
);

-- ─── Entity co-occurrence graph (Phase 4: HippoRAG) ─────────────────────────

CREATE TABLE entity_graph (
    entity_a        TEXT NOT NULL,
    entity_b        TEXT NOT NULL,
    co_occurrences  INT NOT NULL DEFAULT 1,
    paper_ids       BIGINT[],
    PRIMARY KEY (entity_a, entity_b)
);

CREATE INDEX idx_entity_graph_a ON entity_graph (entity_a);
CREATE INDEX idx_entity_graph_b ON entity_graph (entity_b);

-- ─── Ingestion state tracking ────────────────────────────────────────────────

CREATE TYPE ingestion_status AS ENUM (
    'queued', 'fetching', 'fetched', 'parsing', 'parsed',
    'embedding', 'indexed', 'done', 'error', 'retraction'
);

CREATE TABLE ingestion_log (
    pmid            VARCHAR(20) PRIMARY KEY,
    pmcid           VARCHAR(20),
    status          ingestion_status NOT NULL DEFAULT 'queued',
    error_message   TEXT,
    retry_count     SMALLINT DEFAULT 0,
    fetched_at      TIMESTAMPTZ,
    parsed_at       TIMESTAMPTZ,
    indexed_at      TIMESTAMPTZ,
    xml_checksum    CHAR(64),           -- SHA-256 of raw XML for change detection
    queued_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_ingestion_log_status ON ingestion_log (status);
CREATE INDEX idx_ingestion_log_updated ON ingestion_log (updated_at);

-- ─── Query response audit log ───────────────────────────────────────────────

CREATE TABLE query_responses (
    id                  BIGSERIAL PRIMARY KEY,
    query_text          TEXT NOT NULL,
    query_type          TEXT,               -- 'factual', 'exploratory', 'comparative'
    response            JSONB NOT NULL,
    chunk_ids           BIGINT[],
    paper_ids           BIGINT[],
    retrieval_strategy  TEXT,
    model_used          TEXT,               -- NULL for extractive
    latency_ms          INT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Helper function: update updated_at on ingestion_log ────────────────────

CREATE OR REPLACE FUNCTION update_ingestion_log_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_ingestion_log_updated_at
    BEFORE UPDATE ON ingestion_log
    FOR EACH ROW EXECUTE FUNCTION update_ingestion_log_timestamp();
