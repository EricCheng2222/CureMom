-- FDA drug labels — structured data from openFDA Drug Label API.
-- One row per generic drug; the rich label fields back the LLM prompt
-- when a query mentions a known drug name.

CREATE TABLE IF NOT EXISTS fda_drugs (
    id                   BIGSERIAL PRIMARY KEY,
    generic_name         TEXT NOT NULL UNIQUE,
    brand_names          TEXT[],
    application_number   TEXT,
    sponsor              TEXT,
    route                TEXT[],
    dosage_form          TEXT[],
    marketing_status     TEXT,

    -- Clinical / pharmacological label fields (free text from openFDA)
    indications_and_usage    TEXT,
    mechanism_of_action      TEXT,
    pharmacology             TEXT,
    pharmacokinetics         TEXT,
    contraindications        TEXT,
    warnings                 TEXT,
    adverse_reactions        TEXT,
    drug_interactions        TEXT,
    dosage_and_administration TEXT,
    dosage_forms_and_strengths TEXT,

    -- Cross-references
    rxcui                TEXT[],          -- RxNorm concept unique IDs
    unii                 TEXT[],          -- FDA UNII codes
    spl_id               TEXT,            -- Structured Product Label ID

    raw_json             JSONB,           -- full label record for ad-hoc queries
    fetched_at           TIMESTAMPTZ DEFAULT NOW(),
    last_label_update    DATE
);

CREATE INDEX IF NOT EXISTS idx_fda_drugs_generic_lower ON fda_drugs (LOWER(generic_name));
CREATE INDEX IF NOT EXISTS idx_fda_drugs_brand        ON fda_drugs USING GIN (brand_names);
CREATE INDEX IF NOT EXISTS idx_fda_drugs_rxcui        ON fda_drugs USING GIN (rxcui);

-- Trigram index on the generic name + brand names for fuzzy matching
-- ("mephenoxalone" should match "Mephenoxalone" or "MEPHENOXALONE", etc.)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_fda_drugs_generic_trgm
    ON fda_drugs USING GIN (LOWER(generic_name) gin_trgm_ops);
