-- ABIS Schema Migrations: Enrichment Columns & Status
-- Run these statements against PostgreSQL before deploying code updates:

ALTER TABLE donors ADD COLUMN IF NOT EXISTS sex VARCHAR(1) NULL;
ALTER TABLE donors ADD COLUMN IF NOT EXISTS donor_type VARCHAR(30) NULL;
ALTER TABLE donors ADD COLUMN IF NOT EXISTS date_of_birth DATE NULL;

ALTER TABLE facilities ADD COLUMN IF NOT EXISTS keph_level INTEGER NULL;

ALTER TABLE transfusion_requests ADD COLUMN IF NOT EXISTS status VARCHAR(20) NULL DEFAULT 'PENDING';
