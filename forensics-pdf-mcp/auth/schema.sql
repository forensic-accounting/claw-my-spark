-- Registered client public keys
CREATE TABLE IF NOT EXISTS keys (
    key_id      TEXT PRIMARY KEY,
    client_name TEXT NOT NULL,
    public_key  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 1
);

-- Single-use nonces — prevents replay attacks
-- Pruned automatically; rows older than 5 minutes are removed on each insert
CREATE TABLE IF NOT EXISTS nonces (
    nonce    TEXT PRIMARY KEY,
    key_id   TEXT NOT NULL,
    used_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nonces_used_at ON nonces (used_at);
