-- GoAlgo PostgreSQL / Supabase Schema Initialization
-- Matches backend/db/models.py (SQLModel) exactly.
-- Safe to run multiple times (idempotent).

CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    email VARCHAR(254) NOT NULL,
    preferred_theme VARCHAR(15) NOT NULL DEFAULT 'light',
    created_at DATE NOT NULL DEFAULT CURRENT_DATE,
    phone_number VARCHAR(20),
    saved_vis_num INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);

CREATE TABLE IF NOT EXISTS user_auth (
    password TEXT NOT NULL,
    token TEXT NOT NULL DEFAULT '',
    token_created_at DATE NOT NULL DEFAULT CURRENT_DATE,
    user_id INTEGER NOT NULL REFERENCES users (user_id) ON DELETE CASCADE,
    PRIMARY KEY (password, user_id)
);

CREATE INDEX IF NOT EXISTS idx_user_auth_user_id ON user_auth (user_id);

CREATE TABLE IF NOT EXISTS user_post (
    id SERIAL PRIMARY KEY,
    operation_type VARCHAR(30) NOT NULL,
    created_at DATE NOT NULL DEFAULT CURRENT_DATE,
    status VARCHAR(15) NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_post_user_id ON user_post (user_id);

CREATE TABLE IF NOT EXISTS users_saved_visuals (
    id SERIAL PRIMARY KEY,
    saved_visual JSON NOT NULL,
    type VARCHAR(30) NOT NULL,
    updated_at DATE NOT NULL DEFAULT CURRENT_DATE,
    user_id INTEGER NOT NULL REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_users_saved_visuals_user_id ON users_saved_visuals (user_id);

CREATE TABLE IF NOT EXISTS label_correction (
    image_path VARCHAR(256) PRIMARY KEY,
    data_structure_type VARCHAR(30) NOT NULL,
    wrong_label JSON,
    correct_label JSON NOT NULL,
    created_at DATE NOT NULL DEFAULT CURRENT_DATE,
    user_id INTEGER NOT NULL REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_label_correction_user_id ON label_correction (user_id);
