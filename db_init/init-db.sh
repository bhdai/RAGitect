#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE ragitect_test'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'ragitect_test')\gexec
EOSQL
