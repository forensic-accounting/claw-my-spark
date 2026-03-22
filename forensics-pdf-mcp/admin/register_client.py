#!/usr/bin/env python3
"""
Register (or revoke) a client public key in the server's SQLite key registry.

Must be run inside the container (or via docker exec) where DB_PATH is set.

Usage:
    # Register a new key
    python3 admin/register_client.py \\
        --key-id <uuid> \\
        --client-name my-workstation \\
        --public-key-file ~/.forensics-pdf-mcp/public_key.pem

    # Revoke a key
    python3 admin/register_client.py --revoke --key-id <uuid>

    # List all registered keys
    python3 admin/register_client.py --list
"""

import argparse
import os
import pathlib
import sys

# Allow running from repo root as well as from /app
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from auth.key_registry import KeyRegistry


def get_registry() -> KeyRegistry:
    db_path = os.getenv("DB_PATH", "/data/keys.db")
    registry = KeyRegistry(db_path)
    registry.init_db()
    return registry


def main():
    parser = argparse.ArgumentParser(description="Manage forensics-pdf-mcp client keys")
    parser.add_argument("--key-id", help="UUID of the key to register or revoke")
    parser.add_argument("--client-name", help="Human-readable label for the client")
    parser.add_argument("--public-key-file", help="Path to PEM public key file")
    parser.add_argument("--public-key", help="PEM public key content (alternative to --public-key-file)")
    parser.add_argument("--revoke", action="store_true", help="Revoke the specified key")
    parser.add_argument("--list", action="store_true", help="List all registered keys")
    args = parser.parse_args()

    registry = get_registry()

    if args.list:
        keys = registry.list_keys()
        if not keys:
            print("No keys registered.")
        for k in keys:
            status = "active" if k["active"] else "REVOKED"
            print(f"  [{status}] {k['key_id']}  {k['client_name']}  (created {k['created_at']})")
        return

    if args.revoke:
        if not args.key_id:
            parser.error("--key-id is required with --revoke")
        registry.revoke_key(args.key_id)
        print(f"Revoked key {args.key_id}")
        return

    # Register
    if not args.key_id or not args.client_name:
        parser.error("--key-id and --client-name are required to register a key")

    if args.public_key_file:
        pub_pem = pathlib.Path(args.public_key_file).read_text()
    elif args.public_key:
        pub_pem = args.public_key
    else:
        parser.error("Provide --public-key-file or --public-key")

    registry.register_key(args.key_id, args.client_name, pub_pem)
    print(f"Registered key {args.key_id} for client '{args.client_name}'")


if __name__ == "__main__":
    main()
