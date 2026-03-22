#!/usr/bin/env python3
"""
Generate a P-256 ECDSA keypair for a forensics-pdf-mcp client.

Writes three files to --output-dir:
    private_key.pem   EC private key (chmod 600, never share)
    public_key.pem    EC public key  (register with the server)
    key_id.txt        UUID that identifies this keypair

Usage:
    python3 admin/keygen.py --client-name my-workstation
    python3 admin/keygen.py --client-name ci-runner --output-dir /tmp/ci-keys/
"""

import argparse
import os
import pathlib
import stat
import uuid

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def generate_keypair(client_name: str, output_dir: str) -> dict:
    out = pathlib.Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    key_id = str(uuid.uuid4())
    private_key = ec.generate_private_key(ec.SECP256R1())

    priv_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pub_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    priv_path = out / "private_key.pem"
    priv_path.write_bytes(priv_pem)
    os.chmod(priv_path, stat.S_IRUSR | stat.S_IWUSR)  # chmod 600

    (out / "public_key.pem").write_bytes(pub_pem)
    (out / "key_id.txt").write_text(key_id)

    return {
        "key_id": key_id,
        "client_name": client_name,
        "output_dir": str(out),
        "private_key_path": str(priv_path),
        "public_key_path": str(out / "public_key.pem"),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a forensics-pdf-mcp client keypair")
    parser.add_argument("--client-name", required=True, help="Human-readable label for this client")
    parser.add_argument(
        "--output-dir",
        default="~/.forensics-pdf-mcp/",
        help="Directory to write key files (default: ~/.forensics-pdf-mcp/)",
    )
    args = parser.parse_args()

    result = generate_keypair(args.client_name, args.output_dir)

    print(f"Key ID:      {result['key_id']}")
    print(f"Client name: {result['client_name']}")
    print(f"Output dir:  {result['output_dir']}")
    print()
    print("Register this key with the server:")
    print(
        f"  docker exec forensics-pdf-mcp python3 admin/register_client.py \\\n"
        f"    --key-id {result['key_id']} \\\n"
        f"    --client-name '{result['client_name']}' \\\n"
        f"    --public-key-file {result['public_key_path']}"
    )


if __name__ == "__main__":
    main()
