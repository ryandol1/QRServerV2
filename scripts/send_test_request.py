import argparse
import base64
import json
from pathlib import Path
from typing import Any, Dict

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a test payload to the QR redirect webhook."
    )
    parser.add_argument("unique_id", help="Unique identifier to register.")
    parser.add_argument("final_url", help="Destination URL to redirect to.")
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:5000",
        help="Server host (default: http://127.0.0.1:5000).",
    )
    parser.add_argument(
        "--qr-output",
        type=Path,
        default=Path("qr_code.png"),
        help="Path to save the QR code image (default: qr_code.png).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the payload instead of sending the request.",
    )
    return parser.parse_args()


def save_qr_code(image_b64: str, output_path: Path) -> None:
    data = base64.b64decode(image_b64)
    output_path.write_bytes(data)


def main() -> None:
    args = parse_args()
    payload: Dict[str, Any] = {
        "unique_id": args.unique_id,
        "final_url": args.final_url,
    }

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return

    response = requests.post(
        f"{args.host.rstrip('/')}/webhook",
        json=payload,
        timeout=10,
    )

    print(f"Status: {response.status_code}")
    response.raise_for_status()
    data = response.json()
    print(json.dumps(data, indent=2))

    qr_code_base64 = data.get("qr_code_base64")
    if qr_code_base64:
        save_qr_code(qr_code_base64, args.qr_output)
        print(f"Saved QR code to {args.qr_output.resolve()}")
    else:
        print("No QR code returned in response.")


if __name__ == "__main__":
    main()

