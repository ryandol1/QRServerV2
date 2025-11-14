import base64
import csv
import json
import os
import re
from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple, TypedDict

from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    url_for,
)
import qrcode
import requests


class RedirectEntryBase(TypedDict):
    final_url: str
    redirect_slug: str


class RedirectEntry(RedirectEntryBase, total=False):
    local_file_path: str
    qr_local_base64: str


app = Flask(__name__)

# Simple in-memory store mapping external unique IDs to redirect data.
redirect_map: Dict[str, RedirectEntry] = {}

# Directory to store downloaded PDFs
PDF_STORAGE_DIR = Path("pdfs")
PDF_STORAGE_DIR.mkdir(exist_ok=True)


def _sanitize_unique_id(raw_id: str) -> str:
    """
    Convert the provided unique identifier into a URL-safe slug.
    Keeps alphanumeric characters and dashes/underscores, replaces others with dashes.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw_id.strip())
    return cleaned or "link"


def _build_redirect_url(slug: str) -> str:
    """
    Build a fully qualified redirect URL.
    Priority:
      1. BASE_URL environment variable (expected when running on Render)
      2. Request host URL (useful for local development)
    """
    env_base = os.getenv("BASE_URL")
    if env_base:
        base = env_base.rstrip("/")
    else:
        base = request.url_root.rstrip("/")
    return f"{base}/{slug}"


def _generate_qr_png(data: str) -> BytesIO:
    qr_image = qrcode.make(data)
    buffer = BytesIO()
    qr_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def _encode_qr_code(data: str) -> str:
    """Generate a PNG QR code for the provided data and return it as a base64 string."""
    buffer = _generate_qr_png(data)
    return base64.b64encode(buffer.read()).decode("ascii")


def _download_pdf(url: str, unique_id: str) -> Optional[str]:
    """
    Download a PDF from the given URL and save it locally.
    Returns the local file path if successful, None otherwise.
    """
    try:
        # Download the file
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Check if content type is PDF (but don't fail if it's not set correctly)
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            # Still try to save it, but log a warning
            pass
        
        # Generate a safe filename from unique_id
        safe_id = _sanitize_unique_id(unique_id)
        file_path = PDF_STORAGE_DIR / f"{safe_id}.pdf"
        
        # Ensure the directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Verify the file is actually a PDF by checking the header
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
                if header != b"%PDF":
                    print(f"Warning: Downloaded file for {unique_id} does not appear to be a valid PDF (header: {header})")
                    # Still return the path, but log the warning
        except Exception as e:
            print(f"Error verifying PDF for {unique_id}: {e}")
        
        # Return absolute path to avoid path resolution issues
        return str(file_path.resolve())
    except Exception as e:
        # Log error but don't fail the entire request
        print(f"Error downloading PDF from {url}: {e}")
        return None


def _ensure_unique_slug(candidate: str, existing_slugs: Iterable[str] | None = None) -> str:
    if existing_slugs is None:
        slug_set: Set[str] = {entry["redirect_slug"] for entry in redirect_map.values()}
    else:
        slug_set = set(existing_slugs)
    redirect_slug = candidate
    original_slug = redirect_slug
    suffix = 1
    while redirect_slug in slug_set:
        redirect_slug = f"{original_slug}-{suffix}"
        suffix += 1
    return redirect_slug


def _register_redirect(unique_id: str, final_url: str) -> Tuple[RedirectEntry, str, str]:
    unique_id = str(unique_id).strip()
    if not unique_id:
        raise ValueError("unique_id is required")
    if not final_url:
        raise ValueError("final_url is required")
    if not isinstance(final_url, str):
        raise TypeError("final_url must be a string")

    existing_entry = redirect_map.get(unique_id)

    if existing_entry:
        existing_entry["final_url"] = final_url
        status = "updated"
        redirect_slug = existing_entry["redirect_slug"]
    else:
        redirect_slug = _ensure_unique_slug(_sanitize_unique_id(unique_id))
        redirect_map[unique_id] = {
            "final_url": final_url,
            "redirect_slug": redirect_slug,
        }
        status = "created"

    # Download PDF and create local QR code
    local_file_path = _download_pdf(final_url, unique_id)
    qr_local_base64 = None
    
    if local_file_path:
        # Build URL to serve the PDF in browser
        env_base = os.getenv("BASE_URL")
        if env_base:
            base = env_base.rstrip("/")
        else:
            base = request.url_root.rstrip("/")
        local_pdf_url = f"{base}/pdf/{unique_id}"
        
        # Generate QR code for local PDF URL
        qr_local_base64 = _encode_qr_code(local_pdf_url)
        
        # Store in entry
        redirect_map[unique_id]["local_file_path"] = local_file_path
        redirect_map[unique_id]["qr_local_base64"] = qr_local_base64
    else:
        # Clear local file path if download failed
        redirect_map[unique_id].pop("local_file_path", None)
        redirect_map[unique_id].pop("qr_local_base64", None)

    redirect_url = _build_redirect_url(redirect_slug)
    return redirect_map[unique_id], redirect_url, status


def _serialize_entries() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for unique_id, entry in sorted(redirect_map.items(), key=lambda item: item[0]):
        row = {
            "unique_id": unique_id,
            "redirect_slug": entry["redirect_slug"],
            "final_url": entry["final_url"],
        }
        # Include local QR code if available (but not local_file_path as it's server-specific)
        if entry.get("qr_local_base64"):
            row["qr_local_base64"] = entry["qr_local_base64"]
        rows.append(row)
    return rows


def _build_map_from_rows(rows: Iterable[Dict[str, str]]) -> Dict[str, RedirectEntry]:
    prepared: Dict[str, RedirectEntry] = {}
    used_slugs: Set[str] = set()
    def canonicalize(key: str) -> str:
        return re.sub(r"[^a-z0-9]", "", key.strip().lower())

    for raw in rows:
        normalized = {canonicalize(str(k)): v for k, v in raw.items()}
        # Skip completely empty rows.
        if not any((value or "").strip() for value in normalized.values()):
            continue

        def _lookup(keys: Iterable[str]) -> str | None:
            for key in keys:
                value = normalized.get(canonicalize(key))
                if value is not None and str(value).strip():
                    return str(value)
            return None

        unique_id_raw = _lookup(("unique_id", "unique id", "uniqueid", "id"))
        unique_id = str(unique_id_raw or "").strip()
        final_url_raw = _lookup(("final_url", "final url", "finalurl", "url", "destination_url"))
        final_url = final_url_raw
        redirect_slug_raw = _lookup(("redirect_slug", "redirect slug", "redirectslug", "slug"))
        redirect_slug = redirect_slug_raw

        if not unique_id:
            raise ValueError("unique_id is required for each row")
        if unique_id in prepared:
            raise ValueError(f"duplicate unique_id detected: {unique_id}")
        if not final_url:
            raise ValueError(f"final_url is required for unique_id {unique_id}")
        if not isinstance(final_url, str):
            raise ValueError(f"final_url must be a string for unique_id {unique_id}")

        if redirect_slug:
            redirect_slug = str(redirect_slug).strip()
        if not redirect_slug:
            redirect_slug = _sanitize_unique_id(unique_id)

        redirect_slug = _ensure_unique_slug(redirect_slug, used_slugs)
        used_slugs.add(redirect_slug)

        entry: RedirectEntry = {
            "final_url": final_url,
            "redirect_slug": redirect_slug,
        }
        
        # Handle qr_local_base64 if present in import (local_file_path will be regenerated)
        qr_local_base64_raw = _lookup(("qr_local_base64", "qr local base64", "qrlocalbase64", "local_qr"))
        if qr_local_base64_raw:
            entry["qr_local_base64"] = str(qr_local_base64_raw).strip()
        
        prepared[unique_id] = entry

    return prepared


def _import_entries_from_json(payload: str) -> int:
    try:
        data = json.loads(payload or "[]")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc

    if not isinstance(data, list):
        raise ValueError("JSON payload must be a list of objects")

    new_map = _build_map_from_rows(data)
    redirect_map.clear()
    redirect_map.update(new_map)
    return len(new_map)


def _import_entries_from_csv(payload: str) -> int:
    reader = csv.DictReader(StringIO(payload or ""))
    rows: List[Dict[str, str]] = list(reader)
    if not rows and reader.fieldnames is None:
        raise ValueError("CSV payload is empty or missing headers")

    new_map = _build_map_from_rows(rows)
    redirect_map.clear()
    redirect_map.update(new_map)
    return len(new_map)


def _export_entries_as_json() -> str:
    return json.dumps(_serialize_entries(), indent=2)


def _export_entries_as_csv() -> str:
    output = StringIO()
    # Get all rows to determine fieldnames dynamically
    rows = _serialize_entries()
    fieldnames = ["unique_id", "redirect_slug", "final_url"]
    # Add qr_local_base64 if any row has it
    if any("qr_local_base64" in row for row in rows):
        fieldnames.append("qr_local_base64")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


def _resolve_redirect(slug: str):
    for entry in redirect_map.values():
        if entry["redirect_slug"] == slug:
            return redirect(entry["final_url"], code=302)
    return jsonify({"error": "redirect_not_found"}), 404


@app.route("/<slug>", methods=["GET"])
def follow_redirect_root(slug: str):
    # Health and webhook routes take precedence because Flask matches
    # explicit routes before parameterized ones.
    return _resolve_redirect(slug)


@app.route("/redirect/<slug>", methods=["GET"])
def follow_redirect(slug: str):
    return _resolve_redirect(slug)


@app.route("/webhook", methods=["POST"])
def handle_webhook():
    payload = request.get_json(silent=True) or {}

    unique_id = str(payload.get("unique_id", "")).strip()
    final_url = payload.get("final_url")

    if not unique_id:
        return jsonify({"error": "unique_id is required"}), 400

    if not final_url:
        return jsonify({"error": "final_url is required"}), 400

    if not isinstance(final_url, str):
        return jsonify({"error": "final_url must be a string"}), 400

    try:
        entry, redirect_url, status = _register_redirect(unique_id, final_url)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    qr_code_b64 = _encode_qr_code(redirect_url)
    
    response_data = {
        "unique_id": unique_id,
        "redirect_url": redirect_url,
        "final_url": entry["final_url"],
        "qr_code_base64": qr_code_b64,
        "status": status,
    }
    
    # Add local QR code if available
    if entry.get("qr_local_base64"):
        env_base = os.getenv("BASE_URL")
        if env_base:
            base = env_base.rstrip("/")
        else:
            base = request.url_root.rstrip("/")
        response_data["qr_local_base64"] = entry["qr_local_base64"]
        response_data["local_pdf_url"] = f"{base}/pdf/{unique_id}"

    return (
        jsonify(response_data),
        201 if status == "created" else 200,
    )


@app.route("/admin/form", methods=["GET", "POST"])
def admin_form():
    template = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>QR Redirect Admin</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2rem; }
            form { margin-bottom: 2rem; }
            label { display: block; margin-bottom: 0.5rem; }
            input { padding: 0.5rem; width: 24rem; max-width: 100%; margin-bottom: 1rem; }
            button { padding: 0.5rem 1rem; cursor: pointer; }
            .result { border: 1px solid #ccc; padding: 1.5rem; max-width: 30rem; }
            .qr { margin-top: 1rem; }
            .error { color: #b00020; }
            .notice { color: #0b6e02; }
            textarea { width: 100%; max-width: 36rem; min-height: 8rem; padding: 0.5rem; }
            select { padding: 0.4rem; }
        </style>
    </head>
    <body>
        <h1>Create or Update Redirect</h1>
        <form method="post">
            <label>
                Unique ID:
                <input type="text" name="unique_id" value="{{ unique_id|default('') }}" required />
            </label>
            <label>
                Final URL:
                <input type="url" name="final_url" value="{{ final_url|default('') }}" required />
            </label>
            <button type="submit">Submit</button>
        </form>

        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}

        {% if result %}
        <div class="result">
            <p>Status: <strong>{{ result.status }}</strong></p>
            <p>Redirect URL: <a href="{{ result.redirect_url }}">{{ result.redirect_url }}</a></p>
            <p>Final URL: <a href="{{ result.final_url }}">{{ result.final_url }}</a></p>
            <div class="qr">
                <h3>Redirect QR Code</h3>
                <img src="data:image/png;base64,{{ result.qr_code_base64 }}" alt="QR Code" />
            </div>
            {% if result.qr_local_base64 %}
            <div class="qr">
                <h3>Local PDF QR Code</h3>
                <img src="data:image/png;base64,{{ result.qr_local_base64 }}" alt="Local QR Code" />
                <p><a href="{{ result.local_pdf_url }}">View Local PDF</a></p>
            </div>
            {% endif %}
        </div>
        {% endif %}

        <p><a href="{{ url_for('view_entries') }}">View all entries</a></p>
        {% if unique_id %}
            <p><a href="{{ url_for('serve_qr_code', unique_id=unique_id) }}">Direct QR image for {{ unique_id }}</a></p>
        {% endif %}

        <h2>Import / Export</h2>
        <p>
            Download current data:
            <a href="{{ url_for('export_entries', fmt='json') }}">JSON</a> |
            <a href="{{ url_for('export_entries', fmt='csv') }}">CSV</a>
        </p>

        {% if import_status %}
            <p class="notice">{{ import_status }}</p>
        {% endif %}
        {% if import_error %}
            <p class="error">{{ import_error }}</p>
        {% endif %}

        <form method="post" action="{{ url_for('import_entries') }}">
            <label>
                Data format:
                <select name="format">
                    <option value="json">JSON</option>
                    <option value="csv">CSV</option>
                </select>
            </label>
            <label>
                Paste data here:
                <textarea name="payload" placeholder="Paste exported data here..." required></textarea>
            </label>
            <button type="submit">Import entries</button>
        </form>
    </body>
    </html>
    """

    context = {
        "unique_id": "",
        "final_url": "",
        "result": None,
        "error": None,
        "import_status": request.args.get("import_status"),
        "import_error": request.args.get("import_error"),
    }

    if request.method == "POST":
        unique_id = request.form.get("unique_id", "").strip()
        final_url = request.form.get("final_url", "").strip()
        context["unique_id"] = unique_id
        context["final_url"] = final_url

        try:
            entry, redirect_url, status = _register_redirect(unique_id, final_url)
            qr_code_b64 = _encode_qr_code(redirect_url)
            
            env_base = os.getenv("BASE_URL")
            if env_base:
                base = env_base.rstrip("/")
            else:
                base = request.url_root.rstrip("/")
            
            result_data = {
                "status": status,
                "redirect_url": redirect_url,
                "final_url": entry["final_url"],
                "qr_code_base64": qr_code_b64,
            }
            
            if entry.get("qr_local_base64"):
                result_data["qr_local_base64"] = entry["qr_local_base64"]
                result_data["local_pdf_url"] = f"{base}/pdf/{unique_id}"
            
            context["result"] = result_data
        except (ValueError, TypeError) as exc:
            context["error"] = str(exc)

    return render_template_string(template, **context)


@app.route("/qr/<unique_id>", methods=["GET"])
def serve_qr_code(unique_id: str):
    entry = redirect_map.get(unique_id)
    if not entry:
        return jsonify({"error": "unique_id not found"}), 404
    redirect_url = _build_redirect_url(entry["redirect_slug"])
    buffer = _generate_qr_png(redirect_url)
    return send_file(
        buffer,
        mimetype="image/png",
        as_attachment=False,
        download_name=f"{unique_id}.png",
    )


@app.route("/qr-local/<unique_id>", methods=["GET"])
def serve_qr_code_local(unique_id: str):
    """Serve the local PDF QR code for a given unique_id."""
    entry = redirect_map.get(unique_id)
    if not entry:
        return jsonify({"error": "unique_id not found"}), 404
    
    qr_local_base64 = entry.get("qr_local_base64")
    if not qr_local_base64:
        return jsonify({"error": "local QR code not available"}), 404
    
    # Decode base64 and return as PNG
    qr_data = base64.b64decode(qr_local_base64)
    buffer = BytesIO(qr_data)
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype="image/png",
        as_attachment=False,
        download_name=f"{unique_id}-local.png",
    )


def _get_pdf_path(unique_id: str) -> Tuple[Optional[Path], Optional[str]]:
    """Helper function to get the PDF file path for a given unique_id."""
    entry = redirect_map.get(unique_id)
    if not entry:
        return None, "unique_id not found"
    
    local_file_path = entry.get("local_file_path")
    if not local_file_path:
        return None, "local PDF not available"
    
    # Try to resolve the file path - handle both absolute and relative paths
    file_path = Path(local_file_path)
    if not file_path.is_absolute():
        # If relative, resolve it relative to the PDF_STORAGE_DIR
        file_path = PDF_STORAGE_DIR / file_path.name
    
    # If file doesn't exist, try to reconstruct the path from unique_id
    if not file_path.exists():
        safe_id = _sanitize_unique_id(unique_id)
        file_path = PDF_STORAGE_DIR / f"{safe_id}.pdf"
    
    if not file_path.exists():
        return None, f"PDF file not found on server (expected path: {file_path})"
    
    return file_path, None


@app.route("/pdf/<unique_id>", methods=["GET"])
def serve_pdf(unique_id: str):
    """Serve the locally stored PDF file in browser (not as download)."""
    file_path, error = _get_pdf_path(unique_id)
    if not file_path:
        return jsonify({
            "error": error,
            "unique_id": unique_id
        }), 404
    
    try:
        response = send_file(
            str(file_path),
            mimetype="application/pdf",
            as_attachment=False,
            download_name=f"{unique_id}.pdf",
        )
        # Ensure the PDF is displayed inline in the browser
        response.headers["Content-Disposition"] = f'inline; filename="{unique_id}.pdf"'
        response.headers["Content-Type"] = "application/pdf"
        # Add cache control headers
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response
    except Exception as e:
        print(f"Error serving PDF for {unique_id}: {e}")
        return jsonify({
            "error": "Error serving PDF file",
            "details": str(e)
        }), 500


@app.route("/pdf/<unique_id>/download", methods=["GET"])
def download_pdf(unique_id: str):
    """Download the locally stored PDF file."""
    file_path, error = _get_pdf_path(unique_id)
    if not file_path:
        return jsonify({
            "error": error,
            "unique_id": unique_id
        }), 404
    
    try:
        response = send_file(
            str(file_path),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{unique_id}.pdf",
        )
        # Force download
        response.headers["Content-Disposition"] = f'attachment; filename="{unique_id}.pdf"'
        response.headers["Content-Type"] = "application/pdf"
        return response
    except Exception as e:
        print(f"Error downloading PDF for {unique_id}: {e}")
        return jsonify({
            "error": "Error downloading PDF file",
            "details": str(e)
        }), 500


@app.route("/admin/export/<fmt>", methods=["GET"])
def export_entries(fmt: str):
    fmt = fmt.lower()
    if fmt == "json":
        payload = _export_entries_as_json()
        mimetype = "application/json"
        filename = "redirects.json"
    elif fmt == "csv":
        payload = _export_entries_as_csv()
        mimetype = "text/csv"
        filename = "redirects.csv"
    else:
        return jsonify({"error": "unsupported format"}), 400

    response = Response(payload, mimetype=mimetype)
    response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


@app.route("/admin/import", methods=["POST"])
def import_entries():
    fmt = (request.form.get("format") or "json").lower()

    payload = request.form.get("payload", "")
    if not payload and "file" in request.files:
        uploaded = request.files["file"]
        if uploaded:
            payload = uploaded.read().decode("utf-8")

    payload = payload.strip()

    if not payload:
        return redirect(url_for("admin_form", import_error="No data provided for import"))

    try:
        if fmt == "json":
            imported_count = _import_entries_from_json(payload)
        elif fmt == "csv":
            imported_count = _import_entries_from_csv(payload)
        else:
            raise ValueError("Unsupported format; choose JSON or CSV")
    except ValueError as exc:
        return redirect(url_for("admin_form", import_error=str(exc)))

    suffix = "entry" if imported_count == 1 else "entries"
    message = f"Imported {imported_count} {suffix}"
    return redirect(url_for("admin_form", import_status=message))


@app.route("/admin/entries", methods=["GET"])
def view_entries():
    template = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <title>Redirect Map</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 2rem; }
            table { border-collapse: collapse; width: 100%; max-width: 60rem; }
            th, td { border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }
            tr:nth-child(even) { background: #f7f7f7; }
            .actions { white-space: nowrap; }
            .confirm { display: inline-flex; gap: 0.5rem; align-items: center; }
            .notice { color: #0b6e02; }
            .error { color: #b00020; }
        </style>
    </head>
    <body>
        <h1>Registered Redirects</h1>
        {% if message %}
            <p class="notice">{{ message }}</p>
        {% endif %}
        {% if error %}
            <p class="error">{{ error }}</p>
        {% endif %}
        {% if entries %}
        <table>
            <thead>
                <tr>
                    <th>Unique ID</th>
                    <th>Redirect URL</th>
                    <th>Final URL</th>
                    <th>QR Code</th>
                    <th>Local QR Code</th>
                    <th>Local PDF</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
            {% for unique_id, entry in entries %}
                <tr>
                    <td>{{ unique_id }}</td>
                    <td><a href="{{ entry.redirect_url }}">{{ entry.redirect_url }}</a></td>
                    <td><a href="{{ entry.final_url }}">{{ entry.final_url }}</a></td>
                    <td><a href="{{ url_for('serve_qr_code', unique_id=unique_id) }}">QR</a></td>
                    <td>
                        {% if entry.has_local_qr %}
                            <a href="{{ url_for('serve_qr_code_local', unique_id=unique_id) }}">Local QR</a>
                        {% else %}
                            <span style="color: #999;">N/A</span>
                        {% endif %}
                    </td>
                    <td>
                        {% if entry.local_pdf_url %}
                            <a href="{{ entry.local_pdf_url }}">View PDF</a> | 
                            <a href="{{ url_for('download_pdf', unique_id=unique_id) }}">Download</a>
                        {% else %}
                            <span style="color: #999;">N/A</span>
                        {% endif %}
                    </td>
                    <td class="actions">
                        {% if confirm_id == unique_id %}
                        <span class="confirm">
                            <form method="post" action="{{ url_for('delete_entry', unique_id=unique_id) }}">
                                <button type="submit">Confirm delete</button>
                            </form>
                            <a href="{{ url_for('view_entries') }}">Cancel</a>
                        </span>
                        {% else %}
                            <a href="{{ url_for('view_entries', confirm=unique_id) }}">Delete</a>
                        {% endif %}
                    </td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
            <p>No entries registered yet.</p>
        {% endif %}

        <p><a href="{{ url_for('admin_form') }}">Back to form</a></p>
    </body>
    </html>
    """
    rows = []
    for unique_id, entry in redirect_map.items():
        env_base = os.getenv("BASE_URL")
        if env_base:
            base = env_base.rstrip("/")
        else:
            base = request.url_root.rstrip("/")
        
        row_data = {
            "redirect_url": _build_redirect_url(entry["redirect_slug"]),
            "final_url": entry["final_url"],
            "has_local_qr": bool(entry.get("qr_local_base64")),
            "local_pdf_url": f"{base}/pdf/{unique_id}" if entry.get("local_file_path") else None,
        }
        rows.append((unique_id, row_data))
    rows.sort(key=lambda row: row[0])
    confirm_id = request.args.get("confirm", "")
    return render_template_string(
        template,
        entries=rows,
        confirm_id=confirm_id,
        message=request.args.get("message"),
        error=request.args.get("error"),
    )


@app.route("/admin/entries/delete/<unique_id>", methods=["POST"])
def delete_entry(unique_id: str):
    removed = redirect_map.pop(unique_id, None)
    if removed:
        return redirect(url_for("view_entries", message=f"Deleted {unique_id}"))
    return redirect(url_for("view_entries", error=f"{unique_id} not found"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")))

