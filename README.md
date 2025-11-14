# QR Redirect Server

Lightweight Flask server intended for deployment on Render. It accepts POST
requests from a Power Automate flow, maintains an in-memory map of redirect
links, and returns QR codes that point at server-managed redirect URLs.

## Features

- `/webhook` accepts a JSON payload containing a `unique_id` and `final_url`
- Generates a server-hosted redirect URL and corresponding QR code (base64 PNG)
- Updates existing entries when the same `unique_id` is posted again
- `/redirect/<slug>` and `/<slug>` redirect to the stored `final_url` (302 redirect)
- `/health` endpoint for uptime checks
- `/admin/form` simple HTML form to create/update redirects during testing
- `/admin/entries` table view of the in-memory map plus links to `/qr/<unique_id>` and inline delete confirmation
- Import/export helpers for JSON or CSV backups

## Project Structure

- `app.py` – Flask application entry point
- `requirements.txt` – Python dependencies

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
export FLASK_APP=app.py
flask run --reload
```

When running locally, the server will infer the base URL from the incoming
request. For production (Render), set the `BASE_URL` environment variable to the
public origin of your service (for example `https://your-service.onrender.com`).

## Quick Test Helper

Run the server locally, then in a separate terminal execute:

```bash
python scripts/send_test_request.py YOUR-UNIQUE-ID https://example.com
```

- Creates/updates a redirect available at `http://127.0.0.1:5000/YOUR-UNIQUE-ID`
- Saves a `qr_code.png` file pointing to that redirect
- Prints the JSON response, including the ready-to-share QR code (base64)

Use `--qr-output` to pick a different filename or `--host` to target another
deployment (e.g., your Render URL).

## Browser-Based Testing

- Visit `http://localhost:5000/admin/form` to submit a `unique_id` and `final_url` via a web form.
- After submission, the page displays the redirect URL and embedded QR image.
- Visit `http://localhost:5000/admin/entries` to view all stored redirects, download QR links, and delete entries with confirmation.
- Scanning a QR (or visiting `/<slug>`) immediately redirects to the destination URL (302 redirect).
- Use the **Import / Export** section on `admin/form` to download JSON/CSV snapshots or import saved data (useful after redeploys).

## Deploying to Render

1. Push this project to GitHub (for example, create a repo named `qr-redirect-server` and push the files).
2. Visit [Render](https://render.com), create a new **Web Service**, and point it at your GitHub repo.
3. Render auto-detects `render.yaml`; otherwise configure manually:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
4. Add the environment variable `BASE_URL` with your Render service URL (Render shows it after the first deploy, e.g. `https://your-app.onrender.com`).
5. Deploy. Once live, test via:
   - POST to `/webhook`
   - Visit `/admin/form` for manual submissions
   - Fetch QR images from `/qr/<unique_id>`

Render stores no state between deploys or restarts, so redirects reset unless you connect a persistent store.

## Example Request

```bash
curl -X POST http://localhost:5000/webhook \
     -H "Content-Type: application/json" \
     -d '{"unique_id": "ABC-123", "final_url": "https://example.com"}'
```

Response includes:

- `redirect_url`: server-hosted link to embed in QR codes
- `qr_code_base64`: base64-encoded PNG image of the QR code (no logo yet)
- `status`: `created` for new entries, `updated` for existing ones

## Deployment Notes

- Render typically runs Python services with `gunicorn`. A sample start command:
  `gunicorn app:app`
- The current implementation uses in-memory storage. For persistence across
  restarts, integrate a database or cache service (e.g., Redis, PostgreSQL).
- QR customization (logo overlays, styling) can be added later by extending the
  `_encode_qr_code` helper.

