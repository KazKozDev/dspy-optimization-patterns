# Frontend Prototype

This directory contains lightweight browser-based interfaces for testing the DSPy API without adding a frontend build pipeline. Use it when you want a local control panel for QA, RAG, classification, artifact inspection, and basic request history.

## Files

- `index.html` is the simpler baseline UI.
- `index-pro.html` is the more complete dashboard-style prototype.

## What It Covers

- API health status
- QA, RAG, and classification request forms
- Request history stored in browser localStorage
- Artifact listing from the backend
- Local settings and configuration panels

## Quick Start

1. Start the backend API from the repository root.

```bash
make run-api
```

2. Open the prototype in a browser.

```bash
open frontend/index-pro.html
```

3. Confirm the API is reachable at `http://localhost:8000`.

If the backend is healthy, the UI can call the same endpoints exposed in the main API docs.

## Backend Endpoints Used

- `GET /health`
- `POST /qa`
- `POST /rag`
- `POST /classify`
- `GET /artifacts`

## Notes

- The frontend is a static HTML prototype, not a bundled production web app.
- `index-pro.html` loads React 18 and Babel from CDNs and stores local state in the browser.
- Some UI controls may represent future backend capabilities; validate against the actual API before treating them as production features.

## When To Use It

- Demoing the project locally
- Exercising endpoints without curl
- Inspecting request and response shapes during development
- Showing compiled artifacts in a simple visual interface

## Related Docs

- [Main README](../README.md)
- [Quick Start](../QUICKSTART.md)
