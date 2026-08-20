# Security notes for the portfolio demo

Saad.AI is designed as a lightweight Streamlit portfolio project. Chat history is session-local and no database or authentication system is required.

Provider credentials must be stored as Hugging Face Space secrets or server-side environment variables. Never commit `.env`, provider keys, API responses containing secrets, uploaded files, or generated caches.

The application bounds provider timeouts, uploaded file size, and PDF page count through `src/config.py`. Keep those limits enabled when deploying the demo, and review uploaded-file behavior before publishing a public URL.

For a portfolio release, verify that the following checks pass:

| Check | Required outcome |
|---|---|
| Provider keys | Stored only as server-side secrets |
| Upload limits | `MAX_UPLOAD_BYTES` and `MAX_PDF_PAGES` remain configured |
| Error messages | Do not expose API credentials or private environment values |
| Tests | CI passes on the supported Python versions |
| Demo history | Clearly described as session-local, not permanent storage |
