# Security and deployment checklist

Saad.AI must not enable shared Supabase chat persistence in a public deployment until the database has a `scope_id` column and row-level security policies that restrict `select`, `insert`, `update`, and `delete` operations to the authenticated or explicitly assigned scope.

The application keeps `ENABLE_SUPABASE_PERSISTENCE=false` by default. When persistence is enabled, configure `SUPABASE_SCOPE_ID` with a stable value that represents the intended tenant or authenticated user boundary. Do not use a shared service credential in a browser-exposed client, and do not reuse one scope identifier for unrelated users.

Provider credentials must be stored as Hugging Face Space secrets or server-side environment variables. Never commit `.env`, `secrets.toml`, provider keys, Supabase keys, uploaded files, or generated caches.

Before a production release, verify the following:

| Check | Required outcome |
|---|---|
| Supabase RLS | Enabled and tested for cross-scope isolation |
| Persistence feature flag | Disabled unless the database migration and policies are complete |
| Provider keys | Stored only as server-side secrets |
| Upload limits | `MAX_UPLOAD_BYTES` and `MAX_PDF_PAGES` set to acceptable values |
| Error messages | Do not expose provider keys or internal service credentials |
| Dependency updates | Reviewed and tested through CI before deployment |
| Branch release | Deploy a reviewed commit from the stabilization branch, not an uncommitted local tree |
