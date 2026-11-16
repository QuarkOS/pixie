# Auth (web app)

User login is optional magic-link email plus Google and X OAuth. All client IDs, secrets, cookie keys, and public origins come from environment variables. See `.env.example`.

Required for sessions: `AUTH_SECRET`, `AUTH_PUBLIC_URL`.

Optional providers:

- Google: `AUTH_GOOGLE_CLIENT_ID`, `AUTH_GOOGLE_CLIENT_SECRET`
- X: `AUTH_X_CLIENT_ID`, `AUTH_X_CLIENT_SECRET`
- Mail: `AUTH_MAILER`, `AUTH_RESEND_API_KEY`, `AUTH_FROM_EMAIL`

The operator account is identified by `OWNER_EMAIL` / `OWNER_LID`. Do not hard-code an email, a domain, or an OAuth client id.

`AUTH_COOKIE_SECURE` should be true behind HTTPS. Callback URLs must match `AUTH_PUBLIC_URL`.

Admin API auth is separate (`ADMIN_PASSWORD`, `TOKEN_SECRET`) and must also stay in env. Do not put tokens in admin URLs or committed files.
