# Chill AI (starter)
Koyeb settings: Builder = Dockerfile, Port = 8000 (HTTP), Health check path = /health.
Env vars: see .env.example. Put SECRET_KEY, DATABASE_URL, ANTHROPIC_AUTH_TOKEN in Koyeb Secrets.
Owner: sign up with OWNER_EMAIL -> auto approved owner. Others stay "pending" until approved at /admin.
Run locally: pip install -r requirements.txt && uvicorn app.main:app --reload
