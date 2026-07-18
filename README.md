http://127.0.0.1:5000/
# MailMind — AI Email Agent Setup Guide

## Step 1 — Install dependencies

```bash
cd email-agent
uv pip install -r requirements.txt
```

## Step 2 — Get your Google credentials.json

1. Go to https://console.cloud.google.com/
2. Create a new project (e.g. "MailMind")
3. Go to **APIs & Services → Enable APIs**
   - Enable: **Gmail API**
   - Enable: **Google People API**
4. Go to **APIs & Services → OAuth consent screen**
   - User type: External
   - App name: MailMind
   - Add your email as a test user
5. Go to **APIs & Services → Credentials**
   - Click "Create Credentials" → OAuth 2.0 Client ID
   - Application type: **Web application**
   - Authorized redirect URIs: `http://localhost:5000/oauth/callback`
6. Download the JSON → rename it to `credentials.json`
7. Place `credentials.json` in the `email-agent/` folder

## Step 3 — Set up .env

Edit `.env` and set a strong secret key:
```
FLASK_SECRET_KEY=any-long-random-string-here
```

## Step 4 — Run the app

```bash
python app.py
```

Open: http://localhost:5000

## Project structure

```
email-agent/
├── app.py              # Flask backend + Gmail OAuth + API routes
├── credentials.json    # Your Google OAuth credentials (DO NOT commit)
├── .env                # Secret key (DO NOT commit)
├── requirements.txt
└── templates/
    ├── index.html      # Landing page with Connect Google button
    └── dashboard.html  # Main dashboard (emails, categories, priority)
```

## Security notes

- `credentials.json` and `.env` are never committed to git
- Add them to `.gitignore`
- Gmail scope is READ-ONLY — the app cannot send or delete emails
- User can revoke access at: https://myaccount.google.com/permissions
