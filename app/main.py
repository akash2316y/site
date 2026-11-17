import os, secrets, json, time, logging, html, threading
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import bcrypt, httpx
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, text
from sqlalchemy.orm import declarative_base, sessionmaker

log = logging.getLogger("chillai")
logging.basicConfig(level=logging.INFO)
ENV = os.getenv("ENVIRONMENT", "development")
PROD = ENV == "production"
APP_NAME = os.getenv("APP_NAME", "Chill AI")
OWNER_EMAIL = os.getenv("OWNER_EMAIL", "").strip().lower()
SECRET_KEY = os.getenv("SECRET_KEY") or ("dev-only-" + secrets.token_hex(8))
if PROD and not os.getenv("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY must be set in production")
SIGNUPS_OPEN = os.getenv("SIGNUPS_OPEN", "true").lower() == "true"
BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
API_PATH = os.getenv("ANTHROPIC_API_PATH", "/v1/messages")
TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "")
MODELS = [m.strip() for m in (os.getenv("AI_MODELS") or os.getenv("AI_MODEL", "claude-sonnet-5-5")).split(",") if m.strip()]
START = time.time()

# ---- DB ----
db_url = os.getenv("DATABASE_URL", "sqlite:///./local.db")
if db_url.startswith("postgres://"):
    db_url = "postgresql://" + db_url[len("postgres://"):]
kw = {}
if db_url.startswith("postgresql"):
    kw = dict(pool_size=3, max_overflow=2, pool_pre_ping=True, pool_recycle=300,
              connect_args={"connect_timeout": 10, **({"sslmode": "require"} if "sslmode" not in db_url else {})})
engine = create_engine(db_url, **kw)
Session = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()
now = lambda: datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String(254), unique=True, index=True, nullable=False)
    pw_hash = Column(String(100), nullable=False)
    role = Column(String(10), default="user")
    status = Column(String(10), default="pending")
    created_at = Column(DateTime(timezone=True), default=now)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role = Column(String(10))
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), default=now)

class Audit(Base):
    __tablename__ = "audit"
    id = Column(Integer, primary_key=True)
    actor = Column(String(254))
    action = Column(String(60))
    target = Column(String(254))
    created_at = Column(DateTime(timezone=True), default=now)

_ready = False
_lock = threading.Lock()
def init_db(tries=5):
    global _ready
    with _lock:
        if _ready: return True
        for i in range(tries):
            try:
                Base.metadata.create_all(engine); _ready = True; return True
            except Exception as e:
                log.warning("DB not ready (%s), retry %s", type(e).__name__, i + 1)
                time.sleep(min(2 ** i, 8))
    return False

@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=init_db, daemon=True).start()  # never block/crash boot
    yield

def get_db():
    if not _ready and not init_db(tries=2):
        raise HTTPException(503, "Database waking up, try again")
    db = Session()
    try: yield db
    finally: db.close()

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=PROD, same_site="lax", max_age=60*60*24*14)

@app.middleware("http")
async def sec_headers(request: Request, call_next):
    r = await call_next(request)
    r.headers["X-Content-Type-Options"] = "nosniff"
    r.headers["X-Frame-Options"] = "DENY"
    r.headers["Referrer-Policy"] = "same-origin"
    return r

# ---- helpers ----
esc = html.escape
def page(title, body, code=200):
    return HTMLResponse(f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>
:root{{--bg:#faf9f5;--fg:#1f1e1d;--card:#fff;--ac:#c96442;--bd:#e5e2d9}}
@media(prefers-color-scheme:dark){{:root{{--bg:#262624;--fg:#f0eee6;--card:#30302e;--bd:#44433f}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:16px system-ui}}
main{{max-width:640px;margin:0 auto;padding:16px}}input,button{{font:inherit;padding:12px;border-radius:12px;border:1px solid var(--bd);background:var(--card);color:var(--fg);width:100%;margin:6px 0}}
button{{background:var(--ac);color:#fff;border:0;cursor:pointer}}.card{{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:12px;margin:8px 0}}
.row{{display:flex;gap:6px;flex-wrap:wrap}}.row form{{flex:1}}.u{{background:var(--card);border:1px solid var(--bd);border-radius:18px;padding:10px 14px;margin:8px 0 8px auto;width:fit-content;max-width:85%}}
.a{{font-family:Georgia,serif;margin:12px 0;white-space:pre-wrap}}small{{opacity:.6}}</style></head><body><main>{body}</main></body></html>""", status_code=code)

def csrf_token(request):
    if "csrf" not in request.session: request.session["csrf"] = secrets.token_urlsafe(24)
    return request.session["csrf"]
def csrf_ok(request, token):
    return bool(token) and secrets.compare_digest(token, request.session.get("csrf", ""))
def hidden(request): return f'<input type=hidden name=csrf value="{csrf_token(request)}">'

_hits = {}
def limited(key, n=10, per=60):
    t = time.time(); h = [x for x in _hits.get(key, []) if t - x < per]
    h.append(t); _hits[key] = h
    return len(h) > n

def current_user(request, db):
    uid = request.session.get("uid")
    return db.get(User, uid) if uid else None

def approved_user(request: Request, db=Depends(get_db)):
    u = current_user(request, db)
    if not u: raise HTTPException(401, "Login required")
    if u.status != "approved": raise HTTPException(403, "Not approved")
    return u

def owner_user(u=Depends(approved_user)):
    if u.role != "owner": raise HTTPException(403, "Owner only")
    return u

def audit(db, actor, action, target=""):
    db.add(Audit(actor=actor, action=action, target=target)); db.commit()

# ---- routes ----
@app.get("/health")
def health():
    try:
        with engine.connect() as c: c.execute(text("select 1"))
        dbok = True
    except Exception: dbok = False
    return {"status": "ok", "db": dbok, "uptime_s": int(time.time() - START)}

@app.get("/", response_class=HTMLResponse)
def home(request: Request, db=Depends(get_db)):
    u = current_user(request, db)
    if u: return RedirectResponse("/chat", 303)
    c = hidden(request)
    return page(APP_NAME, f"""<h2>{esc(APP_NAME)}</h2><div class=card><b>Login</b>
<form method=post action=/login>{c}<input name=email type=email placeholder=Email required>
<input name=password type=password placeholder=Password required><button>Login</button></form></div>
<div class=card><b>Sign up</b><form method=post action=/signup>{c}<input name=email type=email placeholder=Email required>
<input name=password type=password minlength=8 maxlength=72 placeholder="Password (8+)" required><button>Request access</button></form></div>""")

@app.post("/signup")
async def signup(request: Request, db=Depends(get_db)):
    f = await request.form()
    if not csrf_ok(request, f.get("csrf")) or limited("su:" + request.client.host, 5, 300): raise HTTPException(400, "Bad request")
    email, pw = str(f.get("email", "")).strip().lower(), str(f.get("password", ""))
    is_owner = bool(OWNER_EMAIL) and email == OWNER_EMAIL
    if not (3 < len(email) <= 254 and "@" in email and 8 <= len(pw) <= 72): return page("Error", "<p>Invalid email or password.</p><a href=/>Back</a>", 400)
    if not SIGNUPS_OPEN and not is_owner: return page("Closed", "<p>Signups are closed.</p>", 403)
    if db.query(User).filter_by(email=email).first(): return page("Error", "<p>Could not sign up.</p><a href=/>Back</a>", 400)
    u = User(email=email, pw_hash=bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode(),
             role="owner" if is_owner else "user", status="approved" if is_owner else "pending")
    db.add(u); db.commit(); audit(db, email, "signup", u.status)
    request.session.update(uid=u.id)
    return RedirectResponse("/chat", 303)

@app.post("/login")
async def login(request: Request, db=Depends(get_db)):
    f = await request.form()
    if not csrf_ok(request, f.get("csrf")) or limited("li:" + request.client.host, 10, 60): raise HTTPException(429, "Too many attempts")
    email = str(f.get("email", "")).strip().lower()
    u = db.query(User).filter_by(email=email).first()
    if not u or not bcrypt.checkpw(str(f.get("password", ""))[:72].encode(), u.pw_hash.encode()):
        return page("Error", "<p>Wrong email or password.</p><a href=/>Back</a>", 401)
    if OWNER_EMAIL and email == OWNER_EMAIL and (u.role != "owner" or u.status != "approved"):
        u.role, u.status = "owner", "approved"; db.commit()
    request.session.clear(); request.session["uid"] = u.id
    audit(db, email, "login")
    return RedirectResponse("/chat", 303)

@app.post("/logout")
async def logout(request: Request):
    f = await request.form()
    if csrf_ok(request, f.get("csrf")): request.session.clear()
    return RedirectResponse("/", 303)

@app.get("/chat", response_class=HTMLResponse)
def chat(request: Request, db=Depends(get_db)):
    u = current_user(request, db)
    if not u: return RedirectResponse("/", 303)
    out = f'<form method=post action=/logout>{hidden(request)}<button style="width:auto;background:var(--card);color:var(--fg)">Logout</button></form>'
    if u.status != "approved":
        msg = "Awaiting owner approval" if u.status == "pending" else "Access denied"
        return page(msg, f"<h2>{msg}</h2><p>Aapka account abhi chat use nahi kar sakta.</p>{out}")
    hist = db.query(Message).filter_by(user_id=u.id).order_by(Message.id.desc()).limit(30).all()[::-1]
    msgs = "".join(f'<div class="{"u" if m.role=="user" else "a"}">{esc(m.content)}</div>' for m in hist)
    opts = "".join(f"<option>{esc(m)}</option>" for m in MODELS)
    admin = '<a href=/admin>Admin</a>' if u.role == "owner" else ""
    return page(APP_NAME, f"""{admin}{out}<div id=t>{msgs}</div><div id=err></div>
<select id=m style="width:100%;padding:10px;border-radius:12px">{opts}</select>
<input id=i placeholder="Message..." maxlength=4000><button id=s>Send</button><button id=x style="display:none;background:#777">Stop</button>
<small>{esc(APP_NAME)} is AI and can make mistakes.</small>
<script>
const T="{csrf_token(request)}",t=document.getElementById('t'),i=document.getElementById('i'),s=document.getElementById('s'),x=document.getElementById('x'),e=document.getElementById('err');
let ac;x.onclick=()=>ac&&ac.abort();
s.onclick=async()=>{{const q=i.value.trim();e.textContent='';if(!q){{e.textContent='Message khali hai.';return}}
i.value='';const u=document.createElement('div');u.className='u';u.textContent=q;t.append(u);
const a=document.createElement('div');a.className='a';t.append(a);s.style.display='none';x.style.display='';ac=new AbortController();
try{{const r=await fetch('/api/chat',{{method:'POST',signal:ac.signal,headers:{{'Content-Type':'application/json','X-CSRF-Token':T}},body:JSON.stringify({{message:q,model:document.getElementById('m').value}})}});
if(!r.ok)throw new Error('HTTP '+r.status);const rd=r.body.getReader(),d=new TextDecoder();let b='';
while(true){{const{{done,value}}=await rd.read();if(done)break;b+=d.decode(value,{{stream:true}});let p;
while((p=b.indexOf('\\n\\n'))>=0){{const ev=b.slice(0,p);b=b.slice(p+2);if(ev.startsWith('data: ')){{const j=JSON.parse(ev.slice(6));if(j.t)a.textContent+=j.t;if(j.err)e.textContent=j.err}}}}}}
}}catch(z){{e.textContent='Response was interrupted. Dobara try karein.'}}
s.style.display='';x.style.display='none'}};
</script>""")

@app.post("/api/chat")
async def api_chat(request: Request, u=Depends(approved_user), db=Depends(get_db)):
    if not csrf_ok(request, request.headers.get("X-CSRF-Token")): raise HTTPException(403, "CSRF")
    if limited(f"chat:{u.id}", 20, 60): raise HTTPException(429, "Slow down")
    body = await request.json()
    q, model = str(body.get("message", "")).strip(), body.get("model")
    if not q or len(q) > 4000: raise HTTPException(400, "Invalid message")
    if model not in MODELS: raise HTTPException(400, "Model not allowed")
    if not TOKEN: raise HTTPException(503, "AI not configured")
    hist = db.query(Message).filter_by(user_id=u.id).order_by(Message.id.desc()).limit(20).all()[::-1]
    msgs = [{"role": m.role, "content": m.content} for m in hist] + [{"role": "user", "content": q}]
    db.add(Message(user_id=u.id, role="user", content=q)); db.commit()
    uid = u.id

    async def gen():
        out = []
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10, read=60)) as c:
                async with c.stream("POST", BASE_URL + API_PATH, json={"model": model, "max_tokens": 2048, "stream": True, "messages": msgs},
                                    headers={"x-api-key": TOKEN, "authorization": f"Bearer {TOKEN}", "anthropic-version": "2023-06-01"}) as r:
                    if r.status_code != 200:
                        log.error("provider status %s", r.status_code)
                        yield 'data: ' + json.dumps({"err": "AI provider error. Thodi der baad try karein."}) + '\n\n'; return
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"): continue
                        try: ev = json.loads(line[5:].strip())
                        except ValueError: continue
                        if ev.get("type") == "content_block_delta" and ev.get("delta", {}).get("text"):
                            out.append(ev["delta"]["text"]); yield 'data: ' + json.dumps({"t": ev["delta"]["text"]}) + '\n\n'
        except (httpx.TimeoutException, httpx.TransportError):
            yield 'data: ' + json.dumps({"err": "Response was interrupted."}) + '\n\n'
        finally:
            if out:
                with Session() as s2: s2.add(Message(user_id=uid, role="assistant", content="".join(out))); s2.commit()
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})

@app.get("/admin", response_class=HTMLResponse)
def admin(request: Request, o=Depends(owner_user), db=Depends(get_db)):
    users = db.query(User).order_by(User.id.desc()).all()
    c = hidden(request); rows = ""
    for u in users:
        if u.role == "owner": continue
        btn = lambda a, l: f'<form method=post action="/admin/users/{u.id}/{a}">{c}<button>{l}</button></form>'
        acts = {"pending": [("approve", "Approve"), ("reject", "Reject")], "approved": [("suspend", "Suspend")],
                "rejected": [("approve", "Approve")], "suspended": [("reactivate", "Reactivate")]}[u.status] + [("delete", "Delete")]
        n = db.query(Message).filter_by(user_id=u.id).count()
        rows += f'<div class=card><b>{esc(u.email)}</b><br><small>{u.status} · {n} msgs · {u.created_at:%Y-%m-%d}</small><div class=row>{"".join(btn(a,l) for a,l in acts)}</div></div>'
    logs = "".join(f"<small>{a.created_at:%m-%d %H:%M} {esc(a.actor)}: {esc(a.action)} {esc(a.target or '')}</small><br>" for a in db.query(Audit).order_by(Audit.id.desc()).limit(15))
    total = db.query(Message).count()
    return page("Admin", f'<a href=/chat>← Chat</a><h2>Admin</h2><p>Users: {len(users)} · Messages: {total}</p>{rows or "<p>No users yet.</p>"}<h3>Audit</h3>{logs}')

@app.post("/admin/users/{uid}/{action}")
async def admin_action(uid: int, action: str, request: Request, o=Depends(owner_user), db=Depends(get_db)):
    f = await request.form()
    if not csrf_ok(request, f.get("csrf")): raise HTTPException(403, "CSRF")
    u = db.get(User, uid)
    states = {"approve": "approved", "reject": "rejected", "suspend": "suspended", "reactivate": "approved"}
    if not u or u.role == "owner" or (action not in states and action != "delete"): raise HTTPException(400, "Invalid")
    if action == "delete":
        db.query(Message).filter_by(user_id=u.id).delete(); db.delete(u)
    else: u.status = states[action]
    db.commit(); audit(db, o.email, action, u.email)
    return RedirectResponse("/admin", 303)
