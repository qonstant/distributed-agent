## Admin Panel

FastAPI admin panel for user access, conversations, usage events, and audit actions.

### Local run

```bash
cp .env.example .env
docker-compose --profile local up -d --build
docker-compose run --rm web python scripts/check_schema.py
docker-compose run --rm web python scripts/create_admin.py
```

Open `http://localhost:3910/admin-ui/login`.

By default, Docker binds the admin panel to `127.0.0.1`, so it is reachable only
from the machine running Docker.

To open it from your laptop when it is deployed on a server, use an SSH tunnel:

```bash
ssh -L 3910:127.0.0.1:3910 <user>@<server>
```

Then open `http://localhost:3910/admin-ui/login` locally.

### GitLab CI/CD variables

Production:

- `ADMIN_DATABASE_URL`
- `ADMIN_SECRET_KEY`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`
- `ADMIN_TELEGRAM_ID`
- `ADMIN_BIND_HOST` optional, defaults to `127.0.0.1`
- `ADMIN_HOST_PORT` optional, defaults to `3910`
- `ADMIN_CORS_ORIGINS` optional, comma-separated, defaults to `*`
- `ADMIN_COOKIE_SECURE` optional, defaults to `true` in CI deploy

Test branch equivalents use `_TEST` suffix:

- `ADMIN_DATABASE_URL_TEST`
- `ADMIN_SECRET_KEY_TEST`
- `ADMIN_USERNAME_TEST`
- `ADMIN_PASSWORD_TEST`
- `ADMIN_TELEGRAM_ID_TEST`
- `ADMIN_BIND_HOST_TEST` optional, defaults to `127.0.0.1`
- `ADMIN_HOST_PORT_TEST` optional
- `ADMIN_CORS_ORIGINS_TEST` optional
- `ADMIN_COOKIE_SECURE_TEST` optional, defaults to `true` in CI deploy

`ADMIN_DATABASE_URL` must use SQLAlchemy's asyncpg dialect, for example:

```text
postgresql+asyncpg://postgres:<password>@10.66.66.1:5432/postgres
```

The admin panel uses the same database schema as the Go bot. Deploy validates
that the Go migrations have already created the required tables, then creates or
promotes the configured admin user. It does not own database migrations. If the
schema needs to change, add the migration under `golang/db/migrations`.

`ADMIN_TELEGRAM_ID` should be the real Telegram user ID of the admin account.
The deploy script promotes that existing user if present, or creates an admin
row with that Telegram ID if it does not exist yet. `ADMIN_USERNAME` and
`ADMIN_PASSWORD` are admin panel login credentials; `ADMIN_USERNAME` does not
need to match the Telegram username stored in the `users` table.

Keep `ADMIN_BIND_HOST=127.0.0.1` unless you intentionally put the admin panel
behind a private reverse proxy, VPN, or firewall. Setting it to `0.0.0.0` exposes
the port on the server network interface.

### LightRAG artifact builds

The admin panel includes `/admin-ui/lightrag` for starting LightRAG artifact
generation without interrupting the currently active artifacts. The build flow is:

1. Download source PDFs/docs from `LIGHTRAG_SOURCE_PREFIX` in the vectors bucket.
2. Convert them to markdown in a temporary build directory.
3. Upload generated markdowns to `LIGHTRAG_MARKDOWN_S3_PREFIX`.
4. Generate LightRAG artifacts.
5. Upload LightRAG artifacts to a staged release and update the LightRAG pointer.

Two build actions are available:

- `Continue Build` runs `LIGHTRAG_BUILD_CONTINUE_COMMAND`, intended to reuse
  existing local state and resume an incomplete LightRAG index.
- `Generate From Beginning` runs `LIGHTRAG_BUILD_FULL_COMMAND`, intended to reset
  local source, markdown, and LightRAG state and rebuild from scratch.

After a successful build, the admin panel uploads every file from
`LIGHTRAG_WORK_DIR` into the vectors bucket under:

```text
<LIGHTRAG_S3_PREFIX>/releases/<build_id>/*
```

Then it updates this pointer:

```text
<LIGHTRAG_S3_PREFIX>/releases/current
```

That pointer update is the promotion step. Existing users keep using the previous
artifacts while the new build is running.

The admin container must be able to run the configured build commands. In
production, either mount the repository into `LIGHTRAG_REPO_DIR` or set the
commands to call a separate build runner.

The Docker image installs LightRAG into `/opt/lightrag-venv` instead of the admin
app environment. This avoids mixing LightRAG's Pydantic v2 dependencies with the
admin app's Pydantic v1 runtime.
