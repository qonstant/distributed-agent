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
- `ADMIN_BIND_HOST` optional, defaults to `127.0.0.1`
- `ADMIN_HOST_PORT` optional, defaults to `3910`
- `ADMIN_CORS_ORIGINS` optional, comma-separated, defaults to `*`

Test branch equivalents use `_TEST` suffix:

- `ADMIN_DATABASE_URL_TEST`
- `ADMIN_SECRET_KEY_TEST`
- `ADMIN_USERNAME_TEST`
- `ADMIN_PASSWORD_TEST`
- `ADMIN_BIND_HOST_TEST` optional, defaults to `127.0.0.1`
- `ADMIN_HOST_PORT_TEST` optional
- `ADMIN_CORS_ORIGINS_TEST` optional

`ADMIN_DATABASE_URL` must use SQLAlchemy's asyncpg dialect, for example:

```text
postgresql+asyncpg://postgres:<password>@10.66.66.1:5432/postgres
```

The admin panel uses the same database schema as the Go bot. Deploy validates
that the Go migrations have already created the required tables, then creates or
promotes the configured admin user. It does not run admin-owned Alembic
migrations against the shared bot database.

Admin Alembic migrations are disabled by default with a runtime guard. The
admin app should not create, drop, or alter tables in deployed environments.

Keep `ADMIN_BIND_HOST=127.0.0.1` unless you intentionally put the admin panel
behind a private reverse proxy, VPN, or firewall. Setting it to `0.0.0.0` exposes
the port on the server network interface.
