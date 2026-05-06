This folder stores Nginx configuration snapshots and repo-managed site configs.

Layout:

- `test/` mirrors the currently working test VPS configuration
- `prod/` contains production-ready configs derived from the test setup

Suggested apply flow on a server:

```bash
sudo cp infra/nginx/<env>/sites-available/<site>.conf /etc/nginx/sites-available/<site>
sudo ln -sf /etc/nginx/sites-available/<site> /etc/nginx/sites-enabled/<site>
sudo nginx -t
sudo systemctl reload nginx
```

Example:
```bash
sudo cp infra/nginx/prod/sites-available/admin.nomadmit.com.conf /etc/nginx/sites-available/admin.nomadmit.com
sudo ln -sf /etc/nginx/sites-available/admin.nomadmit.com /etc/nginx/sites-enabled/admin.nomadmit.com

sudo cp infra/nginx/prod/sites-available/nomadmit.com.conf /etc/nginx/sites-available/nomadmit.com
sudo ln -sf /etc/nginx/sites-available/nomadmit.com /etc/nginx/sites-enabled/nomadmit.com

sudo cp infra/nginx/prod/sites-available/s3.nomadmit.com.conf /etc/nginx/sites-available/s3.nomadmit.com
sudo ln -sf /etc/nginx/sites-available/s3.nomadmit.com /etc/nginx/sites-enabled/s3.nomadmit.com

sudo nginx -t
sudo systemctl reload nginx
```

Then issue certificates:

```bash
sudo certbot --nginx -d admin.nomadmit.com
sudo certbot --nginx -d nomadmit.com
sudo certbot --nginx -d s3.nomadmit.com
```

Certificates and private keys are intentionally not stored in git.

CI/CD:

- GitLab can apply the repo-managed site files with:
  - `bash infra/nginx/deploy.sh test admin.nomadmit.dev nomadmit.dev s3.nomadmit.dev`
  - `bash infra/nginx/deploy.sh prod admin.nomadmit.com nomadmit.com s3.nomadmit.com`
- The runner user must have passwordless `sudo` for:
  - `cp`
  - `ln`
  - `nginx -t`
  - `systemctl reload nginx`
- TLS certificate issuance remains a separate step:
  - `sudo certbot --nginx -d ...`
