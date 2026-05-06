#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 <env> <site> [<site> ...]" >&2
  echo "Example: $0 prod nomadmit.com admin.nomadmit.com s3.nomadmit.com" >&2
}

if [ "$#" -lt 2 ]; then
  usage
  exit 1
fi

env_name="$1"
shift

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sites_dir="${script_dir}/${env_name}/sites-available"

if [ ! -d "${sites_dir}" ]; then
  echo "Missing sites directory: ${sites_dir}" >&2
  exit 1
fi

for site in "$@"; do
  src="${sites_dir}/${site}.conf"
  dst="/etc/nginx/sites-available/${site}"
  link="/etc/nginx/sites-enabled/${site}"

  if [ ! -f "${src}" ]; then
    echo "Missing repo config: ${src}" >&2
    exit 1
  fi

  echo "[nginx] installing ${site}"
  sudo cp "${src}" "${dst}"
  sudo ln -sf "${dst}" "${link}"
done

echo "[nginx] testing configuration"
sudo nginx -t

echo "[nginx] reloading service"
sudo systemctl reload nginx

echo "[nginx] active server_name entries"
sudo nginx -T | grep -n "server_name" || true
