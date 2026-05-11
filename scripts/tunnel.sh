#!/usr/bin/env bash
# Keep the serveo reverse tunnel alive.
#
# serveo is a free service with no SLA — the SSH session drops periodically
# and every user request 502s until someone re-runs the ssh command. This
# wrapper respawns on exit with a short backoff so a flapping serveo doesn't
# burn CPU but a one-off drop recovers within seconds.
#
# Usage:
#   scripts/tunnel.sh                 # forwards localhost:8000
#   scripts/tunnel.sh 9000            # forwards localhost:9000
#   SUBDOMAIN=curemom scripts/tunnel.sh
#
# Stop with Ctrl-C — the trap kills the inner ssh too.

set -uo pipefail

PORT="${1:-8000}"
SUBDOMAIN="${SUBDOMAIN:-curemom}"
BACKOFF_S="${BACKOFF_S:-5}"

trap 'echo "[tunnel] caught signal, exiting…"; kill 0; exit 0' INT TERM

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[tunnel] attempt #${attempt}: ssh -R ${SUBDOMAIN}:80:localhost:${PORT} serveo.net"
  ssh -T \
      -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes \
      -R "${SUBDOMAIN}:80:localhost:${PORT}" \
      serveo.net || true
  echo "[tunnel] ssh exited (rc=$?), reconnecting in ${BACKOFF_S}s…"
  sleep "${BACKOFF_S}"
done
