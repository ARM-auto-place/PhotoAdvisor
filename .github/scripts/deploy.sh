#!/usr/bin/env bash
set -euo pipefail

for name in DEPLOY_HOST DEPLOY_USER DEPLOY_PORT DEPLOY_PATH COMPOSE_FILE COMPOSE_PROJECT_NAME COMPOSE_SERVICE SSH_PRIVATE_KEY SSH_KNOWN_HOSTS IMAGE; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing configuration: $name" >&2
    exit 1
  fi
done
[[ "$DEPLOY_HOST" =~ ^[a-zA-Z0-9][a-zA-Z0-9.-]*$ ]]
[[ "$DEPLOY_USER" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*$ ]]
[[ "$DEPLOY_PORT" =~ ^[0-9]+$ ]]
[[ "$DEPLOY_PATH" =~ ^/[a-zA-Z0-9_./-]+$ ]]
[[ "$COMPOSE_FILE" =~ ^[a-zA-Z0-9_./-]+$ ]]
[[ "$COMPOSE_PROJECT_NAME" =~ ^[a-z0-9][a-z0-9_-]*$ ]]
[[ "$COMPOSE_SERVICE" =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]
[[ "$IMAGE" =~ ^ghcr.io/[a-z0-9_./-]+@sha256:[a-f0-9]{64}$ ]]

ssh_dir=$(mktemp -d)
trap 'rm -rf -- "$ssh_dir"' EXIT
chmod 700 "$ssh_dir"
printf '%s\n' "$SSH_PRIVATE_KEY" > "$ssh_dir/key"
printf '%s\n' "$SSH_KNOWN_HOSTS" > "$ssh_dir/known_hosts"
chmod 600 "$ssh_dir/key" "$ssh_dir/known_hosts"

# Arguments are restricted above to shell-safe characters; secrets are never sent as arguments.
ssh -i "$ssh_dir/key" -p "$DEPLOY_PORT" \
  -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o UserKnownHostsFile="$ssh_dir/known_hosts" \
  "$DEPLOY_USER@$DEPLOY_HOST" \
  "bash -s -- '$DEPLOY_PATH' '$COMPOSE_FILE' '$COMPOSE_PROJECT_NAME' '$COMPOSE_SERVICE' '$IMAGE'" <<'REMOTE'
set -euo pipefail
cd -- "$1"
compose_file=$2
project=$3
service=$4
image=$5
test -f "$compose_file"

# Serialize deployments even if different repositories share this Compose project.
exec 9>".codex-deploy-${project}.lock"
flock -w 600 9
docker compose -p "$project" -f "$compose_file" config --services | grep -Fx -- "$service" >/dev/null

override=".deploy-${service}.yaml"
candidate=$(mktemp ".deploy-${service}.XXXXXX.yaml")
trap 'rm -f -- "$candidate"' EXIT
printf 'services:\n  %s:\n    image: %s\n' "$service" "$image" > "$candidate"
compose=(docker compose -p "$project" -f "$compose_file" -f "$candidate")
"${compose[@]}" config --quiet
"${compose[@]}" pull "$service"
"${compose[@]}" up -d --no-deps --no-build --pull never --wait --wait-timeout 180 "$service"
if [[ -f "$override" ]]; then
  cp -- "$override" "${override}.previous"
fi
mv -- "$candidate" "$override"
echo "Deployed $service using $image"
REMOTE
