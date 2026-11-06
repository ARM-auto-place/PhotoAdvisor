#!/bin/sh
set -eu
: "${MINIO_ROOT_USER:?Set a local MinIO administrator name}"
: "${MINIO_ROOT_PASSWORD:?Set a local MinIO administrator password}"
exec minio "$@"
