#!/usr/bin/env bash
set -euo pipefail

ROOT="models/ActionBiasBench/out"
DELETE=0
LIST=0

usage() {
  cat <<'EOF'
Usage:
  scripts/cleanup_model_weights.sh [--root PATH] [--list] [--delete]

Find model weight/checkpoint files under ActionBiasBench out and optionally delete them.

Default mode is a dry run: it reports the number of files and total disk usage.
Use --list to print matching paths.
Use --delete to actually remove matching files.

Matched extensions:
  .pt .pth .ckpt .bin .safetensors

Examples:
  scripts/cleanup_model_weights.sh --root out
  scripts/cleanup_model_weights.sh --root out --list
  scripts/cleanup_model_weights.sh --root out --delete
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || { echo "--root requires a path" >&2; exit 2; }
      ROOT="$2"
      shift 2
      ;;
    --list)
      LIST=1
      shift
      ;;
    --delete)
      DELETE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -d "$ROOT" ]] || { echo "Root does not exist: $ROOT" >&2; exit 1; }

find_weights() {
  find "$ROOT" -type f \( \
    -name '*.pt' -o \
    -name '*.pth' -o \
    -name '*.ckpt' -o \
    -name '*.bin' -o \
    -name '*.safetensors' \
  \) -print0
}

count=0
bytes=0

while IFS= read -r -d '' file; do
  count=$((count + 1))
  if size=$(stat -f '%z' "$file" 2>/dev/null); then
    bytes=$((bytes + size))
  elif size=$(stat -c '%s' "$file" 2>/dev/null); then
    bytes=$((bytes + size))
  fi
  if [[ "$LIST" == "1" ]]; then
    printf '%s\n' "$file"
  fi
done < <(find_weights)

human_size=$(awk -v bytes="$bytes" 'BEGIN {
  split("B KiB MiB GiB TiB", u, " ");
  size = bytes;
  i = 1;
  while (size >= 1024 && i < 5) { size /= 1024; i++ }
  printf "%.2f %s", size, u[i];
}')

echo "Root: $ROOT"
echo "Matched model weight files: $count"
echo "Total size: $human_size"

if [[ "$DELETE" != "1" ]]; then
  echo "Dry run only. Re-run with --delete to remove these files."
  exit 0
fi

if [[ "$count" -eq 0 ]]; then
  echo "Nothing to delete."
  exit 0
fi

echo "Deleting matched model weight files..."
deleted=0
while IFS= read -r -d '' file; do
  rm -f -- "$file"
  deleted=$((deleted + 1))
done < <(find_weights)

echo "Deleted files: $deleted"
