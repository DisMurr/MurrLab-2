#!/usr/bin/env bash
set -euo pipefail

DIR=${1:-gpt-oss/gpt-oss-20b}

mapfile -t parts < <(find "$DIR" -type f -name '*.part*' | sort)

declare -A groups
for p in "${parts[@]}"; do
  base=${p%.part*}
  groups["$base"]=1
done

for base in "${!groups[@]}"; do
  echo "joining: $base"
  cat "$base".part* > "$base"
  rm -f "$base".part*
done

echo "Join complete."
