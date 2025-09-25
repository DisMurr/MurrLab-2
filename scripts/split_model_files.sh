#!/usr/bin/env bash
set -euo pipefail

DIR=${1:-gpt-oss/gpt-oss-20b}
MAX=$((2*1024*1024*1024)) # 2 GiB per part

shopt -s globstar nullglob
mapfile -t files < <(find "$DIR" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o -name '*.gguf' \))

for f in "${files[@]}"; do
  size=$(stat -c%s "$f")
  if (( size <= MAX )); then
    echo "skip (<=2GiB): $f"
    continue
  fi
  echo "splitting: $f (size=$size)"
  split -b "$MAX" -d -a 3 -- "$f" "$f.part"
  rm -f -- "$f"
done

echo "Split complete."
