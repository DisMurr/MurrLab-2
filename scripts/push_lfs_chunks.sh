#!/usr/bin/env bash
set -euo pipefail

# Directory containing model files to push (default is the 20B model dir)
DIR=${1:-gpt-oss/gpt-oss-20b}

# 2 GiB per chunk
CHUNK_BYTES=$((2*1024*1024*1024))

# Only include common model weight extensions; prune cache and VCS dirs
readarray -d '' files < <(\
  find "$DIR" \
    -type d \( -name .git -o -name .cache -o -name cache -o -name downloads -o -name download \) -prune -o \
    -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o -name '*.gguf' \) \
    -print0 \
  | sort -z)

batch=1
bytes=0
pending=()

commit_if_staged() {
  # Commit only if there are staged changes; avoid failing on empty commits
  if ! git diff --cached --quiet; then
    git commit -m "LFS chunk $batch (~$((bytes/1024/1024)) MiB)"
    git push
  fi
}

for f in "${files[@]}"; do
  size=$(stat -c%s "$f")
  if (( bytes + size > CHUNK_BYTES )) && (( ${#pending[@]} > 0 )); then
    git add -- "${pending[@]}"
    commit_if_staged
    batch=$((batch+1))
    bytes=0
    pending=()
  fi
  pending+=("$f")
  bytes=$((bytes + size))
done

if (( ${#pending[@]} > 0 )); then
  git add -- "${pending[@]}"
  commit_if_staged
fi

echo "Done."
