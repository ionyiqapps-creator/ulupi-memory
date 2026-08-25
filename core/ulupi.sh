#!/bin/zsh
# ulupi.sh — shortcut entry: ensure ulupi daemon up, then query or store.
#   ./core/ulupi.sh "question"     -> persona + recall context
#   ./core/ulupi.sh add "q" "a"    -> store memory
PORT=8792
BASE="$HOME/Desktop/om-memory"
PY="${HOME}/Desktop/memOTry/.venv/bin/python"
[ -x "$PY" ] || PY=python3

if ! curl -sm 1 "http://127.0.0.1:$PORT/ping" >/dev/null 2>&1; then
  nohup "$PY" "$BASE/core/server.py" >/tmp/ulupi-server.log 2>&1 &
  for i in {1..20}; do
    curl -sm 1 "http://127.0.0.1:$PORT/ping" >/dev/null 2>&1 && break
    sleep 0.3
  done
fi

emit() {
  if [ -n "$2" ]; then
    printf '%s\n%s:\n%s\n' "$(cat "$BASE/config/persona.md")" "$1" "$2"
  else
    cat "$BASE/config/persona-plain.md"
  fi
}

# announce() removed — was speaking on every step ("searching my memory, sir")
announce() { :; }

if [ "$1" = "add" ]; then
  a="$3"
  case "$a" in
    *.rtf|*.txt) [ -f "$a" ] && a=$(textutil -convert txt -stdout "$a" 2>/dev/null || echo "") ;;
  esac
  R=$(curl -s "http://127.0.0.1:$PORT/add" --data-urlencode "q=$2" --data-urlencode "a=$a")
  [ "$R" = "dupe" ] && announce "Already knew that one, sir" || announce "Committed to memory, sir"
  echo "$R"
else
  announce "Searching my memory, sir"
  R=$(curl -sG "http://127.0.0.1:$PORT/" --data-urlencode "q=$1")
  emit "Retrieved memory (ulupi)" "$R"
fi
