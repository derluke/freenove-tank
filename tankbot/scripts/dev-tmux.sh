#!/usr/bin/env bash

set -euo pipefail

SESSION_NAME="${1:-tankbot}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed"
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  exec tmux attach -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" -n dev -c "$ROOT_DIR"

# Capture pane ids instead of assuming pane indices start at 0.
PANE_SHELL="$(tmux list-panes -t "$SESSION_NAME:dev" -F '#{pane_id}' | head -n1)"
tmux send-keys -t "$PANE_SHELL" "cd \"$ROOT_DIR\"" C-m

# Pane 1: Phoenix web app
PANE_WEB="$(tmux split-window -h -P -F '#{pane_id}' -t "$PANE_SHELL" -c "$ROOT_DIR")"
tmux send-keys -t "$PANE_WEB" "cd \"$ROOT_DIR\" && task web:start" C-m

# Pane 2: vision process
PANE_VISION="$(tmux split-window -v -P -F '#{pane_id}' -t "$PANE_WEB" -c "$ROOT_DIR")"
tmux send-keys -t "$PANE_VISION" "cd \"$ROOT_DIR\" && task vision:start" C-m

# Pane 3: quick iteration / deployment shell
PANE_OPS="$(tmux split-window -v -P -F '#{pane_id}' -t "$PANE_SHELL" -c "$ROOT_DIR")"
tmux send-keys -t "$PANE_OPS" "cd \"$ROOT_DIR\"" C-m

tmux select-layout -t "$SESSION_NAME:dev" tiled

tmux new-window -t "$SESSION_NAME" -n ops -c "$ROOT_DIR"
PANE_EXTRA="$(tmux list-panes -t "$SESSION_NAME:ops" -F '#{pane_id}' | head -n1)"
tmux send-keys -t "$PANE_EXTRA" "cd \"$ROOT_DIR\"" C-m

exec tmux attach -t "$SESSION_NAME"
