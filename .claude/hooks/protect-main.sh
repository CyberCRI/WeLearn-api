#!/usr/bin/env bash
# PreToolUse hook (Bash): block git commit / push that would land on main, and force-pushes.
# Reads the hook JSON on stdin; prints a deny decision when the command is not allowed.

cmd=$(jq -r '.tool_input.command // ""')

deny() {
  jq -n --arg r "$1" '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
  exit 0
}

# Only inspect commands that invoke git commit or git push
echo "$cmd" | grep -Eq '(^|[;&|[:space:]])git([[:space:]]+-[^[:space:]]+)*[[:space:]]+(commit|push)\b' || exit 0

branch=$(git -C "${CLAUDE_PROJECT_DIR:-.}" rev-parse --abbrev-ref HEAD 2>/dev/null)

if echo "$cmd" | grep -Eq '\bgit\b.*\bcommit\b' && [ "$branch" = "main" ]; then
  deny "Committing on main is not allowed (CLAUDE.md). Create a feature branch first."
fi

if echo "$cmd" | grep -Eq '\bgit\b.*\bpush\b'; then
  if echo "$cmd" | grep -Eq '(--force|--force-with-lease|[[:space:]]-f\b|[[:space:]]\+[^[:space:]]+)'; then
    deny "Force-push is not allowed (CLAUDE.md)."
  fi
  if [ "$branch" = "main" ] || echo "$cmd" | grep -Eq '([[:space:]:]|^)main\b'; then
    deny "Pushing to main is not allowed (CLAUDE.md). Changes reach main only through a reviewed pull request."
  fi
fi

exit 0
