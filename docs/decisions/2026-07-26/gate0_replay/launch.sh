#!/bin/sh
cd /Users/kirubeso.r/Documents/PhD/gate0_replay
/opt/homebrew/bin/codex exec -m gpt-5.6-sol -c model_reasoning_effort=xhigh --sandbox workspace-write --skip-git-repo-check -o /Users/kirubeso.r/Documents/PhD/gate0_replay/EXECUTION_LOG.md - < /Users/kirubeso.r/Documents/PhD/gate0_replay/TASK.md
touch /tmp/gate0_done
