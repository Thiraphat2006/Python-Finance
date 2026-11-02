Create the file .github/skills/handoff/SKILL.md with exactly this content, then change nothing else:

---
name: handoff
description: Write handoff.md after finishing a task so the user can attach one file to Claude for review.
---
After completing a task, write `handoff.md` in the repo root (UTF-8) with these sections:
1. Summary: max 5 lines, what changed and why.
2. `git status --short` output.
3. `git --no-pager diff --stat`, then `git --no-pager diff HEAD` (covers staged + unstaged).
   If the diff is over ~400 lines, include the full diff only for logic files and stat only for the rest.
4. Tail of `make quality` output + test count + coverage. On failure, the full error.
5. Open questions / anything uncertain.
Rules:
- Paste the real diff; never replace it with a prose summary.
- Never include secrets or .env content.
- Build the file with PowerShell, e.g. (git status --short; git --no-pager diff HEAD) | Set-Content handoff.md -Encoding utf8, then append the summary and test results.
- Do not commit handoff.md.

Success check: .github/skills/handoff/SKILL.md exists and git status shows only that new file.