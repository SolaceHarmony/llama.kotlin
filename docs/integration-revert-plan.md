## Integration branch revert plan (copilot/fix-45)

Purpose: capture what was merged into `copilot/fix-45` and provide a safe, repeatable procedure to roll back merges that correspond to closed (not merged) PRs, if needed.

Updated: 2025-08-18 (UTC)

### Current PR status summary

Copilot PRs targeting this workstream:

- OPEN: #47 (head=copilot/fix-45), #46 (head=copilot/fix-43), #44 (head=copilot/fix-42), #41 (head=copilot/fix-40), #39 (head=copilot/fix-38)
- MERGED: #54 (copilot/fix-36), #53 (copilot/fix-51), #52 (copilot/fix-50), #49 (copilot/fix-48), #35 (copilot/fix-34)
- CLOSED without merge: none detected as of 2025-08-18

Conclusion: there are currently no closed-unmerged Copilot PRs to roll back. If that changes, follow the steps below.

### Merge commits included in copilot/fix-45

These are the merge commits we introduced while consolidating branches into `copilot/fix-45`:

- ba214cd6: merge: integrate copilot/fix-43 into copilot/fix-45  (PR #46)
- b781be40: merge: unify Copilot branches into copilot/fix-45   (includes copilot/fix-42 / PR #44)
- 9152f197: Merge branch 'copilot/fix-40' into copilot/fix-45    (PR #41)
- 11398aa8: Merge copilot/fix-38 into copilot/fix-45             (PR #39)

Note: b781be40 groups multiple changes and may include additional edits beyond a single branch; prefer targeted reverts where possible.

### How to revert a merged branch safely

Reverting a merge commit preserves history while undoing its diff. Use the first-parent (mainline) when reverting merges into `copilot/fix-45`.

Recommended order: newest to oldest to minimize conflicts.

Steps:

1) Ensure your working tree is clean and you are on `copilot/fix-45`.
2) Revert the merge commit(s) with mainline parent 1 and an explicit message:

   Optional commands (run one per merge to revert):

   - git revert -m 1 ba214cd6 --no-edit
   - git revert -m 1 b781be40 --no-edit
   - git revert -m 1 9152f197 --no-edit
   - git revert -m 1 11398aa8 --no-edit

3) Resolve any conflicts if prompted, then continue the revert and commit.
4) Build locally to verify the integration branch still compiles.
5) Push the branch and update PR #47 with a brief note on what was reverted and why.

Tips:

- If a merge commit message is ambiguous (e.g., grouped merges), identify the exact merge by looking at `git log --merges --oneline` and/or using `git show <sha>`.
- To revert only some files from a merge, consider a targeted revert: `git checkout <pre-merge-sha> -- path1 path2`, then commit with context explaining why.
- If reverts collide with subsequent commits, you may need to revert-and-apply back the intended parts manually (surgical cherry-picks).

### Verification checklist after a revert

- Build: `./gradlew assemble` completes or fails only on known unrelated issues
- No unresolved conflict markers remain
- Key files touched by the revert behave as expected on a quick smoke test
- PR #47 description updated to reflect the rollback

### Appendix: provenance references

Reference log snapshot around the integration work (for quick lookup):

- ba214cd6 (merge copilot/fix-43)
- b781be40 (unify Copilot branches → includes copilot/fix-42)
- 9152f197 (merge copilot/fix-40)
- 11398aa8 (merge copilot/fix-38)
