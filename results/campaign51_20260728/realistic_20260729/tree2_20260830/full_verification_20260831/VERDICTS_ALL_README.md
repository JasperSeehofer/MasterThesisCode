# verdicts_all.json — RECOVERY NOTE (2026-08-31)

`verdicts_all.json` holds 1 of 42 verdict records (T1-1) — a lost-update race between the 42
parallel opus verifier writers clobbered the rest (mtime evidence in DEDUP_CONFLICTS.md §0).
**Do not treat that file as the 42-item verification record.** The records of record are the two
adjudication reports in this directory (FULL_VERIFICATION_TREE1_20260831.md,
FULL_VERIFICATION_TREE2_DECISIONS_20260831.md), whose adjudicators reconstructed every verdict
from the saved per-item numeric artifacts under work/ plus foreground re-execution of the lost
items' scripts. The 2.4 GB work/ evidence directory (per-item re-derivation scripts + outputs) is LOCAL-ONLY (gitignored — it broke the push at 3.68M insertions); every decisive number it holds is quoted in the two reports. Process fix adopted for future passes (tree-1 report G-1): one verdict file PER
verifier, merged by the collector — never a shared read-modify-write array.
