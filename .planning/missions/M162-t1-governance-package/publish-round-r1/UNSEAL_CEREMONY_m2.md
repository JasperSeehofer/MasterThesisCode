# (m2) sealed-draw and unseal ceremony: launch row, blind phase, unseal

Date: 2026-10-19. Mission M162. Author of record: Jasper Seehofer.
**Status: DRAFT FOR RATIFICATION**, together with `ERRATUM_m2_R27_R28.md`. It replaces the
"DESIGN" of `REGISTRATION_DRAFT.md` §4 items 2–4 and the dossier's §3 item 3. Nothing here has
run, and no part of it has been ratified.

The anti-tuning guarantee is the **code freeze**, not the secrecy of the number (draft §4). The
commitment proves when h_inj was fixed relative to the frozen code. The procedure keeps the
reading agents blind until the cell is mechanical.

## Roles

| role | who | may open sealed material? |
|---|---|---|
| drawer | the author, or a script run by the author on the cluster login node | creates `~/.sealed/t1-m2/SEALED_H_INJ.json`; never prints it |
| ops agent (one per batch, `cluster/agent_ssh.sh`) | sonnet | **no**. Submits, then monitors by `squeue`/`sacct` only, and retrieves the whitelisted outputs only |
| pre-unseal reader | fresh sonnet session | **no**. Reads the whitelist of erratum E-6 only |
| unsealer | the author | yes, at step U-1 |
| unseal read (cell assignment) | top-tier decisive verifier (draft §11 item 7, ratified-and-flagged at dossier §1 item 7) | yes, after U-1 |

`draw_sealed_h.py` refuses to `draw` or `unseal` when it detects a Claude Code session
(`CLAUDECODE` / `CLAUDE_CODE_ENTRYPOINT`). The refusal is procedural, not cryptographic: it stops
a mistaken agent, not an adversary.

## Phase L: launch, before any sealed number exists

- L-0. **Preconditions, all recorded:**
  - this erratum ratified, with its rule-block hashes in THREAD.md;
  - the M65 landing done (main reaches the work branch), and H_code chosen on main with
    `H_code:darksiren_emri` == `71acb98b…`;
  - the pin re-read R-1..R-3 green (`POOL_OPTION_MEMO_m2.md` §5);
  - P2 run and gated (E-5), or P1 chosen;
  - `/cluster` preflight `VERDICT: READY ✓`;
  - the reader script and the sealed submit/sbatch committed, with their blob hashes noted.
- L-1. **Draw** (author):
  `python scripts/sealed_mock/draw_sealed_h.py draw --erratum <path> --params-sha256 <THREAD.md value> --h-code <H_code>`
  The output is the public receipt `{commitment, params_sha256, h_code, utc}`. h_inj and the salt
  are never on screen.
- L-2. **Launch ledger row** (`BIAS_HISTORY_LEDGER.md`), appended before any submission. It
  quotes the receipt, the prior (P-A), the pool option and its gate value, H_code, the tree, the
  reader and sbatch blob hashes, the RUN_DIR named by the commitment prefix (for example
  `$WS/t1_m2_<commitment[:12]>`), and the job IDs as they are created. This fills slots E-9.
- L-3. **Submit.** The sealed submit script reads the sealed file at submit time, exports
  `H_VALUE` to the simulate array, and gives every job a name free of h. The script is outside
  `darksiren_emri/` and is written by the T-1 mission, not by M162.

## Phase B: blind, between the draw and the unseal

- B-1. **Monitoring** by `squeue` / `sacct` only (state, elapsed, exit code). **No agent opens**
  `RUN_DIR/logs`, `run_metadata_*`, `provenance_*`, any CRB, any simulate `.out`/`.err`, or
  `~/.sealed/`.
- B-2. A failed task is diagnosed by its exit code and `sacct` fields only. If diagnosis needs a
  log, the author reads it, or the read waits for the unseal; an agent never reads it.
  Resubmitting the same array with the same H_code and seed is allowed. Changing any frozen
  input is not: it spends the seal, and the read becomes NO-READ.
- B-3. **Retrieval** copies the whitelist only: `simulations/posteriors*/h_*.json` and
  `simulations/diagnostics/event_likelihoods.csv`, with a checksum manifest. Everything else
  stays on the cluster until U-1.
- B-4. **Pre-unseal read.** The reader computes, per channel: mean_h, MAP, σ_h, m_lo, m_hi,
  N_used, ρ ingredients and the T-4 pulls. It then writes `PRE_UNSEAL_MOMENTS_m2.json` and
  commits it. Its sha256 goes into a ledger row **before U-1**. These numbers fix every input
  of the cell except h_inj.

## Phase U: unseal

- U-1. **Unseal (author, on the cluster login node where `~/.sealed/` lives).**
  `python scripts/sealed_mock/draw_sealed_h.py unseal --commitment <launch-row commitment>`
  prints `{h_inj, salt, commitment_launch_row, commitment_recomputed, match}`. The author pastes
  it into the **unseal ledger row** together with the `PRE_UNSEAL_MOMENTS_m2.json` sha256.
- U-2. **G-3, by anyone, from the quoted values:**
  `python scripts/sealed_mock/draw_sealed_h.py verify --h-inj <6dp> --salt <hex> --commitment <launch-row value>`
  must print `match: true`.
- U-3. **Ceremony gates** (erratum E-6, second layer), now that sealed-side files may be opened:
  - G-1 pins: the catalogue md5, the pool count and list-md5, and the sealed CRB md5 recorded at
    merge;
  - G-2 freeze: per task, HEAD == H_code, tree == `71acb98b…`, dirty count 0, read from
    `provenance_*.json`;
  - G-4 blindness audit: grep every agent-written record between L-2 and U-1 for h_inj to 3
    decimals, which must return nothing;
  - g-population: CRB rows, N_used, the simulate yield, and the draft §5 SNR-loss expectation.
  Any red ⇒ NO-READ.
- U-4. **Cell assignment** (top-tier verifier). This is mechanical: erratum E-2, then E-3, on
  the committed moments with h_inj substituted. The verifier re-derives the moments from the
  retrieved files and checks them against `PRE_UNSEAL_MOMENTS_m2.json`. A mismatch is a FIX, not
  a re-read. The verifier writes the cell, its sub-reason, every band distance and both ρ values
  into the unseal row.
- U-5. **Disposition.** The erratum E-8 row for the cell goes to the author as a fresh [RULE]. No
  consequence (HALT, VERIFIED stamp, a kill-criterion firing) executes without that ruling.
- U-6. **Record.** THREAD.md "Active experiment" and "Current claims" are updated the same day.
  The vault row EXP-MTC-T1 gets its Result and Verdict. `hypothesis-ledger` and atoms are refreshed
  by proposal.

## Failure handling

| event | consequence |
|---|---|
| draw refused (params hash, freeze, prior, agent env, file exists) | nothing drawn; fix the precondition; this does not use up the one-draw cap |
| sealed file lost before U-1 | the commitment can no longer be opened ⇒ G-3 red ⇒ NO-READ; the seal is spent |
| an agent opened forbidden material before U-1 | G-4 red ⇒ NO-READ; the seal is spent; recorded as a blindness breach |
| a frozen input changed mid-run | NO-READ; the seal is spent |
| no unseal row by 2026-12-15 (Europe/Berlin) | kill criterion 3 fires (erratum E-7) |
