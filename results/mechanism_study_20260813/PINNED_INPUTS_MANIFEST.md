# Pinned Inputs Manifest — mechanism_study_20260813

Provenance-preservation task (2026-08-14), commission findings: pinned inputs consumed by
`darksiren_emri/validation/venue_transfer.py` and `darksiren_emri/validation/closed_loop_gfrac.py`
are git-untracked. This manifest commits their checksums at the time of writing so a future
mismatch is detectable even though the files themselves are not (and, for the large ones, should
not be) tracked in git.

The two hardcoded md5 constants found by grepping the code
(`darksiren_emri/validation/venue_transfer.py:224-226`, the V-T3 pin-integrity block) are
`CRB_CSV_MD5` and `FROZENG_EMIT_MD5`. `PRUNED_CATALOGUE_CSV` and `DEFAULT_INJECTION_DIR`
(`closed_loop_gfrac.py:143-147`) are referenced by path only — the code checks
`os.path.isfile`/`os.path.isdir` for these, not a checksum — so there is no in-code md5 constant
to compare against for those two.

## Table

| # | Input | Path | md5 | Bytes | Matches in-code constant? |
|---|-------|------|-----|-------|---------------------------|
| 1 | Prepared Cramér–Rao-bounds CSV | `results/run_20260804_postfix/iiib/diagnostics/prepared_cramer_rao_bounds.csv` | `9a1f2a14384a9281c97ca3be312ddaab` | 4,216,260 | **Yes** — matches `CRB_CSV_MD5` (`venue_transfer.py:224`) |
| 2 | Frozen-alpha emit JSON (`FROZENG_EMIT_JSON`, h=0.73 posterior) | `results/run_20260804_frozeng/iiib/posteriors_with_bh_mass/h_0_73.json` | `34c50e91028b6a6458a2b145db545705` | 86,162,412 | **Yes** — matches `FROZENG_EMIT_MD5` (`venue_transfer.py:226`) |
| 3 | Pruned GLADE catalogue (`PRUNED_CATALOGUE_CSV`) | `results/campaign51_20260728/realistic_20260729/realizations_staged/cluster_parent_reduced_galaxy_catalogue.csv` | `c52c13b5cab61f6b3f04bbe202550969` | 1,681,954,844 | No in-code md5 constant exists (path-only pin, `venue_transfer.py:227-230`; existence-checked at `venue_transfer.py:2169`) — recorded here for the first time |
| 4 | Injection pool (`DEFAULT_INJECTION_DIR`, `injection_pool_mix200k_20260728/`) | `results/campaign51_20260728/realistic_20260729/gate_b_20260730/injection_pool_mix200k_20260728/` (707 files) | aggregate `62250bef7b1f5b9d5ccef9992083306d` (md5 of the sorted `md5sum`-per-file listing; per-file digests in `injection_pool_mix200k_20260728_filelist_md5.txt` alongside this manifest) | 39,516,039 (total, `du -sb`) | No in-code md5 constant exists — it is a directory, checked only with `os.path.isdir` (`closed_loop_gfrac.py:143-144`, `venue_transfer.py:2179`) — recorded here for the first time |

Inputs 1 and 2 reproduce their registered code-side pins **exactly** — no mismatch found.
Inputs 3 and 4 have no prior code-side pin to compare against; this manifest is their first
committed checksum and becomes the reference point for future drift detection.

## Method

```bash
md5sum <file>
stat -c%s <file>          # bytes, single file
du -sb <dir>               # bytes, directory total
find <dir> -type f -exec md5sum {} \; | sort -k2 | md5sum   # aggregate dir digest
```

## Note

These inputs are git-untracked; this manifest commits their checksums. Archival to persistent
storage (bwHPC workspace expiry risk) remains an open action — see `.commission-notes.md`.
