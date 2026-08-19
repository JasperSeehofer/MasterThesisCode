# History rewrite 2026-08-19 — excision of oversized cluster-staging CSVs

Author-approved (2026-08-19, "option 2"): the two multi-GB cluster-staging CSVs
accidentally swept into commit 68543b2f (row #122 ledger commit, 2026-08-18) were
removed from the 16 unpushed commits via git filter-branch --index-filter.
GitHub's 100 MB pre-receive limit had blocked all pushes since 2026-08-18.

The files (untouched on disk, gitignored henceforth; seed-reproducible, provenance
in the committed observed_catalogue_seed900001.meta.json):
- results/campaign51_20260728/realistic_20260729/realizations_staged/cluster_parent_reduced_galaxy_catalogue.csv (1.6 GB)
- results/campaign51_20260728/realistic_20260729/realizations_staged/observed_catalogue_seed900001.csv (2.4 GB)

Verification: old-tip vs new-tip tree diff contains ONLY the two CSV deletions;
authors/dates/messages preserved; backup refs: branch backup-pre-excise-20260819 +
refs/original/refs/heads/main (prune after the push is confirmed).

Old → new commit map (any hash cited in ledgers/preregs before 2026-08-19 resolves
via this table; notably the prod2d freeze commit d6fc1ccf → 26bcd9a4 — the cluster
tag prod2d-closure-base is re-pointed to 26bcd9a4 after job 6364821 completes):

```
6f3a8c3a6f8792dec3e500d2c109dacb9b23b5c7 6f3a8c3a6f8792dec3e500d2c109dacb9b23b5c7
68543b2fbecc283600834e3d7eea10c9fbf66362 5b56c999e81f2476f654357e1326f6d5aa35a16f
419b1b7e76e6a43be2fb2bb795ac3cd1d0e1286c 2fa8f1343dccecc4eef0cdaf1f694887363d172a
f6e24fa25fc8d19e9cc6520997e3103f97341dec b93343271783bdb096b3eb9586e4607ee79701fc
15473cb0a752a9760f680450c5e9807cc266b9c8 d050dda3c51d9c3d8e9827c6a8deb504cf4fe912
70ccc49c7baa34de13e954bc56d375a30c52e60e e4c00e92a3989c215e589a2d219d6f0960112cb6
6504c8b982614ed85e8b85dc3d4c615be6acabb3 6cfa87011b748a970ba8b018c3018b585312a403
4dd822adbef3046bc65e25eb78e702ccc5cf4997 c1fd28ac310861af43a21fa5fdd4f6136e476d23
00d853ac0a3e7cc875e08fdf3b197874d7975592 8f181651f44d809e501060a5ea9cf70a59af2d54
69152e4e13d4ada17e0560e558d6273cf42280a0 8b54f87ae7ae7fe1065e3cfbedaed208786f3f0d
d6fc1ccf168a8a5bf52595b8aed13ff0e7626cfd 26bcd9a493bba41a4f85e61fd847b283d88d7703
a9d157a7a89cf23f310896ff97fc4a0c61be6268 cdc0b6343bc0619980687a43617d7d726dd61179
c8c0bdc1f33af5fdfa4feb1e32580a649c337780 92512fcf95db9a3bb313a49d5a2191074fdb45a7
5dc882809eece755b3aa1f821e772fd97508bf65 ee0238934deade24611235d2273dde4f66646d5b
58202ccc4104f1ab0ad585ccc5413c867ade7059 67d7907220b41431cc2d57f93d9902137d6ab36c
c9d96a5791fb83bc14df16857aacb5821ef69645 0203af30aaac69e622ca1e6a326bd8f3b696b079
```
