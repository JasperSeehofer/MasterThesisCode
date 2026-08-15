"""Registered null checks AR-1/AR-2/AR-3 for the split-dose arms.

Companion to ``results/mechanism_study_20260813/ARMS.md`` (registered
2026-08-13). These tests are the auditable form of prereg §5 V-M2: the arms
differ by a *mask*, never by a *draw*.
"""

import numpy as np
import pytest

from darksiren_emri.validation.venue_transfer import _apply_dose_mask


@pytest.fixture
def masking_inputs() -> tuple[
    np.ndarray[tuple[int], np.dtype[np.bool_]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    """A three-event ball with hosts scattered through it, as the lexsort leaves them."""
    host_mask = np.array([False, True, False, True, False, False, True], dtype=bool)
    rng = np.random.default_rng(20260813)
    n = host_mask.size
    sigma_pairs = rng.uniform(0.005, 0.05, n)
    noise = rng.standard_normal(n)
    z_obs = rng.uniform(0.1, 0.9, n)
    return host_mask, sigma_pairs, noise, z_obs


def test_ar1_default_target_reproduces_the_registered_dose(masking_inputs) -> None:  # type: ignore[no-untyped-def]
    """AR-1: ``dose_target='all'`` is bit-identical to the pre-arm expression."""
    host_mask, sigma_pairs, noise, z_obs = masking_inputs
    sig_out, z_out = _apply_dose_mask("all", host_mask, sigma_pairs, noise, z_obs)
    np.testing.assert_array_equal(sig_out, sigma_pairs)
    np.testing.assert_array_equal(z_out, z_obs + sigma_pairs * noise)


def test_ar2_host_and_impostor_masks_partition_the_ball(masking_inputs) -> None:  # type: ignore[no-untyped-def]
    """AR-2: the two split arms partition exactly the members the full dose reaches."""
    host_mask, sigma_pairs, noise, z_obs = masking_inputs
    sig_h, z_h = _apply_dose_mask("host", host_mask, sigma_pairs, noise, z_obs)
    sig_i, z_i = _apply_dose_mask("impostors", host_mask, sigma_pairs, noise, z_obs)
    sig_a, z_a = _apply_dose_mask("all", host_mask, sigma_pairs, noise, z_obs)

    # every member is dosed in exactly one of the two split arms
    np.testing.assert_array_equal((sig_h > 0) | (sig_i > 0), sig_a > 0)
    assert not np.any((sig_h > 0) & (sig_i > 0))
    # and the two displacements sum to the full one
    np.testing.assert_allclose((z_h - z_obs) + (z_i - z_obs), z_a - z_obs, rtol=0, atol=0)


def test_ar3_undosed_members_keep_exact_redshift_and_zero_kernel(masking_inputs) -> None:  # type: ignore[no-untyped-def]
    """AR-3: an undosed member is untouched AND gets sigma = 0 (matched-model)."""
    host_mask, sigma_pairs, noise, z_obs = masking_inputs
    sig_h, z_h = _apply_dose_mask("host", host_mask, sigma_pairs, noise, z_obs)

    np.testing.assert_array_equal(z_h[~host_mask], z_obs[~host_mask])
    np.testing.assert_array_equal(sig_h[~host_mask], np.zeros(int((~host_mask).sum())))
    # the dosed members carry the SAME draw the full-dose arm would have used
    np.testing.assert_array_equal(sig_h[host_mask], sigma_pairs[host_mask])
    np.testing.assert_array_equal(
        z_h[host_mask], z_obs[host_mask] + (sigma_pairs * noise)[host_mask]
    )


def test_unknown_dose_target_is_rejected(masking_inputs) -> None:  # type: ignore[no-untyped-def]
    """An unregistered arm name must fail loudly, never fall through to the default."""
    host_mask, sigma_pairs, noise, z_obs = masking_inputs
    with pytest.raises(ValueError, match="unknown dose_target"):
        _apply_dose_mask("hosts", host_mask, sigma_pairs, noise, z_obs)


def test_mech_arm_registry_is_separate_and_disjoint() -> None:
    """The mechanism arms live in their own +50000 decade, outside the v3 envelope.

    MN0X is a deliberate exception: it is a superset re-run of MN0's block
    (higher N to settle a validity check on data, not by widening a band),
    so the disjointness check permits exactly that one containment and
    forbids every other overlap.
    """
    from darksiren_emri.validation import venue_transfer as vt

    assert set(vt.MECH_CELL_SPECS) == {"MN0", "MEH", "MEI", "MN0X"}
    # the venue-transfer registry must NOT have been widened
    assert not (set(vt.CELL_SPECS) & set(vt.MECH_CELL_SPECS))
    # ALL_CELL_SPECS is the union of all five registries (the stage-2
    # estimator-variant arms, M2P_CELL_SPECS, and the stage-3 arms,
    # REN_CELL_SPECS, are disjoint families — see test_m2prime_ablation_arms.py
    # and test_a_jren_stage3_arms.py for their own registration tests).
    assert set(vt.ALL_CELL_SPECS) == (
        set(vt.CELL_SPECS)
        | set(vt.MECH_CELL_SPECS)
        | set(vt.SCAN_CELL_SPECS)
        | set(vt.M2P_CELL_SPECS)
        | set(vt.REN_CELL_SPECS)
    )

    v3_hi = vt.VT_BASE_SEED + vt.V3_SEED_OFFSET_ENVELOPE[1]
    blocks: dict[str, set[int]] = {}
    for name, spec in vt.MECH_CELL_SPECS.items():
        block = set(vt.venue_cell_seeds(spec, 0.730, 0, None))
        assert min(block) > v3_hi, f"{name} collides with the v3 envelope"
        blocks[name] = block
    assert len(blocks["MN0"]) == 15
    assert len(blocks["MEH"]) == 15
    assert len(blocks["MEI"]) == 15
    assert len(blocks["MN0X"]) == 100

    # MN0X is a superset of MN0 (the run seeds are kept, never discarded).
    assert blocks["MN0"] <= blocks["MN0X"]

    # every OTHER pair must be exactly disjoint.
    names = list(blocks)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            overlap = blocks[na] & blocks[nb]
            if {na, nb} == {"MN0", "MN0X"}:
                continue
            assert not overlap, f"{na} and {nb} unexpectedly overlap: {overlap}"

    # dose targets are pinned in the registry, not selectable at the CLI
    assert vt.MECH_CELL_SPECS["MN0"].dose_target == "all"
    assert vt.MECH_CELL_SPECS["MEH"].dose_target == "host"
    assert vt.MECH_CELL_SPECS["MEI"].dose_target == "impostors"
    assert vt.MECH_CELL_SPECS["MN0X"].dose_target == "all"
    # and every arm is otherwise the campaign's decision cell
    for spec in vt.MECH_CELL_SPECS.values():
        assert (spec.balls, spec.sigma_mode, spec.truths) == ("real_k", "glade", (0.730,))


def test_ar1_dose_scales_corners_are_bit_identical_to_dose_target(masking_inputs) -> None:  # type: ignore[no-untyped-def]
    """AR-1 (critical): the (f_host, f_impostors) refactor is bit-identical.

    Multiplying by exactly 1.0 and 0.0 is exact in IEEE754 — verify it, do
    not assume it, because the three arms already run (MN0/MEH/MEI) depend
    on this being true.
    """
    host_mask, sigma_pairs, noise, z_obs = masking_inputs

    for target, scales in (("all", (1.0, 1.0)), ("host", (1.0, 0.0)), ("impostors", (0.0, 1.0))):
        sig_target, z_target = _apply_dose_mask(target, host_mask, sigma_pairs, noise, z_obs)
        sig_scales, z_scales = _apply_dose_mask(
            target, host_mask, sigma_pairs, noise, z_obs, dose_scales=scales
        )
        np.testing.assert_array_equal(sig_target, sig_scales)
        np.testing.assert_array_equal(z_target, z_scales)

        # dose_scales overrides dose_target entirely — even a bogus target
        # string must be ignored once dose_scales is given (per docstring).
        sig_bogus, z_bogus = _apply_dose_mask(
            "not-a-real-target", host_mask, sigma_pairs, noise, z_obs, dose_scales=scales
        )
        np.testing.assert_array_equal(sig_target, sig_bogus)
        np.testing.assert_array_equal(z_target, z_bogus)


def test_ar2_fractional_dose_scales_apply_exactly(masking_inputs) -> None:  # type: ignore[no-untyped-def]
    """AR-2: fractional (f_host, f_impostors) scale each subgroup exactly."""
    host_mask, sigma_pairs, noise, z_obs = masking_inputs
    f_host, f_imp = 0.5, 0.25

    sig_out, z_out = _apply_dose_mask(
        "all", host_mask, sigma_pairs, noise, z_obs, dose_scales=(f_host, f_imp)
    )

    np.testing.assert_array_equal(sig_out[host_mask], sigma_pairs[host_mask] * f_host)
    np.testing.assert_array_equal(sig_out[~host_mask], sigma_pairs[~host_mask] * f_imp)
    np.testing.assert_array_equal(
        z_out[host_mask], z_obs[host_mask] + sigma_pairs[host_mask] * f_host * noise[host_mask]
    )
    np.testing.assert_array_equal(
        z_out[~host_mask],
        z_obs[~host_mask] + sigma_pairs[~host_mask] * f_imp * noise[~host_mask],
    )


def test_scan_cell_specs_grid_is_correctly_named_and_scaled() -> None:
    """The 2-D dose scan is 16 cells, correctly named, scaled, and seeded."""
    from darksiren_emri.validation import venue_transfer as vt

    assert set(vt.SCAN_CELL_SPECS) == {f"S{h}{i}" for h in range(4) for i in range(4)}
    assert len(vt.SCAN_CELL_SPECS) == 16

    for h in range(4):
        for i in range(4):
            name = f"S{h}{i}"
            spec = vt.SCAN_CELL_SPECS[name]
            assert spec.balls == "real_k"
            assert spec.sigma_mode == "glade"
            assert spec.truths == (0.730,)
            # S23 is the sole discriminating cell and gets SCAN_HIGH_N seeds;
            # every other cell in the grid stays at the base 15.
            expected_n = vt.SCAN_HIGH_N if name == vt.SCAN_HIGH_N_CELL else 15
            assert spec.n_seeds == (expected_n,), name
            assert spec.dose_scales == (
                vt.SCAN_DOSE_FRACTIONS[h],
                vt.SCAN_DOSE_FRACTIONS[i],
            )
            assert spec.seed_offsets == (51000 + 100 * (4 * h + i),)

    # spot-check the example from the task spec: S31 = host 1.0, impostor 0.25
    assert vt.SCAN_CELL_SPECS["S31"].dose_scales == (1.0, 0.25)
    assert vt.SCAN_CELL_SPECS["S00"].seed_offsets == (51000,)
    assert vt.SCAN_CELL_SPECS["S33"].seed_offsets == (52500,)

    # explicit S23 exception: N=100 instead of the uniform 15
    assert vt.SCAN_HIGH_N_CELL == "S23"
    assert vt.SCAN_HIGH_N == 100
    assert vt.SCAN_CELL_SPECS["S23"].n_seeds == (100,)


def test_scan_cell_specs_seed_blocks_are_disjoint_from_everything() -> None:
    """Scan blocks are disjoint from each other, the v3 envelope, MECH, and reserved blocks."""
    from darksiren_emri.validation import venue_transfer as vt

    scan_blocks: dict[str, set[int]] = {
        name: set(vt.venue_cell_seeds(spec, 0.730, 0, None))
        for name, spec in vt.SCAN_CELL_SPECS.items()
    }
    for name, block in scan_blocks.items():
        expected_len = vt.SCAN_HIGH_N if name == vt.SCAN_HIGH_N_CELL else 15
        assert len(block) == expected_len, name

    # S23's wider block must terminate exactly one seed below S30's start —
    # a plain disjointness check would not catch an off-by-one here, since
    # S23 (100 seeds) is the only block that could grow into its successor.
    s23_block = scan_blocks["S23"]
    s30_block = scan_blocks["S30"]
    assert max(s23_block) + 1 == min(s30_block)

    # pairwise disjoint among the scan cells themselves
    names = list(scan_blocks)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            overlap = scan_blocks[names[a]] & scan_blocks[names[b]]
            assert not overlap, f"{names[a]} and {names[b]} overlap: {overlap}"

    all_scan: set[int] = set().union(*scan_blocks.values())

    # disjoint from the v3 envelope
    v3_lo = vt.VT_BASE_SEED + vt.V3_SEED_OFFSET_ENVELOPE[0]
    v3_hi = vt.VT_BASE_SEED + vt.V3_SEED_OFFSET_ENVELOPE[1]
    assert not any(v3_lo <= s <= v3_hi for s in all_scan)

    # disjoint from MECH_CELL_SPECS (including MN0X)
    mech_all: set[int] = set()
    for spec in vt.MECH_CELL_SPECS.values():
        mech_all.update(vt.venue_cell_seeds(spec, 0.730, 0, None))
    assert not (all_scan & mech_all)

    # disjoint from the reserved W1/O2 blocks
    for block_name, (lo, hi) in vt.RESERVED_SEED_OFFSET_BLOCKS.items():
        lo_abs, hi_abs = vt.VT_BASE_SEED + lo, vt.VT_BASE_SEED + hi
        assert not any(lo_abs <= s <= hi_abs for s in all_scan), block_name


def test_registry_separation_and_union() -> None:
    """CELL_SPECS, MECH_CELL_SPECS, SCAN_CELL_SPECS, M2P_CELL_SPECS, REN_CELL_SPECS
    are pairwise key-disjoint."""
    from darksiren_emri.validation import venue_transfer as vt

    cell_keys = set(vt.CELL_SPECS)
    mech_keys = set(vt.MECH_CELL_SPECS)
    scan_keys = set(vt.SCAN_CELL_SPECS)
    m2p_keys = set(vt.M2P_CELL_SPECS)
    ren_keys = set(vt.REN_CELL_SPECS)

    assert not (cell_keys & mech_keys)
    assert not (cell_keys & scan_keys)
    assert not (mech_keys & scan_keys)
    assert not (cell_keys & m2p_keys)
    assert not (mech_keys & m2p_keys)
    assert not (scan_keys & m2p_keys)
    assert not (cell_keys & ren_keys)
    assert not (mech_keys & ren_keys)
    assert not (scan_keys & ren_keys)
    assert not (m2p_keys & ren_keys)
    assert set(vt.ALL_CELL_SPECS) == cell_keys | mech_keys | scan_keys | m2p_keys | ren_keys
    assert len(vt.ALL_CELL_SPECS) == (
        len(cell_keys) + len(mech_keys) + len(scan_keys) + len(m2p_keys) + len(ren_keys)
    )


def test_cli_choices_include_new_cells() -> None:
    """The CLI --cell parser accepts MN0X and every scan cell."""
    from darksiren_emri.validation import venue_transfer as vt

    parser = vt.build_parser()
    cell_action = next(a for a in parser._actions if a.dest == "cell")
    assert cell_action.choices is not None
    choices = set(cell_action.choices)
    assert "MN0X" in choices
    assert set(vt.SCAN_CELL_SPECS) <= choices
