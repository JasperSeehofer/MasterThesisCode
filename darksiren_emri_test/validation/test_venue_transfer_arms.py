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
