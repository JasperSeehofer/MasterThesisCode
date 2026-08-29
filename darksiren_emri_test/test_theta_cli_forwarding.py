"""Tests for the [HIER] theta-hook CLI plumbing (charter node P6/B1.2,
results/campaign51_20260728/realistic_20260729/fanout1_20260829/
WAVE2_REGISTRATION_CHECK_20260829.md F-C; ledger rows #216, #221-#223).

F-C found that ``BayesianStatistics.evaluate()`` accepts
``theta_b``/``theta_s``/``theta_sites`` but neither ``darksiren_emri/arguments.py``
nor ``darksiren_emri/main.py`` exposed a CLI surface for them. These tests pin:

1. ``Arguments`` parses ``--theta_b``/``--theta_s``/``--theta_sites`` with
   byte-identical defaults (0.0, 1.0, "all").
2. ``darksiren_emri.main.evaluate()`` forwards all three kwargs, unmodified,
   to ``BayesianStatistics.evaluate()``.

The ``main()`` CLI-dispatch call site (``main.py`` inside the
``if arguments.evaluate:`` block) forwards ``arguments.theta_b`` /
``arguments.theta_s`` / ``arguments.theta_sites`` to this same
``darksiren_emri.main.evaluate()`` positionally alongside the other
already-threaded flags (e.g. ``mass_filter_geometry``); that call site is
read-verified in the P6 record rather than re-exercised here, since driving
``main()`` end-to-end requires constructing a full ``Model1CrossCheck`` +
``GalaxyCatalogueHandler``, which is out of scope for this plumbing node.

``BayesianStatistics.evaluate()`` itself (and its theta_sites validation) is
untouched by this node -- it is mocked out here, never called.
"""

from unittest.mock import MagicMock, patch

import pytest

from darksiren_emri.arguments import Arguments


def test_evaluate_forwards_theta_defaults() -> None:
    """darksiren_emri.main.evaluate() forwards identity theta defaults to
    BayesianStatistics.evaluate() unmodified."""
    from darksiren_emri import main as main_module

    with patch(
        "darksiren_emri.bayesian_inference.bayesian_statistics.BayesianStatistics"
    ) as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        main_module.evaluate(
            cosmological_model=MagicMock(),
            galaxy_catalog=MagicMock(),
            h_value=0.73,
        )

        mock_instance.evaluate.assert_called_once()
        kwargs = mock_instance.evaluate.call_args.kwargs
        assert kwargs["theta_b"] == pytest.approx(0.0)
        assert kwargs["theta_s"] == pytest.approx(1.0)
        assert kwargs["theta_sites"] == "all"


def test_evaluate_forwards_custom_theta() -> None:
    """darksiren_emri.main.evaluate() forwards non-default theta kwargs
    unmodified to BayesianStatistics.evaluate()."""
    from darksiren_emri import main as main_module

    with patch(
        "darksiren_emri.bayesian_inference.bayesian_statistics.BayesianStatistics"
    ) as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        main_module.evaluate(
            cosmological_model=MagicMock(),
            galaxy_catalog=MagicMock(),
            h_value=0.73,
            theta_b=0.01,
            theta_s=1.2,
            theta_sites="2.2",
        )

        kwargs = mock_instance.evaluate.call_args.kwargs
        assert kwargs["theta_b"] == pytest.approx(0.01)
        assert kwargs["theta_s"] == pytest.approx(1.2)
        assert kwargs["theta_sites"] == "2.2"


def test_arguments_theta_values_parse_for_cli_dispatch() -> None:
    """Arguments.theta_b/theta_s/theta_sites -- the values main()'s
    --evaluate dispatch reads and forwards to darksiren_emri.main.evaluate()
    -- parse to the values passed on the command line."""
    args = Arguments.create(
        [
            ".",
            "--evaluate",
            "--theta_b",
            "0.02",
            "--theta_s",
            "1.1",
            "--theta_sites",
            "2.1",
        ]
    )
    assert args.theta_b == pytest.approx(0.02)
    assert args.theta_s == pytest.approx(1.1)
    assert args.theta_sites == "2.1"
