"""Tests for draw_sealed_h.py on a throwaway git repo; no real seal is ever touched.

Run: uv run pytest scripts/sealed_mock/test_draw_sealed_h.py -p no:cacheprovider --no-cov
"""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import draw_sealed_h as dsh  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def frozen_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    (repo / "darksiren_emri").mkdir(parents=True)
    (repo / "darksiren_emri" / "x.py").write_text("X = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "add", ".")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "freeze")
    return repo, _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD:darksiren_emri")


def _erratum(tmp_path: Path, tree: str, lo: str = "0.79") -> Path:
    text = (
        "# erratum\n\n"
        f"{dsh.PARAMS_BEGIN}\n```\n"
        f"prior_lo = {lo}\nprior_hi = 0.82\ncode_tree = {tree}\n"
        "commitment_format = sha256(h_inj_6dp|salt_hex)\nrevision_cap = 1\n"
        f"```\n{dsh.PARAMS_END}\n"
    )
    path = tmp_path / "ERRATUM.md"
    path.write_text(text)
    return path


@pytest.fixture(autouse=True)
def _no_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in dsh.AGENT_ENV_MARKERS:
        monkeypatch.delenv(key, raising=False)


def test_draw_seals_and_commitment_verifies(
    tmp_path: Path, frozen_repo: tuple[Path, str, str]
) -> None:
    repo, head, tree = frozen_repo
    erratum = _erratum(tmp_path, tree)
    sha = dsh.params_sha256(dsh.extract_params_block(erratum.read_text()))
    sealed = tmp_path / "sealed"
    receipt = dsh.draw(erratum, sha, head, repo, sealed)
    assert set(receipt) == {"commitment", "params_sha256", "h_code", "utc"}
    record = json.loads((sealed / dsh.SEALED_FILENAME).read_text())
    assert 0.79 <= float(record["h_inj"]) <= 0.82
    assert stat.S_IMODE((sealed / dsh.SEALED_FILENAME).stat().st_mode) == 0o400
    assert stat.S_IMODE(sealed.stat().st_mode) == 0o700
    assert dsh.verify(record["h_inj"], record["salt"], receipt["commitment"])["match"] is True
    assert dsh.unseal(sealed, receipt["commitment"])["match"] is True


def test_second_draw_refused(tmp_path: Path, frozen_repo: tuple[Path, str, str]) -> None:
    repo, head, tree = frozen_repo
    erratum = _erratum(tmp_path, tree)
    sha = dsh.params_sha256(dsh.extract_params_block(erratum.read_text()))
    dsh.draw(erratum, sha, head, repo, tmp_path / "sealed")
    with pytest.raises(dsh.SealError, match="spent"):
        dsh.draw(erratum, sha, head, repo, tmp_path / "sealed")


@pytest.mark.parametrize("defect", ["params_sha", "prior", "head", "dirty"])
def test_preconditions_refuse(
    tmp_path: Path, frozen_repo: tuple[Path, str, str], defect: str
) -> None:
    repo, head, tree = frozen_repo
    erratum = _erratum(tmp_path, tree, lo="0.78" if defect == "prior" else "0.79")
    sha = dsh.params_sha256(dsh.extract_params_block(erratum.read_text()))
    if defect == "params_sha":
        sha = "0" * 64
    if defect == "head":
        head = "f" * 40
    if defect == "dirty":
        (repo / "darksiren_emri" / "x.py").write_text("X = 2\n")
    with pytest.raises(dsh.SealError):
        dsh.draw(erratum, sha, head, repo, tmp_path / "sealed")
    assert not (tmp_path / "sealed" / dsh.SEALED_FILENAME).exists()


def test_agent_session_refused(
    tmp_path: Path, frozen_repo: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, head, tree = frozen_repo
    erratum = _erratum(tmp_path, tree)
    sha = dsh.params_sha256(dsh.extract_params_block(erratum.read_text()))
    monkeypatch.setenv("CLAUDECODE", "1")
    with pytest.raises(dsh.SealError, match="agent session"):
        dsh.draw(erratum, sha, head, repo, tmp_path / "sealed")


def test_verify_rejects_wrong_salt() -> None:
    good = dsh.commitment_of("0.801234", "ab" * 32)
    assert dsh.verify("0.801234", "cd" * 32, good)["match"] is False
    with pytest.raises(dsh.SealError):
        dsh.verify("0.8012", "ab" * 32, good)
