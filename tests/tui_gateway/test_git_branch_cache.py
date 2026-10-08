"""``git_probe.branch`` runs on every session info build (``session.activate`` and friends).

It used to spawn one or two ``git`` processes per call. It now caches per cwd and
revalidates against the repo's ``HEAD`` file, so repeat calls cost a file read while a
checkout, switch or detached commit is still seen on the very next call.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from tui_gateway import git_probe

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

_ENV_ARGS = ("-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false")


def _git(cwd, *args):
    return subprocess.run(["git", *_ENV_ARGS, *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    (path / "a.txt").write_text("a", encoding="utf-8")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-qm", "init")
    git_probe.invalidate()
    yield path
    git_probe.invalidate()


@pytest.fixture
def spawns(monkeypatch):
    calls: list[tuple] = []
    real = git_probe.run_git

    def counting(cwd, *args):
        calls.append(args)
        return real(cwd, *args)

    monkeypatch.setattr(git_probe, "run_git", counting)
    return calls


def test_repeat_branch_reads_do_not_spawn_git(repo, spawns):
    assert git_probe.branch(str(repo)) == "main"
    first = len(spawns)
    for _ in range(5):
        assert git_probe.branch(str(repo)) == "main"
    assert len(spawns) == first


def test_a_checkout_is_seen_on_the_next_call(repo, spawns):
    assert git_probe.branch(str(repo)) == "main"
    _git(repo, "checkout", "-q", "-b", "feature-x")
    assert git_probe.branch(str(repo)) == "feature-x"
    _git(repo, "checkout", "-q", "main")
    assert git_probe.branch(str(repo)) == "main"


def test_detached_head_reports_the_short_sha_and_follows_new_commits(repo, spawns):
    _git(repo, "checkout", "-q", "--detach")
    assert git_probe.branch(str(repo)) == _git(repo, "rev-parse", "--short", "HEAD")
    (repo / "b.txt").write_text("b", encoding="utf-8")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-qm", "second")
    assert git_probe.branch(str(repo)) == _git(repo, "rev-parse", "--short", "HEAD")


def test_linked_worktree_has_its_own_branch(repo, tmp_path, spawns):
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", "-b", "side", str(wt))
    assert git_probe.branch(str(repo)) == "main"
    assert git_probe.branch(str(wt)) == "side"
    assert git_probe.branch(str(wt)) == "side"


def test_non_repo_cwd_is_empty_and_not_reprobed(tmp_path, spawns):
    plain = tmp_path / "plain"
    plain.mkdir()
    git_probe.invalidate()
    assert git_probe.branch(str(plain)) == ""
    first = len(spawns)
    assert git_probe.branch(str(plain)) == ""
    assert len(spawns) == first
    assert git_probe.branch(str(tmp_path / "missing")) == ""
