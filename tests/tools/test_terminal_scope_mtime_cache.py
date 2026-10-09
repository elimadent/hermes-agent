"""build_profile_terminal_scope caches the file-backed policy by (home, env mtime, config mtime).

Each turn, RPC and heartbeat poll used to rebuild the complete TERMINAL_* mapping from disk,
including a PyYAML parse of config.yaml. Overlay and cwd resolution stay per-call so a launch
overlay cannot leak into the cached base, and YAML still wins over overlay.
"""

from __future__ import annotations

import os

from tools import terminal_scope as ts
from tools.terminal_scope import build_profile_terminal_scope


def _profile(tmp_path, *, config="", dotenv=""):
    home = tmp_path / "profiles" / "work"
    home.mkdir(parents=True)
    if config:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    if dotenv:
        (home / ".env").write_text(dotenv, encoding="utf-8")
    return home


def test_repeat_calls_reuse_the_file_backed_policy(tmp_path, monkeypatch):
    home = _profile(tmp_path, config="terminal:\n  backend: local\n  timeout: 7\n", dotenv="TERMINAL_SSH_HOST=10.0.0.9\n")
    loads = {"env": 0, "yaml": 0}
    real_load_env = __import__("agent.secret_scope", fromlist=["load_env_file"]).load_env_file
    real_yaml = __import__("utils", fromlist=["load_yaml_file_readonly"]).load_yaml_file_readonly

    def counting_env(path):
        loads["env"] += 1
        return real_load_env(path)

    def counting_yaml(path):
        loads["yaml"] += 1
        return real_yaml(path)

    monkeypatch.setattr("agent.secret_scope.load_env_file", counting_env)
    monkeypatch.setattr("utils.load_yaml_file_readonly", counting_yaml)
    getattr(ts, '_reset_terminal_base_scope_cache', lambda: None)()

    first = build_profile_terminal_scope(home)
    second = build_profile_terminal_scope(home)

    assert first["TERMINAL_ENV"] == "local"
    assert first["TERMINAL_TIMEOUT"] == "7"
    assert first["TERMINAL_SSH_HOST"] == "10.0.0.9"
    assert first == second
    assert first is not second
    assert loads == {"env": 1, "yaml": 1}

    third = build_profile_terminal_scope(home, env_overlay={"TERMINAL_SSH_USER": "launch"})
    assert third["TERMINAL_SSH_USER"] == "launch"
    assert first["TERMINAL_SSH_USER"] == ""
    assert loads["env"] == 1
    assert loads["yaml"] == 2  # overlay path re-applies YAML so YAML still wins


def test_mtime_change_invalidates_the_cached_policy(tmp_path, monkeypatch):
    home = _profile(tmp_path, config="terminal:\n  backend: local\n")
    getattr(ts, '_reset_terminal_base_scope_cache', lambda: None)()
    first = build_profile_terminal_scope(home)
    assert first["TERMINAL_ENV"] == "local"

    path = home / "config.yaml"
    path.write_text("terminal:\n  backend: docker\n", encoding="utf-8")
    os.utime(path, ns=(os.stat(path).st_mtime_ns + 1_000_000,) * 2)
    second = build_profile_terminal_scope(home)
    assert second["TERMINAL_ENV"] == "docker"
    assert first["TERMINAL_ENV"] == "local"


def test_overlay_does_not_outrank_yaml(tmp_path):
    home = _profile(tmp_path, config="terminal:\n  backend: local\n")
    getattr(ts, '_reset_terminal_base_scope_cache', lambda: None)()
    scope = build_profile_terminal_scope(home, env_overlay={"TERMINAL_ENV": "docker"})
    assert scope["TERMINAL_ENV"] == "local"
