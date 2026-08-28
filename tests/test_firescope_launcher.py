"""Tests for public.firescope_launcher -- the Grown-ups button's process
launch/activation, kept deliberately independent of FireScope's own repo
(see the module's docstring). No real subprocess or osascript is ever
run here: Popen and osascript's subprocess.run are monkeypatched in
every test that would otherwise touch them."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from public import firescope_launcher as launcher  # noqa: E402


def _make_fake_firescope(tmp_path, name="FireScope"):
    root = tmp_path / name
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("# stub\n")
    return root


class TestFindFireScopeRoot:
    def test_env_override_wins_when_valid(self, tmp_path, monkeypatch):
        root = _make_fake_firescope(tmp_path, "elsewhere")
        monkeypatch.setenv("FIRESCOPE_APP_PATH", str(root))
        assert launcher.find_firescope_root() == root

    def test_env_override_with_no_main_py_is_reported_missing(self, tmp_path, monkeypatch):
        empty = tmp_path / "not_firescope"
        empty.mkdir()
        monkeypatch.setenv("FIRESCOPE_APP_PATH", str(empty))
        assert launcher.find_firescope_root() is None

    def test_sibling_layout_found_when_no_override(self, tmp_path, monkeypatch):
        """find_firescope_root() derives "this repo's root" from its own
        __file__ (parents[2]: public/firescope_launcher.py -> public ->
        src -> repo root), then looks for a 'FireScope' sibling next to
        it. Patch __file__ itself, the one thing the function actually
        reads, rather than any Path internals."""
        monkeypatch.delenv("FIRESCOPE_APP_PATH", raising=False)
        kids_repo = tmp_path / "kids_repo"
        fake_module_file = kids_repo / "src" / "public" / "firescope_launcher.py"
        fake_module_file.parent.mkdir(parents=True)
        expected = _make_fake_firescope(tmp_path, "FireScope")

        monkeypatch.setattr(launcher, "__file__", str(fake_module_file))

        assert launcher.find_firescope_root() == expected

    def test_missing_sibling_reports_none(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FIRESCOPE_APP_PATH", raising=False)
        monkeypatch.setattr(launcher, "_DEFAULT_SIBLING_NAME", "does-not-exist-anywhere")
        assert launcher.find_firescope_root() is None


class TestPidAlive:
    def test_this_process_is_alive(self):
        assert launcher.pid_alive(os.getpid()) is True

    def test_a_pid_that_cannot_exist_is_not_alive(self):
        assert launcher.pid_alive(2**30) is False


class TestActivatePid:
    def test_non_macos_returns_false_without_running_anything(self, monkeypatch):
        monkeypatch.setattr(launcher.sys, "platform", "linux")
        calls = []
        monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: calls.append(1))
        assert launcher.activate_pid(1234) is False
        assert calls == []

    def test_macos_success(self, monkeypatch):
        monkeypatch.setattr(launcher.sys, "platform", "darwin")

        class _Result:
            returncode = 0

        monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: _Result())
        assert launcher.activate_pid(1234) is True

    def test_macos_nonzero_exit_is_false(self, monkeypatch):
        monkeypatch.setattr(launcher.sys, "platform", "darwin")

        class _Result:
            returncode = 1

        monkeypatch.setattr(launcher.subprocess, "run", lambda *a, **k: _Result())
        assert launcher.activate_pid(1234) is False

    def test_osascript_missing_is_reported_not_raised(self, monkeypatch):
        monkeypatch.setattr(launcher.sys, "platform", "darwin")

        def raise_oserror(*a, **k):
            raise OSError("no such file")

        monkeypatch.setattr(launcher.subprocess, "run", raise_oserror)
        assert launcher.activate_pid(1234) is False


class TestLaunchFireScope:
    def test_missing_install_fails_with_honest_message(self, monkeypatch):
        monkeypatch.setattr(launcher, "find_firescope_root", lambda: None)
        process, message = launcher.launch_firescope()
        assert process is None
        assert "FIRESCOPE_APP_PATH" in message

    def test_success_calls_popen_with_correct_cwd_pythonpath_and_pid(self, tmp_path, monkeypatch):
        root = _make_fake_firescope(tmp_path)
        monkeypatch.setattr(launcher, "find_firescope_root", lambda: root)
        # A real, always-present path so the "configured python exists"
        # check passes without patching Path itself.
        monkeypatch.setenv("FIRESCOPE_PYTHON", sys.executable)

        calls = {}
        sentinel = object()

        def fake_popen(args, cwd=None, env=None, **kwargs):
            calls["args"] = args
            calls["cwd"] = cwd
            calls["env"] = env
            calls["start_new_session"] = kwargs.get("start_new_session")
            return sentinel

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        process, message = launcher.launch_firescope()

        assert process is sentinel
        assert message == ""
        assert calls["cwd"] == str(root / "src")
        assert calls["args"][-1] == "main.py"
        assert calls["env"]["PYTHONPATH"] == str(root / "src")
        assert calls["env"]["JUNIOR_FIRE_SCIENTIST_PID"] == str(os.getpid())
        assert calls["start_new_session"] is True

    def test_falls_back_to_this_interpreter_when_configured_python_missing(self, tmp_path, monkeypatch):
        root = _make_fake_firescope(tmp_path)
        monkeypatch.setattr(launcher, "find_firescope_root", lambda: root)
        monkeypatch.setenv("FIRESCOPE_PYTHON", str(tmp_path / "no-such-python"))

        calls = {}

        def fake_popen(args, cwd=None, env=None, **kwargs):
            calls["args"] = args
            return object()

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        process, message = launcher.launch_firescope()

        assert process is not None
        assert calls["args"][0] == sys.executable

    def test_popen_oserror_is_reported_not_raised(self, tmp_path, monkeypatch):
        root = _make_fake_firescope(tmp_path)
        monkeypatch.setattr(launcher, "find_firescope_root", lambda: root)

        def fake_popen(*args, **kwargs):
            raise OSError("no such file or directory")

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        process, message = launcher.launch_firescope()

        assert process is None
        assert "Couldn't start FireScope" in message
