"""Tests for public.firescope_launcher -- the Grown-ups button's process
launch, kept deliberately independent of FireScope's own repo (see the
module's docstring). No real subprocess is ever started here: Popen is
monkeypatched in every launch_firescope() test."""

import os
import sys

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


class TestLaunchFireScope:
    def test_missing_install_fails_with_honest_message(self, monkeypatch):
        monkeypatch.setattr(launcher, "find_firescope_root", lambda: None)
        started, message = launcher.launch_firescope()
        assert started is False
        assert "FIRESCOPE_APP_PATH" in message

    def test_success_calls_popen_with_correct_cwd_and_pythonpath(self, tmp_path, monkeypatch):
        root = _make_fake_firescope(tmp_path)
        monkeypatch.setattr(launcher, "find_firescope_root", lambda: root)
        # A real, always-present path so the "configured python exists"
        # check passes without patching Path itself.
        monkeypatch.setenv("FIRESCOPE_PYTHON", sys.executable)

        calls = {}

        def fake_popen(args, cwd=None, env=None, **kwargs):
            calls["args"] = args
            calls["cwd"] = cwd
            calls["env"] = env
            calls["start_new_session"] = kwargs.get("start_new_session")
            class _Proc:
                pass
            return _Proc()

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        started, message = launcher.launch_firescope()

        assert started is True
        assert message == ""
        assert calls["cwd"] == str(root / "src")
        assert calls["args"][-1] == "main.py"
        assert calls["env"]["PYTHONPATH"] == str(root / "src")
        assert calls["start_new_session"] is True

    def test_falls_back_to_this_interpreter_when_configured_python_missing(self, tmp_path, monkeypatch):
        root = _make_fake_firescope(tmp_path)
        monkeypatch.setattr(launcher, "find_firescope_root", lambda: root)
        monkeypatch.setenv("FIRESCOPE_PYTHON", str(tmp_path / "no-such-python"))

        calls = {}

        def fake_popen(args, cwd=None, env=None, **kwargs):
            calls["args"] = args
            class _Proc:
                pass
            return _Proc()

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        started, message = launcher.launch_firescope()

        assert started is True
        assert calls["args"][0] == sys.executable

    def test_popen_oserror_is_reported_not_raised(self, tmp_path, monkeypatch):
        root = _make_fake_firescope(tmp_path)
        monkeypatch.setattr(launcher, "find_firescope_root", lambda: root)

        def fake_popen(*args, **kwargs):
            raise OSError("no such file or directory")

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        started, message = launcher.launch_firescope()

        assert started is False
        assert "Couldn't start FireScope" in message
