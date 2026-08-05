"""Tests for the repository isolation guard (tests/isolation.py).

The helpers are exercised against synthetic directory layouts so the
failure paths are actually tested, not just the happy one -- the whole
point of this guard is that the broken states previously looked green.
"""

import sys

import pytest

import isolation


@pytest.fixture
def fake_checkouts(tmp_path):
    """Two directory trees that both look like this application's source
    root -- 'mine' and a rival 'sibling'."""
    roots = {}
    for name in ("mine", "sibling"):
        src = tmp_path / name / "src"
        src.mkdir(parents=True)
        for marker in isolation.SOURCE_ROOT_MARKERS:
            (src / marker).write_text("# marker\n")
        roots[name] = src
    return roots


class TestSourceRootDetection:
    def test_recognizes_a_checkout_source_root(self, fake_checkouts):
        assert isolation.looks_like_source_root(fake_checkouts["mine"])

    def test_ignores_an_unrelated_directory(self, tmp_path):
        other = tmp_path / "unrelated" / "src"
        other.mkdir(parents=True)
        (other / "main_window.py").write_text("")  # only one marker
        assert not isolation.looks_like_source_root(other)

    def test_missing_path_is_not_a_source_root(self, tmp_path):
        assert not isolation.looks_like_source_root(tmp_path / "nope")


class TestForeignAndShadowing:
    def test_foreign_root_is_detected(self, fake_checkouts, tmp_path):
        root = tmp_path / "mine"
        paths = [str(fake_checkouts["mine"]), str(fake_checkouts["sibling"])]
        assert isolation.foreign_source_roots(paths, root) == [str(fake_checkouts["sibling"])]

    def test_foreign_root_after_ours_does_not_shadow(self, fake_checkouts, tmp_path):
        root = tmp_path / "mine"
        paths = [str(fake_checkouts["mine"]), str(fake_checkouts["sibling"])]
        assert isolation.shadowing_roots(paths, root) == []

    def test_foreign_root_before_ours_shadows(self, fake_checkouts, tmp_path):
        root = tmp_path / "mine"
        paths = [str(fake_checkouts["sibling"]), str(fake_checkouts["mine"])]
        assert isolation.shadowing_roots(paths, root) == [str(fake_checkouts["sibling"])]

    def test_foreign_root_shadows_when_ours_is_absent(self, fake_checkouts, tmp_path):
        root = tmp_path / "mine"
        paths = [str(fake_checkouts["sibling"])]
        assert isolation.shadowing_roots(paths, root) == [str(fake_checkouts["sibling"])]


class TestContaminationReporting:
    def test_module_loaded_from_another_checkout_is_fatal(self, fake_checkouts, tmp_path):
        root = tmp_path / "mine"
        intruder = type(sys)("main_window")
        intruder.__file__ = str(fake_checkouts["sibling"] / "main_window.py")
        report = isolation.fatal_report(root, modules={"main_window": intruder}, sys_paths=[])
        assert "Repository isolation failure" in report
        assert "main_window" in report
        assert str(fake_checkouts["sibling"]) in report   # WHERE it came from
        assert str(root) in report                        # WHAT was expected
        assert "pip uninstall fdsvis" in report           # HOW to fix

    def test_module_loaded_from_this_checkout_is_clean(self, tmp_path):
        root = tmp_path / "mine"
        (root / "src").mkdir(parents=True)
        good = type(sys)("main_window")
        good.__file__ = str(root / "src" / "main_window.py")
        assert isolation.fatal_report(root, modules={"main_window": good}, sys_paths=[]) == ""

    def test_shadowing_path_is_fatal(self, fake_checkouts, tmp_path):
        root = tmp_path / "mine"
        report = isolation.fatal_report(
            root, modules={}, sys_paths=[str(fake_checkouts["sibling"])])
        assert "takes priority over this one" in report

    def test_test_module_outside_the_checkout_is_fatal(self, tmp_path):
        root = tmp_path / "mine"
        report = isolation.fatal_report(
            root, modules={}, sys_paths=[],
            test_paths=["/somewhere/else/tests/test_integration.py"])
        assert "collected from outside this checkout" in report

    def test_clean_environment_produces_no_report(self, tmp_path):
        root = tmp_path / "mine"
        assert isolation.fatal_report(root, modules={}, sys_paths=[], test_paths=[]) == ""

    def test_non_shadowing_foreign_root_warns_but_does_not_fail(self, fake_checkouts, tmp_path):
        root = tmp_path / "mine"
        paths = [str(fake_checkouts["mine"]), str(fake_checkouts["sibling"])]
        assert isolation.fatal_report(root, modules={}, sys_paths=paths) == ""
        assert "isolation warning" in isolation.warning_report(root, sys_paths=paths)


class TestThisCheckoutIsClean:
    """The live assertions -- these are what would have caught both real
    incidents."""

    def test_project_root_is_this_repository(self):
        root = isolation.project_root()
        assert (root / "tests").is_dir()
        assert (root / "src" / "main_window.py").is_file()

    @pytest.mark.parametrize("name", isolation.CORE_MODULES)
    def test_core_modules_resolve_inside_this_checkout(self, name):
        module = __import__(name, fromlist=["__file__"])
        assert isolation.is_inside(module.__file__, isolation.project_root()), (
            f"{name} loaded from {module.__file__}")

    def test_no_module_is_currently_contaminated(self):
        assert isolation.contaminated_modules(sys.modules, isolation.project_root()) == []

    def test_this_checkouts_source_root_is_importable(self):
        root = isolation.project_root()
        assert isolation.own_source_roots(sys.path, root), (
            "this checkout's src/ is not on sys.path")
