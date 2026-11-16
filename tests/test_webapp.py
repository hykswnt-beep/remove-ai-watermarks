"""Web UI handlers: capability gating, metadata stripping, and the Gradio smoke."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from remove_ai_watermarks import metadata, webapp
from remove_ai_watermarks.identify import ProvenanceReport

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "provenance"
C2PA_SAMPLE = FIXTURES / "chatgpt-1.png"


@pytest.fixture(autouse=True)
def _clear_capability_cache():
    webapp.capabilities.cache_clear()
    yield
    webapp.capabilities.cache_clear()


class TestCapabilities:
    def test_synthid_on_a_non_cuda_host_is_disabled_with_a_reason(self, monkeypatch):
        monkeypatch.setattr("remove_ai_watermarks.invisible_engine.is_available", lambda: True)
        monkeypatch.setattr(webapp, "_cuda_available", lambda: False)
        caps = webapp.capabilities()
        assert not caps.invisible
        assert "NVIDIA CUDA" in caps.invisible_reason

    def test_missing_diffusion_extra_points_at_the_extra(self, monkeypatch):
        monkeypatch.setattr("remove_ai_watermarks.invisible_engine.is_available", lambda: False)
        caps = webapp.capabilities()
        assert not caps.invisible
        assert "qwen-zimage" in caps.invisible_reason


class TestCleanImage:
    def test_no_action_is_refused(self):
        output, message = webapp.clean_image(
            str(C2PA_SAMPLE),
            remove_visible_mark=False,
            strip_metadata=False,
            remove_synthid=False,
        )
        assert output is None
        assert "Select at least one action" in message

    def test_metadata_only_strips_c2pa(self):
        assert metadata.get_ai_metadata(C2PA_SAMPLE), "fixture should carry AI provenance"
        output, message = webapp.clean_image(
            str(C2PA_SAMPLE),
            remove_visible_mark=False,
            strip_metadata=True,
            remove_synthid=False,
        )
        assert output is not None
        try:
            assert metadata.get_ai_metadata(Path(output)) == {}
            assert "stripped" in message.lower()
        finally:
            Path(output).unlink(missing_ok=True)

    def test_synthid_is_refused_before_writing_on_an_unsupported_host(self, monkeypatch):
        monkeypatch.setattr(
            webapp,
            "capabilities",
            lambda: webapp.Capabilities(visible=True, invisible=False, invisible_reason="needs CUDA"),
        )
        output, message = webapp.clean_image(
            str(C2PA_SAMPLE),
            remove_visible_mark=False,
            strip_metadata=True,
            remove_synthid=True,
        )
        assert output is None
        assert "not modified" in message

    def test_output_name_does_not_trust_the_upload_name(self, tmp_path):
        source = tmp_path / "my photo.png"
        source.write_bytes(C2PA_SAMPLE.read_bytes())
        output, _ = webapp.clean_image(
            str(source),
            remove_visible_mark=False,
            strip_metadata=True,
            remove_synthid=False,
        )
        assert output is not None
        try:
            assert source.stem not in Path(output).name
        finally:
            Path(output).unlink(missing_ok=True)


class TestIdentify:
    def test_formats_a_provenance_report(self):
        text = webapp.identify_image(str(C2PA_SAMPLE))
        assert "Verdict:" in text
        assert "Platform:" in text


class TestTempOutputs:
    def test_prune_removes_only_files_past_retention(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webapp, "_TMP_DIR", tmp_path)
        old = tmp_path / "old.png"
        fresh = tmp_path / "fresh.png"
        old.write_bytes(b"x")
        fresh.write_bytes(b"x")
        now = 1_000_000.0
        os.utime(old, (now, now - webapp._OUTPUT_RETENTION_SECONDS - 1))
        os.utime(fresh, (now, now - 60))
        webapp._prune_tmp_dir(now=now)
        assert not old.exists()
        assert fresh.exists()


class TestBuildApp:
    def test_constructs_when_gradio_is_installed(self):
        pytest.importorskip("gradio")
        assert webapp.build_app() is not None

    def test_missing_gradio_names_the_extra(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "gradio", None)
        with pytest.raises(ImportError, match="web"):
            webapp.build_app()


def test_format_report_keeps_the_unknown_caveat():
    report = ProvenanceReport(
        path=Path("x.png"),
        is_ai_generated=None,
        confidence="none",
        platform=None,
        watermarks=[],
        caveats=["Absence of metadata is not proof the image is clean."],
    )
    text = webapp.format_report(report)
    assert "Verdict:  unknown" in text
    assert "not the same as 'clean'" in text
