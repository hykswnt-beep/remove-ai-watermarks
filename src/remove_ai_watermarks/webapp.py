"""Optional local browser UI for image cleaning.

Install the ``web`` extra and run ``remove-ai-watermarks serve``. The server binds
to loopback by default. Image SynthID removal remains CUDA-only; when the host
cannot provide CUDA, the control is disabled with the reason instead of failing
after an upload.

Gradio is imported only inside :func:`build_app`, so importing this module (or the
package) never pulls the web stack in a headless install.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks import optional_deps
from remove_ai_watermarks._internal.watermark_profiles import INVISIBLE_EXTRA

if TYPE_CHECKING:
    from collections.abc import Callable

    from remove_ai_watermarks.identify import ProvenanceReport

WEB_EXTRA = "'remove-ai-watermarks[web]'"
VISIBLE_EXTRA = "'remove-ai-watermarks[visible]'"

_TMP_DIR = Path(tempfile.gettempdir()) / "remove-ai-watermarks-web"

_STAGE_PROGRESS = {"visible": 0.25, "invisible": 0.5, "metadata": 0.95}


@dataclass(frozen=True)
class Capabilities:
    """What this host can run, decided before an upload is processed."""

    visible: bool
    invisible: bool
    invisible_reason: str


def _cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def capabilities() -> Capabilities:
    """Probe the optional backends. Cached: installing a backend needs a restart."""
    from remove_ai_watermarks.invisible_engine import is_available

    visible = optional_deps.module_available("cv2", "numpy")
    if not is_available():
        invisible_reason = f"SynthID image removal needs the CUDA-only extra: pip install {INVISIBLE_EXTRA}"
        invisible = False
    elif not _cuda_available():
        invisible_reason = "SynthID image removal needs an NVIDIA CUDA GPU; this host has none."
        invisible = False
    else:
        invisible_reason = ""
        invisible = True
    return Capabilities(visible=visible, invisible=invisible, invisible_reason=invisible_reason)


def capability_lines() -> list[str]:
    """One-line host capability summary for the CLI banner and the web UI."""
    caps = capabilities()
    visible = "available" if caps.visible else f"unavailable -- pip install {VISIBLE_EXTRA}"
    invisible = "available (CUDA)" if caps.invisible else f"unavailable -- {caps.invisible_reason}"
    return [f"Visible-mark removal: {visible}", f"SynthID image removal: {invisible}"]


_OUTPUT_RETENTION_SECONDS = 24 * 60 * 60


def _prune_tmp_dir(now: float | None = None) -> None:
    """Best-effort removal of old outputs so a long-running server cannot fill the temp dir."""
    if not _TMP_DIR.exists():
        return
    cutoff = (now if now is not None else time.time()) - _OUTPUT_RETENTION_SECONDS
    for path in _TMP_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


def _temp_output(source: Path) -> Path:
    """Unique per-request output path; never derived from the uploaded name."""
    _TMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower() or ".png"
    return _TMP_DIR / f"clean_{uuid.uuid4().hex}{suffix}"


def format_report(report: ProvenanceReport) -> str:
    """Plain-text provenance report for the web UI."""
    verdict = {True: "AI-generated", False: "not AI", None: "unknown"}[report.is_ai_generated]
    if report.is_ai_generated and report.ai_source_kind == "enhanced":
        verdict = "AI-enhanced (real content with an AI-composited region)"
    elif report.is_ai_generated and report.ai_source_kind == "generated":
        verdict = "AI-generated (fully synthetic)"
    lines = [
        f"Verdict:  {verdict}  (confidence: {report.confidence})",
        f"Platform: {report.platform or 'undetermined'}",
    ]
    if report.is_ai_generated is None:
        lines.append(
            "\nNo locally-readable AI signal found. This is not the same as 'clean': "
            "metadata is often stripped by re-encoding, screenshots, or upload, and "
            "SynthID-class pixel watermarks have no local detector."
        )
    if report.integrity_clashes:
        lines.append("\nIntegrity clash (provenance signals contradict each other):")
        lines.extend(f"  - {clash}" for clash in report.integrity_clashes)
    if report.watermarks:
        lines.append("\nWatermarks / provenance markers:")
        lines.extend(f"  - {mark}" for mark in report.watermarks)
    else:
        lines.append("\nNo watermarks or provenance markers found.")
    if report.caveats:
        lines.append("\nCaveats:")
        lines.extend(f"  - {caveat}" for caveat in report.caveats)
    return "\n".join(lines)


def identify_image(source: str | None) -> str:
    if not source:
        return "Upload an image first."
    from remove_ai_watermarks.identify import identify

    try:
        report = identify(Path(source))
    except (OSError, ValueError) as exc:
        return f"Could not read that image: {exc}"
    return format_report(report)


def clean_image(
    source: str | None,
    *,
    remove_visible_mark: bool,
    strip_metadata: bool,
    remove_synthid: bool,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[str | None, str]:
    """Run the selected stages for one upload.

    Returns ``(output_path, message)``; ``output_path`` is None when nothing was
    written. SynthID is gated here as well as in the UI so a direct caller cannot
    start a regeneration the host cannot finish.
    """
    if not source:
        return None, "Upload an image first."
    if not (remove_visible_mark or strip_metadata or remove_synthid):
        return None, "Select at least one action."

    caps = capabilities()
    if remove_visible_mark and not caps.visible:
        return None, f"Visible-mark removal needs the pixel runtime: pip install {VISIBLE_EXTRA}"
    if remove_synthid and not caps.invisible:
        return (
            None,
            f"SynthID image removal is unavailable on this host. {caps.invisible_reason} The image was not modified.",
        )

    from remove_ai_watermarks import api, metadata

    src = Path(source)
    out = _temp_output(src)

    def say(progress_value: float, message: str) -> None:
        if progress is not None:
            progress(progress_value, message)

    try:
        if remove_synthid:

            def report_stage(stage: str, detail: str) -> None:
                say(_STAGE_PROGRESS.get(stage, 0.5), detail)

            result = api.remove_all(src, out, force=True, progress=report_stage)
            lines: list[str] = []
            if result.visible_label:
                lines.append(f"Removed visible mark: {result.visible_label}.")
            lines.append(
                "SynthID scrub ran (pixels were regenerated)."
                if result.invisible == "removed"
                else f"SynthID stage outcome: {result.invisible}."
            )
            lines.append("AI metadata stripped.")
            lines.append(
                "SynthID removal cannot be verified locally; check the result with the "
                "provider's verifier. Review faces, text, and fine detail before reuse."
            )
            return str(result.output), " ".join(lines)

        if remove_visible_mark:
            say(0.25, "Removing visible marks...")
            _, removed = api.remove_visible(
                src,
                out,
                strip_metadata=strip_metadata,
                write_noop=True,
            )
            if removed:
                message = f"Removed visible mark(s): {', '.join(removed)}."
            else:
                message = "No known visible mark detected; pixels were left unchanged."
            if strip_metadata:
                message += " AI metadata stripped."
            return str(out), message

        say(0.25, "Stripping AI metadata...")
        metadata.remove_ai_metadata(src, out)
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"Error: {exc}"

    remaining = metadata.get_ai_metadata(out)
    if remaining:
        return (
            str(out),
            f"Stripped; {len(remaining)} marker(s) remain in the container: " + ", ".join(sorted(remaining)),
        )
    return str(out), "AI metadata stripped (pixels untouched)."


INTRO = """
# Remove AI Watermarks

Inspect provenance and clean content you generated or edited yourself. Upload a
PNG, JPEG, WebP, HEIC, or AVIF image. Metadata stripping (C2PA, EXIF, XMP, IPTC)
and visible-mark removal run on CPU; SynthID pixel removal needs NVIDIA CUDA and
regenerates the image.
"""


def build_app() -> Any:
    """Construct the Gradio app. Raises ImportError when the ``web`` extra is absent."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(f"the web UI needs the 'web' extra: pip install {WEB_EXTRA}") from exc

    caps = capabilities()
    capability_text = "\n".join(f"- {line}" for line in capability_lines())
    _prune_tmp_dir()

    with gr.Blocks(title="Remove AI Watermarks") as demo:
        gr.Markdown(INTRO)
        gr.Markdown(f"**This host**\n{capability_text}")

        with gr.Tab("Identify"):
            id_input = gr.Image(type="filepath", label="Image")
            id_button = gr.Button("Identify", variant="primary")
            id_output = gr.Textbox(label="Provenance report", lines=16)
            id_button.click(identify_image, id_input, id_output)

        with gr.Tab("Clean"):
            clean_input = gr.Image(type="filepath", label="Image")
            visible_box = gr.Checkbox(
                label="Remove visible AI marks (sparkle / text wordmarks)",
                value=caps.visible,
                interactive=caps.visible,
                info="" if caps.visible else f"Unavailable: pip install {VISIBLE_EXTRA}",
            )
            metadata_box = gr.Checkbox(
                label="Strip AI metadata (C2PA, EXIF, XMP, IPTC)",
                value=True,
            )
            synthid_box = gr.Checkbox(
                label="Remove SynthID (regenerate pixels with diffusion)",
                value=False,
                interactive=caps.invisible,
                info=(
                    "Runs the full pipeline: visible marks + SynthID + AI metadata."
                    if caps.invisible
                    else caps.invisible_reason
                ),
            )
            clean_button = gr.Button("Clean", variant="primary")
            clean_output = gr.Image(type="filepath", label="Result")
            clean_status = gr.Textbox(label="Status", lines=8)

            def run_clean(
                path: str | None,
                remove_visible_mark: bool,
                strip_ai_metadata: bool,
                remove_synthid: bool,
                progress: Any = gr.Progress(),  # noqa: B008 -- Gradio injects Progress from this default
            ) -> tuple[str | None, str]:
                return clean_image(
                    path,
                    remove_visible_mark=remove_visible_mark,
                    strip_metadata=strip_ai_metadata,
                    remove_synthid=remove_synthid,
                    progress=lambda value, message: progress(value, desc=message),
                )

            clean_button.click(
                run_clean,
                [clean_input, visible_box, metadata_box, synthid_box],
                [clean_output, clean_status],
            )

        gr.Markdown(
            "Cleaning cannot be verified locally for proprietary SynthID. Keep the "
            "original, process a copy, and confirm important files with the provider's "
            "verifier. A missing local signal is not proof an image is clean."
        )

    return demo
