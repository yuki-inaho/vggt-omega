from __future__ import annotations

import socket
from pathlib import Path

import pytest

import scripts.visualize as visualize_cli


def test_screenshot_with_rrd_input_skips_inference(monkeypatch, tmp_path: Path) -> None:
    rrd = tmp_path / "scene.rrd"
    rrd.write_bytes(b"rrd")
    png = tmp_path / "scene.png"
    calls: dict[str, Path] = {}

    def fail_rrd(*args, **kwargs):
        raise AssertionError("inference should be skipped when --rrd-input is provided")

    def fake_capture(rrd_path: Path, out_png: Path, args) -> None:
        calls["rrd"] = rrd_path
        calls["png"] = out_png
        out_png.write_bytes(b"png")

    monkeypatch.setattr(visualize_cli, "_mode_rrd", fail_rrd)
    monkeypatch.setattr(visualize_cli, "_capture_with_playwright", fake_capture)

    code = visualize_cli.main(["--mode", "screenshot", "--rrd-input", str(rrd), "--output", str(png)])

    assert code == 0
    assert calls == {"rrd": rrd, "png": png}


def test_screenshot_without_rrd_input_uses_separate_rrd_output(monkeypatch, tmp_path: Path) -> None:
    png = tmp_path / "scene.png"
    calls: dict[str, Path] = {}

    def fake_rrd(args, output: Path | None = None) -> Path:
        assert output is not None
        calls["rrd_output"] = output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"rrd")
        return output

    def fake_capture(rrd_path: Path, out_png: Path, args) -> None:
        calls["rrd"] = rrd_path
        calls["png"] = out_png
        out_png.write_bytes(b"png")

    monkeypatch.setattr(visualize_cli, "_mode_rrd", fake_rrd)
    monkeypatch.setattr(visualize_cli, "_capture_with_playwright", fake_capture)

    code = visualize_cli.main(
        [
            "--mode",
            "screenshot",
            "--checkpoint",
            "dummy.pt",
            "--video",
            "input.mp4",
            "--output",
            str(png),
        ]
    )

    assert code == 0
    assert calls["rrd_output"] == png.with_suffix(".rrd")
    assert calls["rrd"] == png.with_suffix(".rrd")
    assert calls["png"] == png


def test_screenshot_honors_explicit_rrd_output(monkeypatch, tmp_path: Path) -> None:
    rrd = tmp_path / "custom.rrd"
    png = tmp_path / "scene.png"
    calls: dict[str, Path] = {}

    def fake_rrd(args, output: Path | None = None) -> Path:
        assert output is not None
        output.write_bytes(b"rrd")
        calls["rrd_output"] = output
        return output

    monkeypatch.setattr(visualize_cli, "_mode_rrd", fake_rrd)
    monkeypatch.setattr(visualize_cli, "_capture_with_playwright", lambda rrd_path, out_png, args: None)

    visualize_cli.main(
        [
            "--mode",
            "screenshot",
            "--checkpoint",
            "dummy.pt",
            "--video",
            "input.mp4",
            "--rrd-output",
            str(rrd),
            "--output",
            str(png),
        ]
    )

    assert calls["rrd_output"] == rrd


def test_rrd_input_is_screenshot_only() -> None:
    with pytest.raises(SystemExit, match="--rrd-input is only valid"):
        visualize_cli.main(["--mode", "rrd", "--rrd-input", "existing.rrd"])


def test_rrd_mode_honors_no_accumulate_points(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "scene.rrd"
    calls: dict[str, bool] = {}

    monkeypatch.setattr(visualize_cli, "_run_inference", lambda args: object())

    def fake_save_results_to_rrd(*args, **kwargs) -> Path:
        calls["accumulate_points"] = kwargs["accumulate_points"]
        out.write_bytes(b"rrd")
        return out

    monkeypatch.setattr(visualize_cli, "save_results_to_rrd", fake_save_results_to_rrd)

    code = visualize_cli.main(
        [
            "--mode",
            "rrd",
            "--checkpoint",
            "dummy.pt",
            "--video",
            "input.mp4",
            "--output",
            str(out),
            "--no-accumulate-points",
        ]
    )

    assert code == 0
    assert calls == {"accumulate_points": False}


def test_capture_rejects_missing_rrd(tmp_path: Path) -> None:
    args = visualize_cli._build_parser().parse_args(["--mode", "screenshot", "--rrd-input", "missing.rrd"])
    with pytest.raises(SystemExit, match="RRD input not found"):
        visualize_cli._capture_with_playwright(tmp_path / "missing.rrd", tmp_path / "out.png", args)


def test_raise_if_viewer_error_detects_rerun_error_page() -> None:
    class FakeLocator:
        def inner_text(self, timeout: int) -> str:
            return "An error occurred during loading:\nfailed to create surface"

    class FakePage:
        def locator(self, selector: str) -> FakeLocator:
            assert selector == "body"
            return FakeLocator()

    with pytest.raises(SystemExit, match="Rerun web viewer failed"):
        visualize_cli._raise_if_viewer_error(FakePage())


def test_resolve_ports_moves_off_busy_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        busy_port = sock.getsockname()[1]
        resolved_web, resolved_grpc = visualize_cli._resolve_ports(busy_port, busy_port + 1, strict=False)

    assert resolved_web != busy_port
    assert resolved_grpc >= busy_port + 1


def test_resolve_ports_strict_rejects_busy_port() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        busy_port = sock.getsockname()[1]
        with pytest.raises(SystemExit, match="already in use"):
            visualize_cli._resolve_ports(busy_port, busy_port + 1, strict=True)
