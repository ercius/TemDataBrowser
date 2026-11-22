"""End-to-end headless Qt + FastMCP integration tests for TemDataBrowser."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import socket
import threading
import time
from pathlib import Path

import imageio.v3 as iio
import ncempy.io.mrc
import numpy as np
import pytest
from fastmcp.client import Client
from qtpy import QtCore, QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from TemDataBrowser import ArrayImageView, create_app, parse_mcp_args


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_port(host: str, port: int, qtapp: QtWidgets.QApplication, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qtapp.processEvents()
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"MCP server did not start listening on {host}:{port} within {timeout}s")


def _extract_dict(result) -> dict:
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    if hasattr(result, "data") and isinstance(result.data, dict):
        return result.data
    return json.loads(result.content[0].text)


def _run_client_coro(qtapp: QtWidgets.QApplication, coro):
    """Run an async FastMCP client coroutine in a worker thread while pumping Qt events."""
    done = threading.Event()
    holder: dict = {}

    def _runner():
        try:
            holder["result"] = asyncio.run(coro)
        except BaseException as exc:
            holder["error"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    while not done.is_set():
        qtapp.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.01)
    thread.join(timeout=5.0)
    qtapp.processEvents()

    if "error" in holder:
        raise holder["error"]
    return holder.get("result")


@pytest.fixture(scope="module")
def tem_mcp_env(tmp_path_factory):
    """Launch a headless TemDataBrowser instance with MCP server on an ephemeral port."""
    port = _find_free_port()
    tmp_dir = tmp_path_factory.mktemp("tem_mcp_data")

    app, mcp_plugin = create_app(
        [
            "--mcp",
            "--mcp-host",
            "127.0.0.1",
            "--mcp-port",
            str(port),
            "--browse_dir",
            str(tmp_dir),
        ]
    )
    assert mcp_plugin is not None
    _wait_for_port("127.0.0.1", port, app.qtapp)

    # Create sample synthetic test files on disk
    png_path = tmp_dir / "sample.png"
    png_arr = np.linspace(0, 255, 64 * 48, dtype=np.uint8).reshape(48, 64)
    iio.imwrite(png_path, png_arr)

    mrc_path = tmp_dir / "tilt_series.mrc"
    mrc_stack = np.zeros((5, 32, 40), dtype=np.float32)
    for i in range(5):
        mrc_stack[i] = float(i + 1) * 10.0 + np.arange(32 * 40, dtype=np.float32).reshape(32, 40)
    ncempy.io.mrc.mrcWriter(str(mrc_path), mrc_stack, pixelSize=(0.25, 0.25, 0.25))

    txt_path = tmp_dir / "notes.txt"
    txt_path.write_text("hello tem")

    yield {
        "app": app,
        "plugin": mcp_plugin,
        "url": f"http://127.0.0.1:{port}/mcp",
        "tmp_dir": tmp_dir,
        "png_path": png_path,
        "mrc_path": mrc_path,
        "txt_path": txt_path,
    }

    mcp_plugin.stop_server()
    app.qtapp.processEvents()
    assert not mcp_plugin.is_running

    # Cleanly tear down Qt widgets and QFileSystemModel before Python exit
    app.tree_view.setModel(None)
    app.fs_model.deleteLater()
    app.ui.close()
    app.ui.deleteLater()
    app.qtapp.processEvents()


def test_parse_mcp_args():
    mcp_args, remaining = parse_mcp_args(
        ["--mcp", "--mcp-host", "0.0.0.0", "--mcp-port", "9123", "--browse_dir", "/tmp"]
    )
    assert mcp_args.mcp is True
    assert mcp_args.mcp_host == "0.0.0.0"
    assert mcp_args.mcp_port == 9123
    assert remaining == ["--browse_dir", "/tmp"]

    defaults, rem2 = parse_mcp_args([])
    assert defaults.mcp is False
    assert defaults.mcp_host == "127.0.0.1"
    assert defaults.mcp_port == 8000
    assert rem2 == []


def test_tool_discovery(tem_mcp_env):
    async def _run():
        async with Client(tem_mcp_env["url"]) as client:
            tools = await client.list_tools()
            return {t.name for t in tools}

    tool_names = _run_client_coro(tem_mcp_env["app"].qtapp, _run())
    expected = {
        "load_image",
        "show_image_base64",
        "show_array",
        "set_slice",
        "set_contrast",
        "reset_zoom",
        "list_views",
        "set_view",
        "set_browse_dir",
        "list_files",
        "get_current_state",
        "get_metadata",
        "get_current_image_preview",
    }
    assert expected.issubset(tool_names)


def test_browse_and_list_files(tem_mcp_env):
    tmp_dir = tem_mcp_env["tmp_dir"]

    async def _run():
        async with Client(tem_mcp_env["url"]) as client:
            b = _extract_dict(
                await client.call_tool("set_browse_dir", {"directory": str(tmp_dir)})
            )
            supported = _extract_dict(
                await client.call_tool("list_files", {"supported_only": True})
            )
            all_files = _extract_dict(
                await client.call_tool("list_files", {"supported_only": False})
            )
            return b, supported, all_files

    b, supported, all_files = _run_client_coro(tem_mcp_env["app"].qtapp, _run())
    assert b["success"] is True
    assert supported["success"] is True
    names = {f["name"] for f in supported["files"]}
    assert "sample.png" in names
    assert "tilt_series.mrc" in names
    assert "notes.txt" not in names
    assert "notes.txt" in {f["name"] for f in all_files["files"]}


def test_load_png_and_mrc_and_controls(tem_mcp_env):
    png_path = tem_mcp_env["png_path"]
    mrc_path = tem_mcp_env["mrc_path"]

    async def _run():
        async with Client(tem_mcp_env["url"]) as client:
            png_res = _extract_dict(
                await client.call_tool("load_image", {"file_path": str(png_path)})
            )
            mrc_res = _extract_dict(
                await client.call_tool("load_image", {"file_path": str(mrc_path)})
            )
            slice_res = _extract_dict(
                await client.call_tool("set_slice", {"index": 3})
            )
            contrast_res = _extract_dict(
                await client.call_tool("set_contrast", {"vmin": 15.0, "vmax": 250.0})
            )
            zoom_res = _extract_dict(await client.call_tool("reset_zoom", {}))
            meta_res = _extract_dict(await client.call_tool("get_metadata", {}))
            state_res = _extract_dict(await client.call_tool("get_current_state", {}))
            return png_res, mrc_res, slice_res, contrast_res, zoom_res, meta_res, state_res

    png_res, mrc_res, slice_res, contrast_res, zoom_res, meta_res, state_res = (
        _run_client_coro(tem_mcp_env["app"].qtapp, _run())
    )

    assert png_res["success"] is True
    assert png_res["view_name"] == "Image viewer (imageio)"
    assert png_res["shape"] == [48, 64]

    assert mrc_res["success"] is True
    assert mrc_res["view_name"] == "TEM data viewer"
    assert mrc_res["shape"] == [5, 32, 40]
    assert mrc_res["num_slices"] == 5

    assert slice_res == {"success": True, "current_slice": 3, "num_slices": 5}
    assert contrast_res["success"] is True
    assert pytest.approx(contrast_res["levels"], rel=1e-3) == [15.0, 250.0]
    assert zoom_res["success"] is True

    assert meta_res["success"] is True
    assert isinstance(meta_res["metadata"], dict)

    assert state_res["success"] is True
    assert state_res["active_view"] == "TEM data viewer"
    assert state_res["current_slice"] == 3
    assert Path(state_res["data_filename"]) == mrc_path


def test_show_in_memory_images_and_preview(tem_mcp_env):
    # 1) Base64 PNG
    img_2d = np.arange(32 * 48, dtype=np.uint8).reshape(32, 48)
    png_b64 = base64.b64encode(iio.imwrite("<bytes>", img_2d, extension=".png")).decode()

    # 2) Base64 .npy 3D float32 stack
    stack_3d = np.random.default_rng(42).normal(size=(4, 24, 30)).astype(np.float32)
    npy_buf = io.BytesIO()
    np.save(npy_buf, stack_3d)
    npy_b64 = base64.b64encode(npy_buf.getvalue()).decode()

    # 3) Raw C-contiguous float32 bytes
    raw_arr = np.linspace(-1.0, 1.0, 20 * 25, dtype=np.float32).reshape(20, 25)
    raw_b64 = base64.b64encode(raw_arr.tobytes()).decode()

    # 4) Nested JSON list
    small_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

    async def _run():
        async with Client(tem_mcp_env["url"]) as client:
            r_png = _extract_dict(
                await client.call_tool(
                    "show_image_base64",
                    {
                        "image_base64": f"data:image/png;base64,{png_b64}",
                        "title": "Synthetic PNG",
                        "pixel_size": 0.5,
                        "pixel_unit": "nm",
                    },
                )
            )
            r_npy = _extract_dict(
                await client.call_tool(
                    "show_image_base64",
                    {
                        "image_base64": npy_b64,
                        "title": "NPY 3D Stack",
                        "pixel_size": [0.1, 0.2],
                        "pixel_unit": "nm",
                    },
                )
            )
            r_slice = _extract_dict(await client.call_tool("set_slice", {"index": 2}))
            r_raw = _extract_dict(
                await client.call_tool(
                    "show_array",
                    {
                        "raw_base64": raw_b64,
                        "shape": [20, 25],
                        "dtype": "float32",
                        "title": "Raw Buffer",
                        "pixel_size": 0.05,
                        "pixel_unit": "1/nm",
                    },
                )
            )
            r_list = _extract_dict(
                await client.call_tool(
                    "show_array",
                    {"data": small_list, "dtype": "float64", "title": "Small List"},
                )
            )
            preview = await client.call_tool("get_current_image_preview", {"max_size": 128})
            return r_png, r_npy, r_slice, r_raw, r_list, preview

    r_png, r_npy, r_slice, r_raw, r_list, preview = _run_client_coro(
        tem_mcp_env["app"].qtapp, _run()
    )

    assert r_png["success"] is True
    assert r_png["view_name"] == ArrayImageView.name
    assert r_png["shape"] == [32, 48]
    assert r_png["pixel_size"] == [0.5, 0.5]
    assert r_png["pixel_unit"] == "nm"

    assert r_npy["success"] is True
    assert r_npy["shape"] == [4, 24, 30]
    assert r_npy["num_slices"] == 4
    assert r_slice == {"success": True, "current_slice": 2, "num_slices": 4}

    assert r_raw["success"] is True
    assert r_raw["shape"] == [20, 25]
    assert r_raw["pixel_unit"] == "1/nm"

    assert r_list["success"] is True
    assert r_list["shape"] == [2, 3]
    assert r_list["dtype"] == "float64"

    # Verify returned FastMCP Image content is valid PNG bytes
    img_block = preview.content[0]
    assert img_block.type == "image"
    mime_type = getattr(img_block, "mime_type", None) or getattr(img_block, "mimeType", None)
    assert mime_type == "image/png"
    decoded_preview = iio.imread(base64.b64decode(img_block.data))
    assert decoded_preview.shape[:2] == (2, 3)


def test_plugin_ui_and_set_view(tem_mcp_env):
    plugin = tem_mcp_env["plugin"]
    app = tem_mcp_env["app"]

    # Exercise lazy UI setup and refresh
    plugin.show()
    app.qtapp.processEvents()
    assert plugin.ui is not None
    assert plugin._status_label.text() == "Running"
    assert plugin._url_line.text() == tem_mcp_env["url"]
    plugin.hide()
    app.qtapp.processEvents()

    async def _run():
        async with Client(tem_mcp_env["url"]) as client:
            meta_view = _extract_dict(
                await client.call_tool("set_view", {"view_name": "TEM metadata viewer"})
            )
            tem_view = _extract_dict(
                await client.call_tool("set_view", {"view_name": "TEM data viewer"})
            )
            return meta_view, tem_view

    meta_view, tem_view = _run_client_coro(app.qtapp, _run())
    assert meta_view["success"] is True
    assert meta_view["view_name"] == "TEM metadata viewer"
    assert tem_view["success"] is True
    assert tem_view["view_name"] == "TEM data viewer"


def test_concurrency_and_error_handling(tem_mcp_env):
    async def _run():
        async with Client(tem_mcp_env["url"]) as client:
            # Concurrent GUI-dispatched calls
            results = await asyncio.gather(
                client.call_tool("list_views", {}),
                client.call_tool("get_current_state", {}),
                client.call_tool("set_contrast", {"auto": True}),
                client.call_tool("reset_zoom", {}),
            )
            # Error cases
            missing = _extract_dict(
                await client.call_tool("load_image", {"file_path": "/nonexistent/file.dm4"})
            )
            bad_view = _extract_dict(
                await client.call_tool("set_view", {"view_name": "no_such_view"})
            )
            bad_slice = _extract_dict(
                await client.call_tool("set_slice", {"index": 999})
            )
            bad_b64 = _extract_dict(
                await client.call_tool("show_image_base64", {"image_base64": "not-an-image!!!"})
            )
            return [_extract_dict(r) for r in results], missing, bad_view, bad_slice, bad_b64

    concurrent_res, missing, bad_view, bad_slice, bad_b64 = _run_client_coro(
        tem_mcp_env["app"].qtapp, _run()
    )

    assert all(r["success"] is True for r in concurrent_res)
    assert missing["success"] is False
    assert bad_view["success"] is False
    assert bad_slice["success"] is False
    assert bad_b64["success"] is False
