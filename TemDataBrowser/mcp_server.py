"""Embedded FastMCP server plugin for TemDataBrowser.

Exposes typed MCP tools over Streamable HTTP so external clients (AI agents,
scripts, microscope controllers) can load disk files, display in-memory images
or arrays, navigate 3D stacks, adjust contrast, inspect TEM metadata, and fetch
rendered image previews — all marshaled safely onto the Qt GUI thread.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import io
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import imageio.v3 as iio
import numpy as np
from qtpy import QtCore, QtWidgets
from scipy.ndimage import zoom as ndimage_zoom

from ScopeFoundry.data_browser.data_browser_plug_in import DataBrowserPlugIn

try:
    from fastmcp import FastMCP
    from fastmcp.utilities.types import Image as FastMCPImage

    FASTMCP_AVAILABLE = True
except ImportError:
    FastMCP = None
    FastMCPImage = None
    FASTMCP_AVAILABLE = False


class _GuiCall:
    """Per-request synchronization container for cross-thread GUI dispatch."""

    __slots__ = ("fn", "event", "result", "error")

    def __init__(self, fn: Callable[[], Any]) -> None:
        self.fn = fn
        self.event = threading.Event()
        self.result: Any = None
        self.error: Optional[BaseException] = None


class GuiDispatcher(QtCore.QObject):
    """Concurrency-safe dispatcher that runs callables on the Qt main thread."""

    _call_requested = QtCore.Signal(object)

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._call_requested.connect(
            self._execute, QtCore.Qt.ConnectionType.QueuedConnection
        )

    @QtCore.Slot(object)
    def _execute(self, call: _GuiCall) -> None:
        try:
            call.result = call.fn()
        except BaseException as exc:  # pragma: no cover - surfaced to caller
            call.error = exc
        finally:
            call.event.set()

    def dispatch(self, fn: Callable[[], Any], timeout: float = 30.0) -> Any:
        """Execute *fn* on the GUI thread and return its result (or raise its error)."""
        if QtCore.QThread.currentThread() == self.thread():
            return fn()

        call = _GuiCall(fn)
        self._call_requested.emit(call)
        if not call.event.wait(timeout=timeout):
            raise TimeoutError(f"GUI operation timed out after {timeout:.1f}s")
        if call.error is not None:
            raise call.error
        return call.result


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert NumPy/Path/bytes values into JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    return obj


def _activate_view(app, view_name: str):
    """Switch DataBrowser's active view cleanly without firing redundant reload loops."""
    if view_name not in app.views:
        raise ValueError(
            f"Unknown view '{view_name}'. Available views: {list(app.views.keys())}"
        )
    target_view = app.views[view_name]
    # Set current_view BEFORE setup_view to work around ScopeFoundry's
    # DataBrowser.setup_view adding self.current_view.ui instead of view.ui.
    app.current_view = target_view
    app.setup_view(target_view)

    view_lq = app.settings.get_lq("view_name")
    view_lq.val = view_name
    view_lq.oldval = view_name
    combo = app.ui.view_name_comboBox
    combo.blockSignals(True)
    try:
        idx = combo.findText(view_name)
        if idx >= 0:
            combo.setCurrentIndex(idx)
    finally:
        combo.blockSignals(False)
    return target_view


def _get_active_image_view(app):
    """Return the active DataBrowserView if it exposes a PyQtGraph ImageView."""
    view = app.current_view
    if view is None or not hasattr(view, "imview") or view.imview is None:
        active_name = getattr(view, "name", None)
        raise ValueError(
            f"Active view '{active_name}' is not an image viewer. "
            "Load an image or switch to an image view first."
        )
    return view


def _get_slice_info(view) -> tuple[int, int]:
    """Return (current_slice, num_slices) for an image view."""
    imview = view.imview
    img = getattr(imview, "image", None)
    if img is None:
        return 0, 0
    t_axis = imview.axes.get("t") if hasattr(imview, "axes") else None
    if t_axis is not None and img.ndim > t_axis:
        return int(imview.currentIndex), int(img.shape[t_axis])
    return 0, 1


def _get_levels_list(view) -> Optional[List[float]]:
    """Return [vmin, vmax] for the active ImageView, or None if unavailable."""
    try:
        levels = view.imview.getLevels()
        if levels is None:
            return None
        if isinstance(levels, (tuple, list, np.ndarray)) and len(levels) == 2:
            # RGB images can return per-channel [[r0, r1], [g0, g1], [b0, b1]]
            if isinstance(levels[0], (int, float, np.number)):
                return [float(levels[0]), float(levels[1])]
            return _to_json_safe(levels)
        return _to_json_safe(levels)
    except Exception:
        return None


def _summarize_view_image(view) -> Dict[str, Any]:
    """Build a consistent image summary dict for a loaded view."""
    data = getattr(view, "data", None)
    info: Dict[str, Any] = {
        "view_name": view.name,
        "shape": list(data.shape) if isinstance(data, np.ndarray) else None,
        "dtype": str(data.dtype) if isinstance(data, np.ndarray) else None,
        "ndim": int(data.ndim) if isinstance(data, np.ndarray) else None,
        "pixel_size": _to_json_safe(getattr(view, "pixel_size", None)),
        "pixel_unit": getattr(view, "pixel_unit", None),
        "title": getattr(view, "title", None),
    }
    if hasattr(view, "imview") and view.imview is not None:
        current_slice, num_slices = _get_slice_info(view)
        info["current_slice"] = current_slice
        info["num_slices"] = num_slices
        info["levels"] = _get_levels_list(view)
    return info


def _bring_window_to_front(app) -> None:
    """Show and raise the TemDataBrowser main window."""
    if hasattr(app, "ui") and app.ui is not None:
        app.ui.show()
        app.ui.raise_()
        app.ui.activateWindow()


def _decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode base64-encoded PNG/TIFF/JPEG bytes or NumPy .npy bytes into an ndarray."""
    payload = image_base64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:
        raise ValueError(f"Invalid base64 image payload: {exc}") from exc

    if not raw:
        raise ValueError("Decoded base64 payload is empty.")

    # Auto-detect NumPy .npy magic header (\x93NUMPY)
    if raw.startswith(b"\x93NUMPY"):
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        return np.asarray(arr)

    try:
        return np.asarray(iio.imread(raw))
    except Exception as exc:
        raise ValueError(f"Failed to decode image bytes: {exc}") from exc


def _render_preview_png(view, max_size: int = 1024) -> bytes:
    """Render the currently displayed 2D frame of *view* to PNG bytes."""
    if max_size < 16:
        raise ValueError("max_size must be at least 16 pixels.")

    imview = view.imview
    img_item = imview.getImageItem()
    frame = getattr(img_item, "image", None)
    if frame is None:
        raise ValueError("No image is currently displayed.")

    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if arr.dtype == np.uint8:
            rgb = arr
        else:
            f = arr.astype(np.float64)
            lo = float(np.nanmin(f))
            hi = float(np.nanmax(f))
            if hi > lo:
                f = (f - lo) / (hi - lo) * 255.0
            else:
                f = np.zeros_like(f)
            rgb = np.clip(f, 0, 255).astype(np.uint8)
        out = rgb
    else:
        if arr.ndim > 2:
            arr = arr[0]
        f = np.nan_to_num(arr.astype(np.float64), nan=0.0)
        levels = _get_levels_list(view)
        if levels and len(levels) == 2 and isinstance(levels[0], (int, float)):
            lo, hi = float(levels[0]), float(levels[1])
        else:
            lo, hi = float(np.min(f)), float(np.max(f))
        if hi > lo:
            scaled = (f - lo) / (hi - lo) * 255.0
        else:
            scaled = np.zeros_like(f)
        out = np.clip(scaled, 0, 255).astype(np.uint8)

    h, w = out.shape[:2]
    longest = max(h, w)
    if longest > max_size:
        scale = max_size / float(longest)
        if out.ndim == 3:
            out = ndimage_zoom(out, (scale, scale, 1.0), order=1)
        else:
            out = ndimage_zoom(out, (scale, scale), order=1)
        out = np.clip(out, 0, 255).astype(np.uint8)

    return iio.imwrite("<bytes>", out, extension=".png")


class MCPServerThread(QtCore.QThread):
    """Background QThread running FastMCP over Streamable HTTP with its own asyncio loop."""

    server_started = QtCore.Signal()
    server_error = QtCore.Signal(str)
    server_stopped = QtCore.Signal()

    def __init__(
        self,
        app,
        dispatcher: GuiDispatcher,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        super().__init__()
        self.app = app
        self.dispatcher = dispatcher
        self.host = host
        self.port = int(port)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.mcp = None
        self._server_task: Optional[asyncio.Task] = None
        self._uvicorn_server = None
        self._running = False

    def run(self) -> None:
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            from . import __version__

            self.mcp = FastMCP(name="TemDataBrowser", version=__version__)
            self._register_tools()

            self._running = True
            self.server_started.emit()

            self.loop.run_until_complete(self._run_server())
        except Exception as exc:
            self.server_error.emit(str(exc))
        finally:
            self._running = False
            self._uvicorn_server = None
            if self.loop is not None:
                try:
                    pending = asyncio.all_tasks(self.loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        self.loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                finally:
                    self.loop.close()
                    self.loop = None
            self.server_stopped.emit()

    async def _run_server(self) -> None:
        self._server_task = asyncio.current_task()
        try:
            if hasattr(self.mcp, "http_app"):
                import uvicorn

                asgi_app = self.mcp.http_app(transport="http")
                config = uvicorn.Config(
                    asgi_app,
                    host=self.host,
                    port=self.port,
                    timeout_graceful_shutdown=2,
                    lifespan="on",
                    ws="websockets-sansio",
                    log_level="warning",
                )
                server = uvicorn.Server(config)
                # Never overwrite process signal handlers from a background QThread.
                server.install_signal_handlers = lambda: None
                self._uvicorn_server = server
                await server.serve()
            else:
                await self.mcp.run_http_async(host=self.host, port=self.port)
        except asyncio.CancelledError:
            pass

    def stop_server(self) -> None:
        """Request graceful shutdown of the FastMCP HTTP server."""
        server = self._uvicorn_server
        loop = self.loop
        if server is not None:
            server.should_exit = True
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(lambda: None)
            return

        task = self._server_task
        if loop is not None and loop.is_running() and task is not None:
            loop.call_soon_threadsafe(task.cancel)

    async def _on_gui(self, fn: Callable[[], Any], timeout: float = 30.0) -> Any:
        """Run *fn* on the Qt main thread without blocking the asyncio loop."""
        return await asyncio.to_thread(self.dispatcher.dispatch, fn, timeout)

    def _register_tools(self) -> None:
        from . import ArrayImageView, read_tem_metadata

        @self.mcp.tool()
        async def load_image(
            file_path: str,
            view_name: Optional[str] = None,
            update_browse_dir: bool = False,
            bring_to_front: bool = True,
        ) -> Dict[str, Any]:
            """Load and display an image or TEM data file from disk in TemDataBrowser.

            Supports TEM formats (.dm3, .dm4, .mrc, .ali, .rec, .emd, .ser, .img)
            and standard image formats (.png, .tif, .tiff, .jpg).

            Args:
                file_path: Path to the file to open.
                view_name: Optional explicit viewer name (auto-selected if omitted).
                update_browse_dir: If True, set the file browser root to the file's directory.
                bring_to_front: If True, raise the TemDataBrowser window.
            """
            try:
                path = Path(file_path).expanduser().resolve()
                if not path.is_file():
                    return {"success": False, "error": f"File not found: {path}"}

                def _do() -> Dict[str, Any]:
                    fname = str(path)
                    target_name = view_name or self.app.auto_select_view(fname)
                    target_view = _activate_view(self.app, target_name)

                    if update_browse_dir:
                        self.app.settings["browse_dir"] = str(path.parent)

                    # Sync data_filename without triggering DataBrowser's duplicate reload
                    lq = self.app.settings.get_lq("data_filename")
                    lq.val = fname
                    lq.oldval = fname
                    line_edit = self.app.ui.data_filename_lineEdit
                    line_edit.blockSignals(True)
                    try:
                        line_edit.setText(fname)
                    finally:
                        line_edit.blockSignals(False)

                    target_view.on_change_data_filename(fname)
                    self.app.on_change_data_filename_handle_plugins()

                    if bring_to_front:
                        _bring_window_to_front(self.app)

                    info = _summarize_view_image(target_view)
                    info["success"] = True
                    info["file_path"] = fname
                    return info

                return await self._on_gui(_do)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def show_image_base64(
            image_base64: str,
            title: str = "In-memory image",
            pixel_size: Optional[Union[float, List[float]]] = None,
            pixel_unit: Optional[str] = None,
            bring_to_front: bool = True,
        ) -> Dict[str, Any]:
            """Display an in-memory image encoded as base64 in TemDataBrowser.

            Accepts base64-encoded PNG, TIFF (including 16-bit and float32), JPEG,
            data URIs ('data:image/png;base64,...'), or NumPy '.npy' byte payloads.

            Args:
                image_base64: Base64-encoded image or .npy bytes.
                title: Display title shown above the image.
                pixel_size: Optional pixel size (scalar float or [xscale, yscale]).
                pixel_unit: Optional physical unit (e.g. 'nm', 'um', 'A', '1/nm', 'mrad').
                bring_to_front: If True, raise the TemDataBrowser window.
            """
            try:
                arr = _decode_base64_image(image_base64)

                def _do() -> Dict[str, Any]:
                    target_view = _activate_view(self.app, ArrayImageView.name)
                    target_view.set_array(
                        arr,
                        title=title,
                        pixel_size=pixel_size,
                        pixel_unit=pixel_unit,
                    )
                    if bring_to_front:
                        _bring_window_to_front(self.app)
                    info = _summarize_view_image(target_view)
                    info["success"] = True
                    return info

                return await self._on_gui(_do)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def show_array(
            data: Optional[List[Any]] = None,
            raw_base64: Optional[str] = None,
            shape: Optional[List[int]] = None,
            dtype: str = "float32",
            title: str = "In-memory array",
            pixel_size: Optional[Union[float, List[float]]] = None,
            pixel_unit: Optional[str] = None,
            bring_to_front: bool = True,
        ) -> Dict[str, Any]:
            """Display a 2D or 3D numeric array in TemDataBrowser without writing a file.

            Provide either *data* (nested numeric list) OR *raw_base64* (C-contiguous
            bytes of a numpy array) together with *shape* and *dtype*.

            Args:
                data: Nested 2D or 3D list of numbers.
                raw_base64: Base64-encoded raw C-contiguous array bytes.
                shape: Required when using raw_base64 (e.g. [512, 512] or [10, 256, 256]).
                dtype: NumPy dtype string (default: 'float32').
                title: Display title shown above the image.
                pixel_size: Optional pixel size (scalar float or [xscale, yscale]).
                pixel_unit: Optional physical unit (e.g. 'nm', 'um', 'A', '1/nm').
                bring_to_front: If True, raise the TemDataBrowser window.
            """
            try:
                if (data is None) == (raw_base64 is None):
                    return {
                        "success": False,
                        "error": "Provide exactly one of 'data' or 'raw_base64'.",
                    }

                np_dtype = np.dtype(dtype)
                if raw_base64 is not None:
                    if not shape or len(shape) not in (2, 3):
                        return {
                            "success": False,
                            "error": "'shape' must be a 2- or 3-element list when using 'raw_base64'.",
                        }
                    raw = base64.b64decode(raw_base64.strip(), validate=False)
                    expected_bytes = int(np.prod(shape)) * np_dtype.itemsize
                    if len(raw) != expected_bytes:
                        return {
                            "success": False,
                            "error": (
                                f"Byte length mismatch: got {len(raw)} bytes, "
                                f"expected {expected_bytes} for shape={shape}, dtype={np_dtype}."
                            ),
                        }
                    arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
                else:
                    arr = np.asarray(data, dtype=np_dtype)

                def _do() -> Dict[str, Any]:
                    target_view = _activate_view(self.app, ArrayImageView.name)
                    target_view.set_array(
                        arr,
                        title=title,
                        pixel_size=pixel_size,
                        pixel_unit=pixel_unit,
                    )
                    if bring_to_front:
                        _bring_window_to_front(self.app)
                    info = _summarize_view_image(target_view)
                    info["success"] = True
                    return info

                return await self._on_gui(_do)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def set_slice(index: int) -> Dict[str, Any]:
            """Select a slice index in a 3D image stack (e.g., tomography tilt series).

            Args:
                index: Zero-based slice index.
            """
            try:
                def _do() -> Dict[str, Any]:
                    view = _get_active_image_view(self.app)
                    _cur, num_slices = _get_slice_info(view)
                    if num_slices <= 1:
                        raise ValueError(
                            f"Current image is not a 3D stack (num_slices={num_slices})."
                        )
                    if index < 0 or index >= num_slices:
                        raise ValueError(
                            f"Slice index {index} out of range [0, {num_slices - 1}]."
                        )
                    view.imview.setCurrentIndex(int(index))
                    cur, total = _get_slice_info(view)
                    return {
                        "success": True,
                        "current_slice": cur,
                        "num_slices": total,
                    }

                return await self._on_gui(_do)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def set_contrast(
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            auto: bool = False,
        ) -> Dict[str, Any]:
            """Adjust contrast/intensity display levels on the active image view.

            Args:
                vmin: Minimum display intensity (required unless auto=True).
                vmax: Maximum display intensity (required unless auto=True).
                auto: If True, automatically scale contrast to the image range.
            """
            try:
                if not auto and (vmin is None or vmax is None):
                    return {
                        "success": False,
                        "error": "Provide both 'vmin' and 'vmax', or set 'auto=True'.",
                    }

                def _do() -> Dict[str, Any]:
                    view = _get_active_image_view(self.app)
                    if auto:
                        view.imview.autoLevels()
                    else:
                        view.imview.setLevels(float(vmin), float(vmax))
                    return {
                        "success": True,
                        "levels": _get_levels_list(view),
                    }

                return await self._on_gui(_do)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def reset_zoom() -> Dict[str, Any]:
            """Reset zoom and pan to fit the full image in the active image view."""
            try:
                def _do() -> Dict[str, Any]:
                    view = _get_active_image_view(self.app)
                    if hasattr(view, "plt") and view.plt is not None:
                        view.plt.autoRange()
                    else:
                        view.imview.autoRange()
                    return {"success": True, "view_name": view.name}

                return await self._on_gui(_do)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def list_views() -> Dict[str, Any]:
            """List all registered DataBrowser views and the currently active view."""
            try:
                def _do() -> Dict[str, Any]:
                    return {
                        "success": True,
                        "active_view": getattr(self.app.current_view, "name", None),
                        "views": list(self.app.views.keys()),
                    }

                return await self._on_gui(_do)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def set_view(view_name: str) -> Dict[str, Any]:
            """Switch the active DataBrowser view and reload the current file if applicable.

            Args:
                view_name: Name of the target view (from list_views).
            """
            try:
                def _do() -> Dict[str, Any]:
                    target_view = _activate_view(self.app, view_name)
                    fname = self.app.settings["data_filename"]
                    if (
                        target_view.name != ArrayImageView.name
                        and fname
                        and Path(fname).is_file()
                    ):
                        target_view.on_change_data_filename(fname)
                    info = _summarize_view_image(target_view)
                    info["success"] = True
                    return info

                return await self._on_gui(_do)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def set_browse_dir(
            directory: str,
            file_filter: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Set the root directory (and optional glob filter) of the file tree browser.

            Args:
                directory: Directory path to browse.
                file_filter: Optional comma-separated glob pattern (e.g. '*.dm4, *.mrc').
            """
            try:
                dir_path = Path(directory).expanduser().resolve()
                if not dir_path.is_dir():
                    return {"success": False, "error": f"Directory not found: {dir_path}"}

                def _do() -> Dict[str, Any]:
                    self.app.settings["browse_dir"] = str(dir_path)
                    if file_filter is not None:
                        self.app.settings["file_filter"] = file_filter
                    return {
                        "success": True,
                        "browse_dir": str(dir_path),
                        "file_filter": self.app.settings["file_filter"],
                    }

                return await self._on_gui(_do)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def list_files(
            directory: Optional[str] = None,
            supported_only: bool = True,
            pattern: Optional[str] = None,
        ) -> Dict[str, Any]:
            """List files in a directory along with their auto-detected TemDataBrowser view.

            Args:
                directory: Target directory (defaults to current browse_dir).
                supported_only: If True, only include files supported by a TEM/image viewer.
                pattern: Optional glob pattern (e.g. '*.dm4').
            """
            try:
                def _get_dir_and_views():
                    d = directory or self.app.settings["browse_dir"]
                    return str(d), list(self.app.views.items())

                target_dir_str, views_items = await self._on_gui(_get_dir_and_views)
                dir_path = Path(target_dir_str).expanduser().resolve()
                if not dir_path.is_dir():
                    return {"success": False, "error": f"Directory not found: {dir_path}"}

                def _detect_view(fname: str) -> str:
                    for vname, vobj in reversed(views_items):
                        if vobj.is_file_supported(fname):
                            return vname
                    return "file_info"

                entries: List[Dict[str, Any]] = []
                for child in sorted(dir_path.iterdir(), key=lambda p: p.name.lower()):
                    if not child.is_file():
                        continue
                    if pattern and not fnmatch.fnmatch(child.name, pattern):
                        continue
                    detected = _detect_view(str(child))
                    if supported_only and detected == "file_info":
                        continue
                    entries.append(
                        {
                            "name": child.name,
                            "path": str(child),
                            "extension": child.suffix.lower(),
                            "size_bytes": child.stat().st_size,
                            "view_name": detected,
                        }
                    )

                return {
                    "success": True,
                    "directory": str(dir_path),
                    "count": len(entries),
                    "files": entries,
                }
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def get_current_state() -> Dict[str, Any]:
            """Return the current state of TemDataBrowser (file, view, image stats, contrast)."""
            try:
                def _do() -> Dict[str, Any]:
                    view = self.app.current_view
                    summary = _summarize_view_image(view) if view is not None else {}
                    return {
                        "success": True,
                        "data_filename": str(self.app.settings["data_filename"]),
                        "browse_dir": str(self.app.settings["browse_dir"]),
                        "file_filter": str(self.app.settings["file_filter"]),
                        "auto_select_view": bool(self.app.settings["auto_select_view"]),
                        "active_view": getattr(view, "name", None),
                        "available_views": list(self.app.views.keys()),
                        **summary,
                    }

                return await self._on_gui(_do)
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def get_metadata(file_path: Optional[str] = None) -> Dict[str, Any]:
            """Read parsed TEM metadata for a file (defaults to the currently loaded file).

            Args:
                file_path: Optional file path; uses currently loaded data_filename if omitted.
            """
            try:
                if file_path is None:
                    file_path = await self._on_gui(
                        lambda: str(self.app.settings["data_filename"])
                    )
                if not file_path:
                    return {"success": False, "error": "No file is currently loaded."}

                path = Path(file_path).expanduser().resolve()
                if not path.is_file():
                    return {"success": False, "error": f"File not found: {path}"}

                meta = await asyncio.to_thread(read_tem_metadata, str(path))
                return {
                    "success": True,
                    "file_path": str(path),
                    "metadata": _to_json_safe(meta),
                }
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        @self.mcp.tool()
        async def get_current_image_preview(max_size: int = 1024):
            """Return a rendered PNG preview of the currently displayed image/slice.

            Args:
                max_size: Maximum width or height in pixels for the returned PNG (default: 1024).
            """
            def _do() -> bytes:
                view = _get_active_image_view(self.app)
                return _render_preview_png(view, max_size=max_size)

            png_bytes = await self._on_gui(_do)
            return FastMCPImage(data=png_bytes, format="png")


class MCPServerPlugIn(DataBrowserPlugIn):
    """ScopeFoundry DataBrowserPlugIn that manages the embedded FastMCP server."""

    name = "mcp_server"
    button_text = "MCP"
    show_keyboard_key = QtCore.Qt.Key_M
    description = "Embedded MCP HTTP server for remote image display (Ctrl + M)"

    def __init__(
        self,
        databrowser,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        super().__init__(databrowser)
        if not FASTMCP_AVAILABLE:
            raise ImportError(
                "FastMCP is not installed. Install with: pip install 'TemDataBrowser[mcp]'"
            )

        # Initialize settings and GUI dispatcher immediately (DataBrowserPlugIn.setup()
        # is lazy and only runs when the user clicks the plugin toolbar button).
        self.settings.New("host", dtype=str, initial=host)
        self.settings.New("port", dtype=int, initial=int(port), vmin=1, vmax=65535)
        self.settings.New("running", dtype=bool, initial=False, ro=True)
        self.settings.New("last_error", dtype=str, initial="", ro=True)

        self.dispatcher = GuiDispatcher(parent=self.q_object)
        self.mcp_thread: Optional[MCPServerThread] = None

        self._toggle_btn: Optional[QtWidgets.QPushButton] = None
        self._status_label: Optional[QtWidgets.QLabel] = None
        self._url_line: Optional[QtWidgets.QLineEdit] = None
        self._error_label: Optional[QtWidgets.QLabel] = None

        if hasattr(self.databrowser, "qtapp") and self.databrowser.qtapp is not None:
            self.databrowser.qtapp.aboutToQuit.connect(self.stop_server)

    @property
    def endpoint_url(self) -> str:
        return f"http://{self.settings['host']}:{self.settings['port']}/mcp"

    @property
    def is_running(self) -> bool:
        return bool(self.mcp_thread is not None and self.mcp_thread.isRunning())

    def setup(self) -> None:
        """Build the collapsible MCP server control panel."""
        self.ui = QtWidgets.QGroupBox("MCP Server")
        layout = QtWidgets.QGridLayout(self.ui)

        host_lq = self.settings.get_lq("host")
        port_lq = self.settings.get_lq("port")
        host_edit = host_lq.new_default_widget()
        port_spin = port_lq.new_default_widget()
        host_lq.add_listener(self._refresh_ui)
        port_lq.add_listener(self._refresh_ui)

        self._toggle_btn = QtWidgets.QPushButton()
        self._toggle_btn.clicked.connect(self.toggle_server)

        self._status_label = QtWidgets.QLabel()
        self._url_line = QtWidgets.QLineEdit()
        self._url_line.setReadOnly(True)
        self._error_label = QtWidgets.QLabel()
        self._error_label.setStyleSheet("color: red;")

        layout.addWidget(QtWidgets.QLabel("Host:"), 0, 0)
        layout.addWidget(host_edit, 0, 1)
        layout.addWidget(QtWidgets.QLabel("Port:"), 0, 2)
        layout.addWidget(port_spin, 0, 3)
        layout.addWidget(self._toggle_btn, 0, 4)

        layout.addWidget(QtWidgets.QLabel("Endpoint:"), 1, 0)
        layout.addWidget(self._url_line, 1, 1, 1, 3)
        layout.addWidget(self._status_label, 1, 4)
        layout.addWidget(self._error_label, 2, 0, 1, 5)

        self._refresh_ui()

    def update(self, fname: Optional[str] = None) -> None:
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        running = bool(self.settings["running"])
        if self._toggle_btn is not None:
            self._toggle_btn.setText("Stop Server" if running else "Start Server")
        if self._status_label is not None:
            self._status_label.setText("Running" if running else "Stopped")
            color = "green" if running else "gray"
            self._status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        if self._url_line is not None:
            self._url_line.setText(self.endpoint_url)
        if self._error_label is not None:
            err = self.settings["last_error"]
            self._error_label.setText(err)
            self._error_label.setVisible(bool(err))

    def toggle_server(self) -> None:
        if self.is_running:
            self.stop_server()
        else:
            self.start_server()

    def start_server(self) -> None:
        """Start the embedded MCP HTTP server on a background QThread."""
        if self.is_running:
            return

        host = str(self.settings["host"])
        port = int(self.settings["port"])

        self.settings.get_lq("last_error").update_value("")
        self.mcp_thread = MCPServerThread(
            app=self.databrowser,
            dispatcher=self.dispatcher,
            host=host,
            port=port,
        )
        self.mcp_thread.server_started.connect(self._on_server_started)
        self.mcp_thread.server_error.connect(self._on_server_error)
        self.mcp_thread.server_stopped.connect(self._on_server_stopped)
        self.mcp_thread.start()

    def stop_server(self) -> None:
        """Gracefully stop the background MCP server thread."""
        thread = self.mcp_thread
        if thread is not None:
            if thread.isRunning():
                thread.stop_server()
                if not thread.wait(5000):
                    thread.terminate()
                    thread.wait(2000)
            self.mcp_thread = None
        self.settings.get_lq("running").update_value(False)
        self._refresh_ui()

    def _on_server_started(self) -> None:
        self.settings.get_lq("running").update_value(True)
        self.settings.get_lq("last_error").update_value("")
        self._refresh_ui()

    def _on_server_error(self, error_msg: str) -> None:
        self.settings.get_lq("running").update_value(False)
        self.settings.get_lq("last_error").update_value(error_msg)
        self._refresh_ui()

    def _on_server_stopped(self) -> None:
        self.settings.get_lq("running").update_value(False)
        self._refresh_ui()
