"""Headless Qt tests for TemDataBrowser quick fixes:
- imageioView orientation (no spurious transpose)
- CLI positional path + QSettings browse_dir / window geometry persistence
- Single-load view switching (no double-load)
- Background file loading with stale-load cancellation
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import imageio.v3 as iio
import ncempy.io.mrc
import numpy as np
import pytest
from qtpy import QtCore

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from TemDataBrowser import TemView, create_app, imageioView, parse_cli_args


def _wait_until(qtapp, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qtapp.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _cleanup_app(app):
    for view in app.views.values():
        if hasattr(view, "_loader"):
            view._loader.close()
    if hasattr(app, "_stop_file_refresh"):
        app._stop_file_refresh()
    app.tree_view.setModel(None)
    app.fs_model.deleteLater()
    app.ui.close()
    app.ui.deleteLater()
    app.qtapp.processEvents()


def test_parse_cli_args():
    args, rem = parse_cli_args(["/some/path", "--file_filter", "*.dm4"])
    assert args.path == "/some/path"
    assert rem == ["--file_filter", "*.dm4"]

    args2, rem2 = parse_cli_args([])
    assert args2.path is None
    assert rem2 == []


def test_imageio_orientation_not_transposed(tmp_path):
    """Verify imageioView preserves (rows, cols) orientation under row-major mode."""
    # Asymmetric non-square image: height=30, width=60, bright top-left quadrant
    img = np.zeros((30, 60), dtype=np.uint8)
    img[:10, :20] = 200
    png_path = tmp_path / "asymmetric.png"
    iio.imwrite(png_path, img)

    qsettings = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat)
    app = create_app([str(png_path)], qsettings=qsettings)
    try:
        view = app.views[imageioView.name]
        assert _wait_until(
            app.qtapp,
            lambda: view.data is not None and view.imview.image is not None,
        )
        assert view.data.shape == (30, 60)
        assert view.imview.image.shape == (30, 60)
        assert int(view.imview.image[5, 10]) == 200
        assert int(view.imview.image[25, 50]) == 0
    finally:
        _cleanup_app(app)


def test_qsettings_persistence_and_cli_override(tmp_path):
    dir_a = tmp_path / "dir_a"
    dir_b = tmp_path / "dir_b"
    dir_a.mkdir()
    dir_b.mkdir()
    ini_path = str(tmp_path / "prefs.ini")

    # 1) Launch with positional dir_a and save UI state
    qs1 = QtCore.QSettings(ini_path, QtCore.QSettings.Format.IniFormat)
    app1 = create_app([str(dir_a)], qsettings=qs1)
    try:
        assert Path(app1.settings["browse_dir"]) == dir_a
        app1.save_ui_state()
    finally:
        _cleanup_app(app1)

    # 2) Relaunch with no args -> should restore dir_a
    qs2 = QtCore.QSettings(ini_path, QtCore.QSettings.Format.IniFormat)
    app2 = create_app([], qsettings=qs2)
    try:
        assert Path(app2.settings["browse_dir"]) == dir_a
    finally:
        _cleanup_app(app2)

    # 3) Relaunch with explicit positional dir_b -> should override saved dir_a
    qs3 = QtCore.QSettings(ini_path, QtCore.QSettings.Format.IniFormat)
    app3 = create_app([str(dir_b)], qsettings=qs3)
    try:
        assert Path(app3.settings["browse_dir"]) == dir_b
    finally:
        _cleanup_app(app3)


def test_no_double_load_on_view_switch(tmp_path):
    """Switching from file_info -> TEM data viewer should call on_change_data_filename once."""
    mrc_path = tmp_path / "single_load.mrc"
    arr = np.arange(16 * 24, dtype=np.float32).reshape(1, 16, 24)
    ncempy.io.mrc.mrcWriter(str(mrc_path), arr, pixelSize=(0.2, 0.2, 0.2))

    qsettings = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat)
    app = create_app([str(tmp_path)], qsettings=qsettings)
    try:
        tem_view = app.views[TemView.name]
        call_count = 0
        orig = tem_view.on_change_data_filename

        def counted(fname):
            nonlocal call_count
            call_count += 1
            return orig(fname)

        tem_view.on_change_data_filename = counted
        app.settings["data_filename"] = str(mrc_path)

        assert _wait_until(app.qtapp, lambda: tem_view.data is not None)
        assert call_count == 1
        assert tem_view.data.shape == (16, 24)
    finally:
        _cleanup_app(app)


def test_async_loader_cancels_stale_load(tmp_path):
    """If a second file is selected while the first is still loading, only the second is shown."""
    slow_path = tmp_path / "slow.png"
    fast_path = tmp_path / "fast.png"
    iio.imwrite(slow_path, np.full((12, 12), 50, dtype=np.uint8))
    iio.imwrite(fast_path, np.full((20, 24), 220, dtype=np.uint8))

    gate = threading.Event()
    slow_started = threading.Event()

    qsettings = QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat)
    app = create_app([str(tmp_path)], qsettings=qsettings)
    try:
        view = app.views[imageioView.name]
        app.current_view = view
        app.setup_view(view)

        def fake_reader(fname):
            if Path(fname).name == "slow.png":
                slow_started.set()
                gate.wait(timeout=3.0)
            return iio.imread(fname)

        view._loader.load(str(slow_path), fake_reader)
        assert slow_started.wait(timeout=2.0)

        # Now request fast.png before slow.png finishes
        view._loader.load(str(fast_path), fake_reader)
        assert _wait_until(
            app.qtapp,
            lambda: view.data is not None and view.data.shape == (20, 24),
        )

        # Release slow.png and verify it does NOT overwrite fast.png
        gate.set()
        time.sleep(0.15)
        app.qtapp.processEvents()
        assert view.data.shape == (20, 24)
        assert int(view.data[0, 0]) == 220
    finally:
        gate.set()
        _cleanup_app(app)
