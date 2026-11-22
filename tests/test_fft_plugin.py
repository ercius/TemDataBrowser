"""Headless Qt + numerical tests for TemDataBrowser's Live FFT plugin."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

import ncempy.io.mrc
import numpy as np
import pytest
from qtpy import QtCore, QtWidgets

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ScopeFoundry.data_browser import DataBrowser
from TemDataBrowser import TemView, imageioView
from TemDataBrowser.fft_plugin import (
    FFTPlugIn,
    compute_fft,
    compute_reciprocal_calibration,
    format_d_spacing,
)


def _pump(qtapp, sec=0.1):
    end = time.monotonic() + sec
    while time.monotonic() < end:
        qtapp.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.01)


def _cleanup(app):
    if hasattr(app, "_stop_file_refresh"):
        app._stop_file_refresh()
    app.tree_view.setModel(None)
    app.fs_model.deleteLater()
    app.ui.close()
    app.ui.deleteLater()
    app.qtapp.processEvents()


def test_reciprocal_calibration_recovers_lattice_d_spacing():
    """Synthetic Au (111)-like cosine lattice (d = 0.235 nm, dx = 0.02 nm) recovers d within 1%."""
    n = 256
    dx_nm = 0.02
    d_true_nm = 0.235
    k_true = 1.0 / d_true_nm  # ~4.255 1/nm

    y, x = np.mgrid[0:n, 0:n]
    img = np.cos(2.0 * np.pi * k_true * (x * dx_nm)).astype(np.float32)

    fft_img, (vmin, vmax) = compute_fft(img, apply_window=True, scale="linear", mask_dc_radius=3)
    assert vmax > vmin

    dkx, dky, unit = compute_reciprocal_calibration(fft_img.shape, dx_nm, dx_nm)
    assert unit == "1/nm"

    # Find dominant peak in positive-kx half-plane excluding DC
    cy, cx = n // 2, n // 2
    half = fft_img.copy()
    half[:, : cx + 3] = 0.0
    py, px = np.unravel_index(np.argmax(half), half.shape)

    kx = (px - cx) * dkx
    ky = (py - cy) * dky
    d_measured = 1.0 / np.hypot(kx, ky)
    assert abs(d_measured - d_true_nm) / d_true_nm < 0.02

    readout = format_d_spacing(kx, ky, unit)
    assert "1/nm" in readout
    assert "Å" in readout


def test_compute_fft_scales_and_rgb():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    for scale in ("log", "sqrt", "linear"):
        out, (vmin, vmax) = compute_fft(rgb, apply_window=True, scale=scale, mask_dc_radius=2)
        assert out.shape == (64, 64)
        assert np.all(np.isfinite(out))
        assert vmax > vmin


def test_dock_fixed_right_and_syncs_button(tmp_path):
    with patch.object(sys, "argv", ["test"]):
        app = DataBrowser(["test"])
    try:
        app.add_view(TemView(app))
        plugin = FFTPlugIn(app)
        app.add_plugin(plugin)

        plugin.show_button.setChecked(True)
        _pump(app.qtapp)

        assert plugin.dock is not None
        assert plugin.dock.isVisible()
        # Dock must be closable only (not movable or floatable)
        features = plugin.dock.features()
        assert features == QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        assert (
            plugin.dock.allowedAreas()
            == QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )

        # Closing the dock unchecks the toolbar button
        plugin.dock.close()
        _pump(app.qtapp)
        assert not plugin.show_button.isChecked()
    finally:
        _cleanup(app)


def test_fft_updates_on_3d_mrc_slice_change_and_roi(tmp_path):
    """Verify live FFT tracks 3D slice scrubbing and ROI box moves."""
    n = 128
    y, x = np.mgrid[0:n, 0:n]

    # Slice 0: horizontal fringes along X (peaks on kx axis)
    # Slice 1: split image — left half has X fringes, right half has Y fringes
    slice0 = np.cos(2.0 * np.pi * x / 8.0).astype(np.float32)
    slice1 = np.zeros((n, n), dtype=np.float32)
    slice1[:, : n // 2] = np.cos(2.0 * np.pi * x[:, : n // 2] / 8.0)
    slice1[:, n // 2 :] = np.cos(2.0 * np.pi * y[:, n // 2 :] / 8.0)

    stack = np.stack([slice0, slice1], axis=0)
    mrc_path = tmp_path / "lattice_stack.mrc"
    ncempy.io.mrc.mrcWriter(str(mrc_path), stack, pixelSize=(0.1, 0.1, 0.1))

    with patch.object(sys, "argv", ["test"]):
        app = DataBrowser(["test"])
    try:
        tem_view = TemView(app)
        app.add_view(imageioView(app))
        app.add_view(tem_view)
        plugin = FFTPlugIn(app)
        app.add_plugin(plugin)

        app.settings["data_filename"] = str(mrc_path)
        _pump(app.qtapp, 0.15)

        plugin.show_button.setChecked(True)
        _pump(app.qtapp, 0.15)
        assert plugin.fft_data is not None
        assert plugin._recip_unit == "1/nm"

        def _dominant_axis(fft_arr):
            ny, nx = fft_arr.shape
            cy, cx = ny // 2, nx // 2
            tmp = fft_arr.copy()
            tmp[cy - 3 : cy + 4, cx - 3 : cx + 4] = 0.0
            py, px = np.unravel_index(np.argmax(tmp), tmp.shape)
            return "kx" if abs(px - cx) > abs(py - cy) else "ky"

        # Slice 0 has fringes varying along X -> FFT peaks along kx
        assert _dominant_axis(plugin.fft_data) == "kx"

        # Switch to slice 1 and enable ROI
        tem_view.imview.setCurrentIndex(1)
        _pump(app.qtapp, 0.15)

        plugin.roi_checkbox.setChecked(True)
        _pump(app.qtapp, 0.15)
        assert plugin._roi is not None

        img_item = tem_view.imview.getImageItem()
        bounds = img_item.mapRectToParent(img_item.boundingRect()).normalized()

        # Move ROI to left half (X fringes -> kx peaks)
        plugin._roi.setPos([bounds.left() + bounds.width() * 0.05, bounds.top() + bounds.height() * 0.1])
        plugin._roi.setSize([bounds.width() * 0.4, bounds.height() * 0.8])
        plugin.recompute_fft()
        assert _dominant_axis(plugin.fft_data) == "kx"

        # Move ROI to right half (Y fringes -> ky peaks, rotated 90 deg)
        plugin._roi.setPos([bounds.left() + bounds.width() * 0.55, bounds.top() + bounds.height() * 0.1])
        plugin._roi.setSize([bounds.width() * 0.4, bounds.height() * 0.8])
        plugin.recompute_fft()
        assert _dominant_axis(plugin.fft_data) == "ky"

        # Unchecking ROI removes it cleanly
        plugin.roi_checkbox.setChecked(False)
        assert plugin._roi is None
    finally:
        _cleanup(app)
