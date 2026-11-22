"""Live FFT DataBrowserPlugIn for TemDataBrowser.

Provides a fixed right-side QDockWidget displaying the calibrated 2D FFT of
the active image view (TemView, imageioView, etc.) with optional draggable ROI,
Hann windowing, DC peak masking, Log/Sqrt/Linear intensity scaling, and
hover/click d-spacing measurement.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets
from scipy.fft import fft2, fftshift

from ScopeFoundry.data_browser.data_browser_plug_in import DataBrowserPlugIn


def compute_fft(
    image_2d: np.ndarray,
    apply_window: bool = True,
    scale: str = "log",
    mask_dc_radius: int = 2,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Compute a centered 2D FFT magnitude image and robust display contrast levels.

    Parameters
    ----------
    image_2d : np.ndarray
        2D grayscale array (ny, nx) or 2D RGB/RGBA array (ny, nx, 3|4).
    apply_window : bool
        If True, multiply by a separable 2D Hann window to suppress edge streaks.
    scale : {'log', 'sqrt', 'linear'}
        Intensity mapping applied to the FFT magnitude.
    mask_dc_radius : int
        Pixel radius around the DC center excluded when computing auto-contrast
        levels so the central DC peak does not crush Bragg spots.

    Returns
    -------
    display_fft : np.ndarray
        2D float32 centered FFT image of shape (ny, nx).
    levels : tuple[float, float]
        Suggested (vmin, vmax) display levels.
    """
    arr = np.asarray(image_2d)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        # Convert RGB/RGBA to luminance
        arr = (
            0.2126 * arr[..., 0].astype(np.float32)
            + 0.7152 * arr[..., 1].astype(np.float32)
            + 0.0722 * arr[..., 2].astype(np.float32)
        )
    elif arr.ndim != 2:
        raise ValueError(f"compute_fft expects a 2D image, got shape {arr.shape}")
    else:
        arr = arr.astype(np.float32, copy=False)

    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"Image too small for 2D FFT: {arr.shape}")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = arr - float(np.mean(arr))

    if apply_window:
        ny, nx = arr.shape
        wy = np.hanning(ny).astype(np.float32)
        wx = np.hanning(nx).astype(np.float32)
        arr = arr * (wy[:, None] * wx[None, :])

    mag = np.abs(fftshift(fft2(arr))).astype(np.float32)

    scale_key = (scale or "log").lower()
    if scale_key == "log":
        display = np.log1p(mag)
    elif scale_key == "sqrt":
        display = np.sqrt(mag)
    elif scale_key == "linear":
        display = mag
    else:
        raise ValueError(f"Unsupported FFT scale '{scale}'. Use 'log', 'sqrt', or 'linear'.")

    ny, nx = display.shape
    cy, cx = ny // 2, nx // 2
    if mask_dc_radius > 0 and min(ny, nx) > 2 * mask_dc_radius + 2:
        yy, xx = np.ogrid[:ny, :nx]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 > mask_dc_radius ** 2
        sample = display[mask]
    else:
        sample = display.ravel()

    if sample.size > 0:
        vmin = float(np.percentile(sample, 1.0))
        vmax = float(np.percentile(sample, 99.8))
        if vmax <= vmin:
            vmin = float(np.min(sample))
            vmax = float(np.max(sample))
            if vmax <= vmin:
                vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    return display, (vmin, vmax)


def extract_pixel_calibration(view) -> Tuple[Optional[float], Optional[float]]:
    """Extract (dx_nm, dy_nm) pixel pitch in nm from a DataBrowser image view.

    Returns (None, None) when the view has no physical spatial calibration.
    """
    if view is None or not hasattr(view, "imview") or view.imview is None:
        return None, None

    plt = getattr(view, "plt", None)
    if plt is None:
        return None, None

    axis_unit = (plt.getAxis("bottom").labelUnits or "").strip()
    if not axis_unit or axis_unit == "pixels":
        return None, None

    img_item = view.imview.getImageItem()
    if img_item is None:
        return None, None

    tr = img_item.transform()
    sx = abs(float(tr.m11()))
    sy = abs(float(tr.m22()))
    if sx <= 0.0 or sy <= 0.0:
        return None, None

    if axis_unit == "m":
        return sx * 1e9, sy * 1e9
    if axis_unit == "nm":
        return sx, sy
    if axis_unit in ("um", "µm", "μm"):
        return sx * 1e3, sy * 1e3
    if axis_unit in ("A", "Ang", "Å"):
        return sx * 0.1, sy * 0.1
    if axis_unit == "pm":
        return sx * 1e-3, sy * 1e-3
    return None, None


def compute_reciprocal_calibration(
    shape: Tuple[int, int],
    dx_nm: Optional[float],
    dy_nm: Optional[float],
) -> Tuple[float, float, str]:
    """Return (dkx, dky, unit) for an FFT of shape (ny, nx)."""
    ny, nx = int(shape[0]), int(shape[1])
    if dx_nm is not None and dy_nm is not None and dx_nm > 0 and dy_nm > 0:
        return 1.0 / (nx * dx_nm), 1.0 / (ny * dy_nm), "1/nm"
    return 1.0 / nx, 1.0 / ny, "cycles/px"


def format_d_spacing(kx: float, ky: float, unit: str) -> str:
    """Format reciprocal coordinates into a human-readable d-spacing readout."""
    k = math.hypot(kx, ky)
    angle = math.degrees(math.atan2(ky, kx))
    if k <= 1e-12:
        return f"|k| = 0 {unit} | d = ∞ | θ = 0.0°"

    if unit == "1/nm":
        d_nm = 1.0 / k
        d_ang = d_nm * 10.0
        return f"|k| = {k:.3f} 1/nm | d = {d_nm:.4f} nm ({d_ang:.3f} Å) | θ = {angle:.1f}°"

    wavelength_px = 1.0 / k
    return f"|k| = {k:.4f} {unit} | λ = {wavelength_px:.2f} px | θ = {angle:.1f}°"


class FFTPlugIn(DataBrowserPlugIn):
    """Side-by-side Live FFT plugin docked on the right side of DataBrowser."""

    name = "live_fft"
    button_text = "FFT"
    show_keyboard_key = QtCore.Qt.Key_F
    description = "Live FFT side panel with ROI and d-spacing readout (Ctrl + F)"

    def __init__(self, databrowser) -> None:
        super().__init__(databrowser)
        self.dock: Optional[QtWidgets.QDockWidget] = None
        self.fft_plt: Optional[pg.PlotItem] = None
        self.fft_imview: Optional[pg.ImageView] = None
        self.roi_checkbox: Optional[QtWidgets.QCheckBox] = None
        self.hann_checkbox: Optional[QtWidgets.QCheckBox] = None
        self.dc_mask_checkbox: Optional[QtWidgets.QCheckBox] = None
        self.scale_combo: Optional[QtWidgets.QComboBox] = None
        self.readout_label: Optional[QtWidgets.QLabel] = None
        self.clear_pin_btn: Optional[QtWidgets.QPushButton] = None
        self._pin_marker: Optional[pg.ScatterPlotItem] = None

        self._connected_image_items = set()
        self._roi: Optional[pg.RectROI] = None
        self._roi_view = None
        self._pinned_coord: Optional[Tuple[float, float]] = None

        self.fft_data: Optional[np.ndarray] = None
        self._dkx: float = 1.0
        self._dky: float = 1.0
        self._recip_unit: str = "cycles/px"

        self._debounce_timer = QtCore.QTimer(self.q_object)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(30)
        self._debounce_timer.timeout.connect(self.recompute_fft)

        self.databrowser.settings.view_name.add_listener(self._on_view_changed)

    def get_show_hide_button(self) -> QtWidgets.QPushButton:
        """Return the toggle button with native Qt styling matching sibling toolbar buttons."""
        self.show_button.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.MinimumExpanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        return self.show_button

    def setup(self) -> None:
        """Build the fixed right-side QDockWidget containing controls and FFT ImageView."""
        # Leave self.ui = None so DataBrowserPlugIn.show() does not stack us vertically
        # inside data_view_layout; we manage self.dock directly.
        self.ui = None

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        controls = QtWidgets.QHBoxLayout()
        self.roi_checkbox = QtWidgets.QCheckBox("ROI")
        self.roi_checkbox.setToolTip("Toggle draggable ROI box on the real-space image")
        self.roi_checkbox.toggled.connect(self._on_roi_toggled)

        self.hann_checkbox = QtWidgets.QCheckBox("Hann")
        self.hann_checkbox.setChecked(True)
        self.hann_checkbox.setToolTip("Apply 2D Hann window to reduce edge streaks")
        self.hann_checkbox.toggled.connect(self.recompute_fft)

        self.dc_mask_checkbox = QtWidgets.QCheckBox("Mask DC")
        self.dc_mask_checkbox.setChecked(True)
        self.dc_mask_checkbox.setToolTip("Ignore central DC peak when auto-scaling contrast")
        self.dc_mask_checkbox.toggled.connect(self.recompute_fft)

        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["Log", "Sqrt", "Linear"])
        self.scale_combo.currentTextChanged.connect(self.recompute_fft)

        controls.addWidget(self.roi_checkbox)
        controls.addWidget(self.hann_checkbox)
        controls.addWidget(self.dc_mask_checkbox)
        controls.addWidget(QtWidgets.QLabel("Scale:"))
        controls.addWidget(self.scale_combo)
        controls.addStretch()
        layout.addLayout(controls)

        self.fft_plt = pg.PlotItem(
            labels={"bottom": ("kx", "cycles/px"), "left": ("ky", "cycles/px")}
        )
        self.fft_plt.setAspectLocked(True)
        self.fft_imview = pg.ImageView(view=self.fft_plt)
        self.fft_imview.ui.roiBtn.hide()
        self.fft_imview.ui.menuBtn.hide()
        layout.addWidget(self.fft_imview, stretch=1)

        self._pin_marker = pg.ScatterPlotItem(
            size=12,
            symbol="+",
            pen=pg.mkPen("y", width=2),
            brush=None,
        )
        self.fft_plt.addItem(self._pin_marker)

        readout_bar = QtWidgets.QHBoxLayout()
        self.readout_label = QtWidgets.QLabel("Hover or click FFT to measure d-spacing")
        self.readout_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.clear_pin_btn = QtWidgets.QPushButton("Clear")
        self.clear_pin_btn.setMaximumWidth(55)
        self.clear_pin_btn.setVisible(False)
        self.clear_pin_btn.clicked.connect(self.clear_pin)
        readout_bar.addWidget(self.readout_label, stretch=1)
        readout_bar.addWidget(self.clear_pin_btn)
        layout.addLayout(readout_bar)

        self.fft_plt.scene().sigMouseMoved.connect(self._on_fft_mouse_moved)
        self.fft_plt.scene().sigMouseClicked.connect(self._on_fft_mouse_clicked)

        self.dock = QtWidgets.QDockWidget("Live FFT", self.databrowser.ui)
        self.dock.setObjectName("LiveFFTDock")
        self.dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.dock.setMinimumWidth(360)
        self.dock.setWidget(container)
        self.dock.visibilityChanged.connect(self._on_dock_visibility_changed)
        self.databrowser.ui.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock
        )

    def show(self) -> None:
        if not self.is_loaded:
            self.setup()
            self.is_loaded = True
        if self.dock is not None:
            self.dock.show()
        self._ensure_active_view_connected()
        if self.roi_checkbox is not None and self.roi_checkbox.isChecked():
            self._attach_roi_to_active_view()
        self.recompute_fft()

    def hide(self) -> None:
        self._debounce_timer.stop()
        self._detach_roi()
        if self.dock is not None:
            self.dock.hide()

    def update(self, fname: Optional[str] = None) -> None:
        if not self.is_showing:
            return
        self._ensure_active_view_connected()
        if self.roi_checkbox is not None and self.roi_checkbox.isChecked():
            self._attach_roi_to_active_view()
        self.recompute_fft()

    def _on_dock_visibility_changed(self, visible: bool) -> None:
        if not visible:
            self._detach_roi()
            if self.show_button.isChecked():
                self.show_button.blockSignals(True)
                try:
                    self.show_button.setChecked(False)
                finally:
                    self.show_button.blockSignals(False)

    def _on_view_changed(self) -> None:
        if not self.is_showing:
            return
        self._detach_roi()
        self._ensure_active_view_connected()
        if self.roi_checkbox is not None and self.roi_checkbox.isChecked():
            self._attach_roi_to_active_view()
        self.recompute_fft()

    def _get_active_image_view(self):
        view = getattr(self.databrowser, "current_view", None)
        if view is None or not hasattr(view, "imview") or view.imview is None:
            return None
        return view

    def _ensure_active_view_connected(self) -> None:
        view = self._get_active_image_view()
        if view is None:
            return
        img_item = view.imview.getImageItem()
        if img_item is None:
            return
        item_id = id(img_item)
        if item_id not in self._connected_image_items:
            img_item.sigImageChanged.connect(self._on_source_image_changed)
            self._connected_image_items.add(item_id)

    def _on_source_image_changed(self) -> None:
        if not self.is_showing:
            return
        if self.roi_checkbox is not None and self.roi_checkbox.isChecked():
            self._ensure_roi_bounds()
        self._schedule_recompute()

    def _schedule_recompute(self) -> None:
        if self.is_showing:
            self._debounce_timer.start()

    def _on_roi_toggled(self, enabled: bool) -> None:
        if enabled:
            self._attach_roi_to_active_view()
        else:
            self._detach_roi()
        self.recompute_fft()

    def _get_viewbox_for_view(self, view):
        if hasattr(view, "plt") and view.plt is not None:
            return view.plt.getViewBox()
        if hasattr(view, "imview") and view.imview is not None:
            return view.imview.getView()
        return None

    def _attach_roi_to_active_view(self) -> None:
        view = self._get_active_image_view()
        if view is None:
            self._detach_roi()
            return

        if self._roi is not None and self._roi_view is view:
            self._ensure_roi_bounds()
            return

        self._detach_roi()
        img_item = view.imview.getImageItem()
        vb = self._get_viewbox_for_view(view)
        if img_item is None or vb is None or img_item.image is None:
            return

        parent_rect = img_item.mapRectToParent(img_item.boundingRect()).normalized()
        if parent_rect.width() <= 0 or parent_rect.height() <= 0:
            return

        w = parent_rect.width() * 0.5
        h = parent_rect.height() * 0.5
        x0 = parent_rect.left() + (parent_rect.width() - w) * 0.5
        y0 = parent_rect.top() + (parent_rect.height() - h) * 0.5

        roi = pg.RectROI(
            [x0, y0],
            [w, h],
            maxBounds=parent_rect,
            pen=pg.mkPen("y", width=2),
        )
        roi.sigRegionChanged.connect(self._schedule_recompute)
        vb.addItem(roi)
        self._roi = roi
        self._roi_view = view

    def _ensure_roi_bounds(self) -> None:
        if self._roi is None or self._roi_view is None:
            return
        img_item = self._roi_view.imview.getImageItem()
        if img_item is None or img_item.image is None:
            return
        parent_rect = img_item.mapRectToParent(img_item.boundingRect()).normalized()
        if parent_rect.width() <= 0 or parent_rect.height() <= 0:
            return
        self._roi.maxBounds = parent_rect
        # Reset ROI inside new image bounds if the previous image had different dimensions/scale
        if not parent_rect.intersects(self._roi.parentBounds().normalized()):
            w = parent_rect.width() * 0.5
            h = parent_rect.height() * 0.5
            self._roi.setPos([parent_rect.left() + w * 0.5, parent_rect.top() + h * 0.5])
            self._roi.setSize([w, h])

    def _detach_roi(self) -> None:
        if self._roi is not None and self._roi_view is not None:
            vb = self._get_viewbox_for_view(self._roi_view)
            if vb is not None:
                try:
                    vb.removeItem(self._roi)
                except Exception:
                    pass
        self._roi = None
        self._roi_view = None

    def _extract_source_frame(self, view) -> Optional[np.ndarray]:
        img_item = view.imview.getImageItem()
        if img_item is None or img_item.image is None:
            return None
        frame = np.asarray(img_item.image)
        if frame.ndim < 2:
            return None

        if (
            self._roi is not None
            and self._roi_view is view
            and self.roi_checkbox is not None
            and self.roi_checkbox.isChecked()
        ):
            local_rect = img_item.mapRectFromParent(self._roi.parentBounds()).normalized()
            ny, nx = frame.shape[:2]
            x0 = max(0, min(nx, int(round(local_rect.left()))))
            x1 = max(0, min(nx, int(round(local_rect.right()))))
            y0 = max(0, min(ny, int(round(local_rect.top()))))
            y1 = max(0, min(ny, int(round(local_rect.bottom()))))
            if (x1 - x0) >= 8 and (y1 - y0) >= 8:
                frame = frame[y0:y1, x0:x1]

        return frame

    def recompute_fft(self) -> None:
        """Compute and display the 2D FFT for the active image view (or its ROI)."""
        if not self.is_loaded or self.fft_imview is None:
            return

        view = self._get_active_image_view()
        if view is None:
            self.fft_data = None
            self.fft_imview.setImage(np.zeros((10, 10), dtype=np.float32))
            if self.readout_label is not None:
                self.readout_label.setText("No image view active")
            return

        frame = self._extract_source_frame(view)
        if frame is None or min(frame.shape[:2]) < 2:
            self.fft_data = None
            return

        apply_window = bool(self.hann_checkbox.isChecked()) if self.hann_checkbox else True
        mask_radius = 2 if (self.dc_mask_checkbox and self.dc_mask_checkbox.isChecked()) else 0
        scale = self.scale_combo.currentText().lower() if self.scale_combo else "log"

        try:
            display_fft, (vmin, vmax) = compute_fft(
                frame,
                apply_window=apply_window,
                scale=scale,
                mask_dc_radius=mask_radius,
            )
        except Exception as exc:
            if self.readout_label is not None:
                self.readout_label.setText(f"FFT error: {exc}")
            return

        self.fft_data = display_fft
        ny, nx = display_fft.shape
        dx_nm, dy_nm = extract_pixel_calibration(view)
        dkx, dky, unit = compute_reciprocal_calibration((ny, nx), dx_nm, dy_nm)
        self._dkx = dkx
        self._dky = dky
        self._recip_unit = unit

        self.fft_imview.setImage(display_fft, autoLevels=False, levels=(vmin, vmax))
        fft_item = self.fft_imview.getImageItem()
        tr = QtGui.QTransform()
        tr.scale(dkx, dky)
        tr.translate(-(nx // 2) - 0.5, -(ny // 2) - 0.5)
        fft_item.setTransform(tr)

        self.fft_plt.getAxis("bottom").setLabel("kx", units=unit)
        self.fft_plt.getAxis("left").setLabel("ky", units=unit)
        self.fft_plt.autoRange()

        if self._pinned_coord is not None:
            kx, ky = self._pinned_coord
            self._update_pin_readout(kx, ky)

    def clear_pin(self) -> None:
        self._pinned_coord = None
        if self._pin_marker is not None:
            self._pin_marker.setData([], [])
        if self.clear_pin_btn is not None:
            self.clear_pin_btn.setVisible(False)
        if self.readout_label is not None:
            self.readout_label.setText("Hover or click FFT to measure d-spacing")

    def _update_pin_readout(self, kx: float, ky: float) -> None:
        if self.readout_label is not None:
            self.readout_label.setText(
                f"Pinned: {format_d_spacing(kx, ky, self._recip_unit)}"
            )

    def _on_fft_mouse_moved(self, scene_pos) -> None:
        if self.fft_data is None or self.fft_plt is None or self._pinned_coord is not None:
            return
        vb = self.fft_plt.getViewBox()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return
        pt = vb.mapSceneToView(scene_pos)
        if self.readout_label is not None:
            self.readout_label.setText(
                format_d_spacing(pt.x(), pt.y(), self._recip_unit)
            )

    def _on_fft_mouse_clicked(self, event) -> None:
        if self.fft_data is None or self.fft_plt is None:
            return
        vb = self.fft_plt.getViewBox()
        scene_pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(scene_pos):
            return
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.clear_pin()
            return
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pt = vb.mapSceneToView(scene_pos)
            kx, ky = float(pt.x()), float(pt.y())
            self._pinned_coord = (kx, ky)
            if self._pin_marker is not None:
                self._pin_marker.setData([kx], [ky])
            if self.clear_pin_btn is not None:
                self.clear_pin_btn.setVisible(True)
            self._update_pin_readout(kx, ky)
