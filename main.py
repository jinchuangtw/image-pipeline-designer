import sys
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QGroupBox,
    QTextEdit,
    QMessageBox,
    QComboBox,
    QSlider,
    QCheckBox,
    QFrame,
    QSizePolicy,
    QStackedWidget,
)


# ----------------------------
# Utilities: image <-> QImage
# ----------------------------


def cv_to_qimage(img: np.ndarray) -> QImage:
    if img is None:
        return QImage()

    if img.ndim == 2:
        h, w = img.shape
        img8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
        return QImage(img8.data, w, h, w, QImage.Format_Grayscale8).copy()

    if img.ndim == 3 and img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    img8 = np.clip(img, 0, 255).astype(np.uint8)
    if img8.ndim == 2:
        h, w = img8.shape
        return QImage(img8.data, w, h, w, QImage.Format_Grayscale8).copy()
    if img8.ndim == 3 and img8.shape[2] == 3:
        rgb = cv2.cvtColor(img8, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
    return QImage()


def fit_pixmap_to_label(pixmap: QPixmap, label: QLabel) -> QPixmap:
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)


def ensure_gray(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def read_image_any_path(path: str) -> Optional[np.ndarray]:
    """Robust read for unicode paths on Windows/Linux."""
    if not path:
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def resize_to_match(src_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Resize src to exactly match target shape (H, W)."""
    th, tw = target_bgr.shape[:2]
    return cv2.resize(src_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)


# ----------------------------
# Parameter widgets helpers
# ----------------------------


class OddSlider(QWidget):
    """Slider that only outputs odd values from a predefined list."""

    def __init__(
        self, title: str, odd_values: List[int], default_value: int, parent=None
    ):
        super().__init__(parent)
        self.odd_values = odd_values
        if default_value not in odd_values:
            default_value = odd_values[len(odd_values) // 2]

        self.title_label = QLabel(title)
        self.value_label = QLabel(str(default_value))
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, len(odd_values) - 1)
        self.slider.setValue(odd_values.index(default_value))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(1)

        layout = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(self.title_label)
        top.addWidget(self.value_label)
        layout.addLayout(top)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, idx: int):
        self.value_label.setText(str(self.odd_values[idx]))

    def value(self) -> int:
        return self.odd_values[self.slider.value()]


class IntSlider(QWidget):
    def __init__(self, title: str, min_v: int, max_v: int, default: int, parent=None):
        super().__init__(parent)
        self.title_label = QLabel(title)
        self.value_label = QLabel(str(default))
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_v, max_v)
        self.slider.setValue(default)
        self.slider.setSingleStep(1)

        layout = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(self.title_label)
        top.addWidget(self.value_label)
        layout.addLayout(top)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, v: int):
        self.value_label.setText(str(v))

    def value(self) -> int:
        return int(self.slider.value())


class FloatSlider(QWidget):
    """float slider via integer scale."""

    def __init__(
        self,
        title: str,
        min_f: float,
        max_f: float,
        default: float,
        scale: int = 10,
        parent=None,
    ):
        super().__init__(parent)
        self.scale = scale
        self.title_label = QLabel(title)
        self.value_label = QLabel(f"{default:.2f}")
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_f * scale), int(max_f * scale))
        self.slider.setValue(int(default * scale))
        self.slider.setSingleStep(1)

        layout = QVBoxLayout()
        top = QHBoxLayout()
        top.addWidget(self.title_label)
        top.addWidget(self.value_label)
        layout.addLayout(top)
        layout.addWidget(self.slider)
        self.setLayout(layout)

        self.slider.valueChanged.connect(self._on_change)

    def _on_change(self, v: int):
        self.value_label.setText(f"{v / self.scale:.2f}")

    def value(self) -> float:
        return float(self.slider.value()) / self.scale


# ----------------------------
# Processing operations
# ----------------------------


@dataclass
class OperationResult:
    image: np.ndarray
    description: str


class Processor:
    @staticmethod
    def avg_blur(img: np.ndarray, k: int) -> OperationResult:
        out = cv2.blur(img, (k, k))
        return OperationResult(out, f"Average Blur (ksize={k})")

    @staticmethod
    def gaussian_blur(img: np.ndarray, k: int, sigma: float) -> OperationResult:
        out = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
        return OperationResult(out, f"Gaussian Blur (ksize={k}, sigma={sigma:.2f})")

    @staticmethod
    def median_blur(img: np.ndarray, k: int) -> OperationResult:
        out = cv2.medianBlur(img, k)
        return OperationResult(out, f"Median Blur (ksize={k})")

    @staticmethod
    def threshold(img: np.ndarray, thresh: int, invert: bool) -> OperationResult:
        gray = ensure_gray(img)
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, out = cv2.threshold(gray, thresh, 255, flag)
        return OperationResult(
            ensure_bgr(out), f"Threshold (t={thresh}, invert={invert})"
        )

    @staticmethod
    def adaptive_threshold(
        img: np.ndarray, block_size: int, C: int, method: str
    ) -> OperationResult:
        gray = ensure_gray(img)
        m = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if method == "Mean"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        out = cv2.adaptiveThreshold(gray, 255, m, cv2.THRESH_BINARY, block_size, C)
        return OperationResult(
            ensure_bgr(out),
            f"Adaptive Threshold (method={method}, block={block_size}, C={C})",
        )

    @staticmethod
    def otsu_threshold(img: np.ndarray, invert: bool) -> OperationResult:
        gray = ensure_gray(img)
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, out = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
        return OperationResult(ensure_bgr(out), f"Otsu Threshold (invert={invert})")

    @staticmethod
    def sobel(img: np.ndarray, ksize: int, mode: str) -> OperationResult:
        gray = ensure_gray(img).astype(np.float32)
        if mode == "X":
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
            mag = np.abs(sx)
        elif mode == "Y":
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
            mag = np.abs(sy)
        else:
            sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
            sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
            mag = cv2.magnitude(sx, sy)
        out = np.clip(mag, 0, 255).astype(np.uint8)
        return OperationResult(ensure_bgr(out), f"Sobel (mode={mode}, ksize={ksize})")

    @staticmethod
    def canny(
        img: np.ndarray, t1: int, t2: int, aperture: int, l2: bool
    ) -> OperationResult:
        gray = ensure_gray(img)
        out = cv2.Canny(
            gray, threshold1=t1, threshold2=t2, apertureSize=aperture, L2gradient=l2
        )
        return OperationResult(
            ensure_bgr(out), f"Canny (t1={t1}, t2={t2}, aperture={aperture}, L2={l2})"
        )

    @staticmethod
    def morphology(
        img: np.ndarray, op: str, shape: str, k: int, iters: int
    ) -> OperationResult:
        gray = ensure_gray(img)
        st = (
            cv2.MORPH_RECT
            if shape == "Rect"
            else (cv2.MORPH_ELLIPSE if shape == "Ellipse" else cv2.MORPH_CROSS)
        )
        kernel = cv2.getStructuringElement(st, (k, k))

        if op == "Erode":
            out = cv2.erode(gray, kernel, iterations=iters)
        elif op == "Dilate":
            out = cv2.dilate(gray, kernel, iterations=iters)
        else:
            mop_map = {
                "Open": cv2.MORPH_OPEN,
                "Close": cv2.MORPH_CLOSE,
                "Gradient": cv2.MORPH_GRADIENT,
                "TopHat": cv2.MORPH_TOPHAT,
                "BlackHat": cv2.MORPH_BLACKHAT,
            }
            out = cv2.morphologyEx(gray, mop_map[op], kernel, iterations=iters)

        return OperationResult(
            ensure_bgr(out),
            f"Morphology (op={op}, shape={shape}, ksize={k}, iters={iters})",
        )

    @staticmethod
    def absdiff(
        img: np.ndarray, ref_img: np.ndarray, use_abs: bool, normalize: bool
    ) -> OperationResult:
        base = img
        ref = ref_img

        if use_abs:
            out = cv2.absdiff(base, ref)
        else:
            out = cv2.subtract(base, ref)  # saturating subtract

        if normalize:
            # Normalize per channel to 0..255 for better visibility
            out_f = out.astype(np.float32)
            out_n = np.zeros_like(out_f)
            for c in range(3):
                ch = out_f[:, :, c]
                mn, mx = float(np.min(ch)), float(np.max(ch))
                if mx - mn < 1e-6:
                    out_n[:, :, c] = 0.0
                else:
                    out_n[:, :, c] = (ch - mn) * (255.0 / (mx - mn))
            out = np.clip(out_n, 0, 255).astype(np.uint8)

        return OperationResult(
            out, f"Difference (abs={use_abs}, normalize={normalize})"
        )

    @staticmethod
    def clahe(img: np.ndarray, clip_limit: float, tile: int) -> OperationResult:
        gray = ensure_gray(img)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        out = clahe.apply(gray)
        return OperationResult(
            ensure_bgr(out), f"CLAHE (clip={clip_limit:.2f}, tile={tile})"
        )

    @staticmethod
    def unsharp_mask(
        img: np.ndarray, k: int, sigma: float, amount: float
    ) -> OperationResult:
        # amount: 0..3
        blur = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
        # sharpen = img*(1+amount) - blur*amount
        out = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return OperationResult(
            out, f"Unsharp Mask (k={k}, sigma={sigma:.2f}, amount={amount:.2f})"
        )

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma: float) -> OperationResult:
        # gamma > 0
        inv = 1.0 / max(gamma, 1e-6)
        table = np.array([((i / 255.0) ** inv) * 255.0 for i in range(256)]).astype(
            np.uint8
        )
        out = cv2.LUT(img, table)
        return OperationResult(out, f"Gamma (gamma={gamma:.2f})")


# ----------------------------
# Main Window
# ----------------------------


class ImageLabel(QLabel):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.title = title
        self._pixmap_orig: Optional[QPixmap] = None
        self.setText(f"{title}\n\n(No Image)")

    def set_cv_image(self, img: Optional[np.ndarray]):
        if img is None:
            self._pixmap_orig = None
            self.setText(f"{self.title}\n\n(No Image)")
            self.setPixmap(QPixmap())
            return
        pm = QPixmap.fromImage(cv_to_qimage(img))
        self._pixmap_orig = pm
        self._refresh()

    def _refresh(self):
        if self._pixmap_orig is None:
            return
        self.setPixmap(fit_pixmap_to_label(self._pixmap_orig, self))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Pipeline Designer")
        self.resize(1400, 800)

        # Data model
        self.original_image: Optional[np.ndarray] = None
        self.states: List[np.ndarray] = []
        self.history_steps: List[str] = []
        self.state_index: int = -1
        self.preview_image: Optional[np.ndarray] = None
        self.current_operation: str = "Average Blur"

        # For Difference operation
        self.diff_image_raw: Optional[np.ndarray] = None
        self.diff_image_path: str = ""

        # Debounce preview updates
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_preview)

        # Build UI
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        # Left: main panel + history
        left = QWidget()
        left_layout = QVBoxLayout(left)

        img_group = QGroupBox("Main Panel")
        img_grid = QGridLayout(img_group)

        self.before_label = ImageLabel("Before")
        self.after_label = ImageLabel("After")
        img_grid.addWidget(self.before_label, 0, 0)
        img_grid.addWidget(self.after_label, 0, 1)

        left_layout.addWidget(img_group, stretch=6)

        hist_group = QGroupBox("History")
        hist_layout = QVBoxLayout(hist_group)
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.TypeWriter)
        self.history_text.setFont(mono)
        hist_layout.addWidget(self.history_text)
        left_layout.addWidget(hist_group, stretch=4)

        # Right: control panels
        right = QWidget()
        right_layout = QVBoxLayout(right)

        func_group = QGroupBox("Function Panel")
        func_layout = QVBoxLayout(func_group)

        self.btn_load = QPushButton("Load Image")
        self.btn_apply = QPushButton("Apply Step")
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        self.btn_reset = QPushButton("Reset")

        func_layout.addWidget(self.btn_load)
        func_layout.addWidget(self.btn_apply)

        row2 = QHBoxLayout()
        row2.addWidget(self.btn_undo)
        row2.addWidget(self.btn_redo)
        func_layout.addLayout(row2)

        func_layout.addWidget(self.btn_reset)
        right_layout.addWidget(func_group)

        op_group = QGroupBox("Image Operation Panel")
        op_layout = QVBoxLayout(op_group)

        self.op_selector = QComboBox()
        self.op_selector.addItems(
            [
                "Average Blur",
                "Gaussian Blur",
                "Median Blur",
                "Threshold",
                "Adaptive Threshold",
                "Otsu Threshold",
                "Sobel",
                "Canny",
                "Morphology",
                "Difference",
                "CLAHE",
                "Unsharp Mask",
                "Gamma",
            ]
        )
        op_layout.addWidget(QLabel("Operation"))
        op_layout.addWidget(self.op_selector)

        # Parameters: use QStackedWidget
        self.param_group = QGroupBox("Parameters")
        self.param_stack = QStackedWidget()
        pg_layout = QVBoxLayout(self.param_group)
        pg_layout.addWidget(self.param_stack)
        op_layout.addWidget(self.param_group)

        right_layout.addWidget(op_group, stretch=1)
        right_layout.addStretch(1)

        main_layout.addWidget(left, stretch=7)
        main_layout.addWidget(right, stretch=3)

        # Build parameter pages + signals
        self.params: Dict[str, Dict[str, Any]] = {}
        self.op_to_stack_index: Dict[str, int] = {}
        self._build_param_pages()
        self._connect_param_signals()

        # Wire events
        self.btn_load.clicked.connect(self.load_image)
        self.btn_apply.clicked.connect(self.apply_step)
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.btn_reset.clicked.connect(self.reset)
        self.op_selector.currentTextChanged.connect(self.on_operation_changed)

        self.on_operation_changed(self.op_selector.currentText())
        self.set_controls_enabled(False)
        self.refresh_history()

    # ----------------------------
    # Parameter pages
    # ----------------------------

    def _add_param_page(self, op_name: str, page_widget: QWidget):
        idx = self.param_stack.addWidget(page_widget)
        self.op_to_stack_index[op_name] = idx
        self.params[op_name]["page"] = page_widget

    def _build_param_pages(self):
        odd_1_31 = list(range(1, 32, 2))
        odd_3_31 = list(range(3, 32, 2))

        # Average Blur
        self.params["Average Blur"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        avg_k = OddSlider("Kernel Size", odd_1_31, 5)
        l.addWidget(avg_k)
        l.addStretch(1)
        self.params["Average Blur"].update({"avg_k": avg_k})
        self._add_param_page("Average Blur", w)

        # Gaussian Blur
        self.params["Gaussian Blur"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        g_k = OddSlider("Kernel Size", odd_1_31, 5)
        g_sigma = FloatSlider("Sigma", 0.0, 10.0, 1.2, scale=10)
        l.addWidget(g_k)
        l.addWidget(g_sigma)
        l.addStretch(1)
        self.params["Gaussian Blur"].update({"g_k": g_k, "g_sigma": g_sigma})
        self._add_param_page("Gaussian Blur", w)

        # Median Blur
        self.params["Median Blur"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        m_k = OddSlider("Kernel Size", odd_1_31, 5)
        l.addWidget(m_k)
        l.addStretch(1)
        self.params["Median Blur"].update({"m_k": m_k})
        self._add_param_page("Median Blur", w)

        # Threshold (NEW)
        self.params["Threshold"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        t_val = IntSlider("Threshold", 0, 255, 128)
        t_inv = QCheckBox("Invert")
        l.addWidget(t_val)
        l.addWidget(t_inv)
        l.addStretch(1)
        self.params["Threshold"].update({"t_val": t_val, "t_inv": t_inv})
        self._add_param_page("Threshold", w)

        # Adaptive Threshold
        self.params["Adaptive Threshold"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        at_block = OddSlider("Block Size", odd_3_31, 11)
        at_C = IntSlider("C", -20, 20, 2)
        at_method = QComboBox()
        at_method.addItems(["Mean", "Gaussian"])
        l.addWidget(at_block)
        l.addWidget(at_C)
        l.addWidget(QLabel("Method"))
        l.addWidget(at_method)
        l.addStretch(1)
        self.params["Adaptive Threshold"].update(
            {"at_block": at_block, "at_C": at_C, "at_method": at_method}
        )
        self._add_param_page("Adaptive Threshold", w)

        # Otsu Threshold
        self.params["Otsu Threshold"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        otsu_inv = QCheckBox("Invert")
        l.addWidget(otsu_inv)
        l.addStretch(1)
        self.params["Otsu Threshold"].update({"otsu_inv": otsu_inv})
        self._add_param_page("Otsu Threshold", w)

        # Sobel
        self.params["Sobel"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        s_ksize = QComboBox()
        s_ksize.addItems(["1", "3", "5", "7"])
        s_mode = QComboBox()
        s_mode.addItems(["X", "Y", "XY"])
        l.addWidget(QLabel("Kernel Size"))
        l.addWidget(s_ksize)
        l.addWidget(QLabel("Direction"))
        l.addWidget(s_mode)
        l.addStretch(1)
        self.params["Sobel"].update({"s_ksize": s_ksize, "s_mode": s_mode})
        self._add_param_page("Sobel", w)

        # Canny
        self.params["Canny"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        c_t1 = IntSlider("Threshold 1", 0, 255, 80)
        c_t2 = IntSlider("Threshold 2", 0, 255, 160)
        c_ap = QComboBox()
        c_ap.addItems(["3", "5", "7"])
        c_l2 = QCheckBox("L2 Gradient")
        l.addWidget(c_t1)
        l.addWidget(c_t2)
        l.addWidget(QLabel("Aperture Size"))
        l.addWidget(c_ap)
        l.addWidget(c_l2)
        l.addStretch(1)
        self.params["Canny"].update(
            {"c_t1": c_t1, "c_t2": c_t2, "c_ap": c_ap, "c_l2": c_l2}
        )
        self._add_param_page("Canny", w)

        # Morphology
        self.params["Morphology"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        mo_op = QComboBox()
        mo_op.addItems(
            ["Erode", "Dilate", "Open", "Close", "Gradient", "TopHat", "BlackHat"]
        )
        mo_shape = QComboBox()
        mo_shape.addItems(["Rect", "Ellipse", "Cross"])
        mo_k = OddSlider("Kernel Size", odd_1_31, 5)
        mo_it = IntSlider("Iterations", 1, 10, 1)
        l.addWidget(QLabel("Operation"))
        l.addWidget(mo_op)
        l.addWidget(QLabel("Kernel Shape"))
        l.addWidget(mo_shape)
        l.addWidget(mo_k)
        l.addWidget(mo_it)
        l.addStretch(1)
        self.params["Morphology"].update(
            {"mo_op": mo_op, "mo_shape": mo_shape, "mo_k": mo_k, "mo_it": mo_it}
        )
        self._add_param_page("Morphology", w)

        # Difference
        self.params["Difference"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        diff_btn = QPushButton("Load Second Image")
        diff_path = QLabel("(No second image)")
        diff_path.setWordWrap(True)
        diff_abs = QCheckBox("Abs Difference")
        diff_abs.setChecked(True)
        diff_norm = QCheckBox("Normalize Output")
        diff_norm.setChecked(False)
        l.addWidget(diff_btn)
        l.addWidget(diff_path)
        l.addWidget(diff_abs)
        l.addWidget(diff_norm)
        l.addStretch(1)
        self.params["Difference"].update(
            {
                "diff_btn": diff_btn,
                "diff_path": diff_path,
                "diff_abs": diff_abs,
                "diff_norm": diff_norm,
            }
        )
        self._add_param_page("Difference", w)

        # CLAHE
        self.params["CLAHE"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        clahe_clip = FloatSlider("Clip Limit", 0.5, 10.0, 2.0, scale=10)
        clahe_tile = IntSlider("Tile Size", 2, 16, 8)
        l.addWidget(clahe_clip)
        l.addWidget(clahe_tile)
        l.addStretch(1)
        self.params["CLAHE"].update(
            {"clahe_clip": clahe_clip, "clahe_tile": clahe_tile}
        )
        self._add_param_page("CLAHE", w)

        # Unsharp Mask
        self.params["Unsharp Mask"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        us_k = OddSlider("Kernel Size", odd_1_31, 5)
        us_sigma = FloatSlider("Sigma", 0.0, 10.0, 1.0, scale=10)
        # amount 0..3.0
        us_amount = FloatSlider("Amount", 0.0, 3.0, 1.0, scale=100)
        l.addWidget(us_k)
        l.addWidget(us_sigma)
        l.addWidget(us_amount)
        l.addStretch(1)
        self.params["Unsharp Mask"].update(
            {"us_k": us_k, "us_sigma": us_sigma, "us_amount": us_amount}
        )
        self._add_param_page("Unsharp Mask", w)

        # Gamma
        self.params["Gamma"] = {}
        w = QWidget()
        l = QVBoxLayout(w)
        # gamma 0.1..3.0
        gamma = FloatSlider("Gamma", 0.10, 3.00, 1.00, scale=100)
        l.addWidget(gamma)
        l.addStretch(1)
        self.params["Gamma"].update({"gamma": gamma})
        self._add_param_page("Gamma", w)

        # Hook Difference button here (needs access to self)
        self.params["Difference"]["diff_btn"].clicked.connect(
            self.load_difference_image
        )

    def _connect_param_signals(self):
        def hook(widget):
            if isinstance(widget, OddSlider):
                widget.slider.valueChanged.connect(self.schedule_preview_update)
            elif isinstance(widget, IntSlider):
                widget.slider.valueChanged.connect(self.schedule_preview_update)
            elif isinstance(widget, FloatSlider):
                widget.slider.valueChanged.connect(self.schedule_preview_update)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.schedule_preview_update)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self.schedule_preview_update)

        for op_name, pack in self.params.items():
            for k, v in pack.items():
                if k == "page":
                    continue
                # Difference button handled separately; QLabel no need
                if op_name == "Difference" and k in ("diff_btn", "diff_path"):
                    continue
                hook(v)

    # ----------------------------
    # Controls enable/disable
    # ----------------------------

    def set_controls_enabled(self, enabled: bool):
        self.btn_apply.setEnabled(enabled)
        self.btn_undo.setEnabled(enabled)
        self.btn_redo.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)
        self.op_selector.setEnabled(enabled)
        self.param_group.setEnabled(enabled)

    # ----------------------------
    # File loading
    # ----------------------------

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not path:
            return

        img = read_image_any_path(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image.")
            return

        self.original_image = img
        self.states = [img.copy()]
        self.history_steps = []
        self.state_index = 0
        self.preview_image = img.copy()

        # reset difference state (new base image -> new reference makes sense)
        self.diff_image_raw = None
        self.diff_image_path = ""
        self.params["Difference"]["diff_path"].setText("(No second image)")

        self.before_label.set_cv_image(self.states[self.state_index])
        self.after_label.set_cv_image(self.preview_image)

        self.set_controls_enabled(True)
        self.refresh_history()
        self.schedule_preview_update()

    def load_difference_image(self):
        if self.state_index < 0 or not self.states:
            QMessageBox.information(self, "Info", "Please load an image first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Second Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if not path:
            return

        img2 = read_image_any_path(path)
        if img2 is None:
            QMessageBox.critical(self, "Error", "Failed to load second image.")
            return

        self.diff_image_raw = img2
        self.diff_image_path = path
        base_name = os.path.basename(path)
        self.params["Difference"]["diff_path"].setText(f"Second: {base_name}")
        self.schedule_preview_update()

    # ----------------------------
    # Operation selection & preview
    # ----------------------------

    def on_operation_changed(self, op_name: str):
        self.current_operation = op_name
        idx = self.op_to_stack_index.get(op_name, 0)
        self.param_stack.setCurrentIndex(idx)
        self.schedule_preview_update()

    def schedule_preview_update(self):
        if self.state_index < 0 or not self.states:
            return
        self.preview_timer.start(50)

    def update_preview(self):
        if self.state_index < 0 or not self.states:
            return
        base = self.states[self.state_index]
        op = self.current_operation
        try:
            result = self._run_operation(op, base)
            self.preview_image = result.image
            self.after_label.set_cv_image(self.preview_image)
        except Exception as e:
            QMessageBox.warning(
                self, "Preview Error", f"Failed to preview operation.\n\n{e}"
            )

    def _run_operation(self, op: str, base: np.ndarray) -> OperationResult:
        if base is None:
            raise ValueError("No image loaded.")

        if op == "Average Blur":
            k = self.params[op]["avg_k"].value()
            return Processor.avg_blur(base, k)

        if op == "Gaussian Blur":
            k = self.params[op]["g_k"].value()
            sigma = self.params[op]["g_sigma"].value()
            return Processor.gaussian_blur(base, k, sigma)

        if op == "Median Blur":
            k = self.params[op]["m_k"].value()
            return Processor.median_blur(base, k)

        if op == "Threshold":
            t = self.params[op]["t_val"].value()
            inv = self.params[op]["t_inv"].isChecked()
            return Processor.threshold(base, t, inv)

        if op == "Adaptive Threshold":
            block = self.params[op]["at_block"].value()
            C = self.params[op]["at_C"].value()
            method = self.params[op]["at_method"].currentText()
            return Processor.adaptive_threshold(base, block, C, method)

        if op == "Otsu Threshold":
            inv = self.params[op]["otsu_inv"].isChecked()
            return Processor.otsu_threshold(base, inv)

        if op == "Sobel":
            ksize = int(self.params[op]["s_ksize"].currentText())
            mode = self.params[op]["s_mode"].currentText()
            return Processor.sobel(base, ksize, mode)

        if op == "Canny":
            t1 = self.params[op]["c_t1"].value()
            t2 = self.params[op]["c_t2"].value()
            aperture = int(self.params[op]["c_ap"].currentText())
            l2 = self.params[op]["c_l2"].isChecked()
            return Processor.canny(base, t1, t2, aperture, l2)

        if op == "Morphology":
            mop = self.params[op]["mo_op"].currentText()
            shape = self.params[op]["mo_shape"].currentText()
            k = self.params[op]["mo_k"].value()
            iters = self.params[op]["mo_it"].value()
            return Processor.morphology(base, mop, shape, k, iters)

        if op == "Difference":
            if self.diff_image_raw is None:
                raise ValueError("Difference: please click 'Load Second Image' first.")
            ref = self.diff_image_raw
            ref = resize_to_match(ref, base)
            use_abs = self.params[op]["diff_abs"].isChecked()
            normalize = self.params[op]["diff_norm"].isChecked()
            res = Processor.absdiff(base, ref, use_abs=use_abs, normalize=normalize)
            # enrich description with file name (nice for history)
            name = (
                os.path.basename(self.diff_image_path)
                if self.diff_image_path
                else "second"
            )
            res.description = f"{res.description} (ref={name})"
            return res

        if op == "CLAHE":
            clip = self.params[op]["clahe_clip"].value()
            tile = self.params[op]["clahe_tile"].value()
            return Processor.clahe(base, clip_limit=clip, tile=tile)

        if op == "Unsharp Mask":
            k = self.params[op]["us_k"].value()
            sigma = self.params[op]["us_sigma"].value()
            amount = self.params[op]["us_amount"].value()
            return Processor.unsharp_mask(base, k=k, sigma=sigma, amount=amount)

        if op == "Gamma":
            g = self.params[op]["gamma"].value()
            return Processor.gamma_correction(base, gamma=g)

        raise ValueError(f"Unknown operation: {op}")

    # ----------------------------
    # Pipeline actions
    # ----------------------------

    def apply_step(self):
        if self.state_index < 0 or self.preview_image is None:
            return

        base = self.states[self.state_index]
        try:
            result = self._run_operation(self.current_operation, base)
        except Exception as e:
            QMessageBox.warning(
                self, "Apply Error", f"Failed to apply operation.\n\n{e}"
            )
            return

        # If user had undone before, truncate redo branch
        if self.state_index < len(self.states) - 1:
            self.states = self.states[: self.state_index + 1]
            self.history_steps = self.history_steps[: self.state_index]

        self.states.append(result.image.copy())
        self.history_steps.append(result.description)
        self.state_index += 1

        self.before_label.set_cv_image(self.states[self.state_index])
        self.preview_image = self.states[self.state_index].copy()
        self.after_label.set_cv_image(self.preview_image)

        self.refresh_history()
        self.schedule_preview_update()

    def undo(self):
        if self.state_index <= 0:
            return
        self.state_index -= 1
        cur = self.states[self.state_index]
        self.before_label.set_cv_image(cur)
        self.preview_image = cur.copy()
        self.after_label.set_cv_image(self.preview_image)
        self.refresh_history()
        self.schedule_preview_update()

    def redo(self):
        if self.state_index < 0:
            return
        if self.state_index >= len(self.states) - 1:
            return
        self.state_index += 1
        cur = self.states[self.state_index]
        self.before_label.set_cv_image(cur)
        self.preview_image = cur.copy()
        self.after_label.set_cv_image(self.preview_image)
        self.refresh_history()
        self.schedule_preview_update()

    def reset(self):
        if self.original_image is None:
            return
        self.states = [self.original_image.copy()]
        self.history_steps = []
        self.state_index = 0
        self.preview_image = self.states[0].copy()
        self.before_label.set_cv_image(self.states[0])
        self.after_label.set_cv_image(self.preview_image)
        self.refresh_history()
        self.schedule_preview_update()

    # ----------------------------
    # History
    # ----------------------------

    def refresh_history(self):
        if self.original_image is None or self.state_index < 0:
            self.history_text.setPlainText("")
            return

        steps_to_show = self.history_steps[: self.state_index]
        lines = ["0) Original"]
        for i, s in enumerate(steps_to_show, start=1):
            lines.append(f"{i}) {s}")
        lines.append("")
        lines.append(f"Current index: {self.state_index} / {len(self.states) - 1}")
        self.history_text.setPlainText("\n".join(lines))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
