import sys
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

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
    QSpinBox,
    QFrame,
    QSizePolicy,
    QStackedWidget,
)

# ----------------------------
# Utilities: image <-> QImage
# ----------------------------

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


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
    return cv2.imread(path, cv2.IMREAD_COLOR)


def write_image_any_path(path: str, bgr: np.ndarray) -> None:
    """Robust write for unicode paths on Windows/Linux using imencode+tofile."""
    ext = os.path.splitext(path)[1].lower()
    if ext == "":
        path += ".png"
        ext = ".png"

    ext_map = {
        ".png": ".png",
        ".jpg": ".jpg",
        ".jpeg": ".jpg",
        ".bmp": ".bmp",
        ".tif": ".tif",
        ".tiff": ".tif",
        ".webp": ".webp",
    }
    if ext not in ext_map:
        raise ValueError("Unsupported file extension.")

    ok, buf = cv2.imencode(ext_map[ext], bgr)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    buf.tofile(path)


def resize_to_match(src_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """Resize src to exactly match target shape (H, W)."""
    th, tw = target_bgr.shape[:2]
    return cv2.resize(src_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)


def list_images_in_dir(folder: str) -> List[str]:
    if not folder or not os.path.isdir(folder):
        return []
    files = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and name.lower().endswith(IMAGE_EXTS):
            files.append(p)
    files.sort(key=lambda s: os.path.basename(s).lower())
    return files


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
        if use_abs:
            out = cv2.absdiff(img, ref_img)
        else:
            out = cv2.subtract(img, ref_img)  # saturating subtract

        if normalize:
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
    def cc_area_filter(
        img: np.ndarray,
        area_thresh_px: int,
        mode: str,
        invert_input: bool,
        ignore_border: bool,
        connectivity: int,
    ) -> OperationResult:
        """
        Connected-component filter using **pixel area** threshold.

        - mode: "Remove Larger Than" or "Remove Smaller Than"
        - invert_input: invert the binary before processing (if your foreground is black)
        - ignore_border: ignore components that touch the image border
        - connectivity: 4 or 8
        Notes:
          This operation binarizes the image via Otsu internally. If your input is already binary,
          Otsu typically preserves it.
        """
        if img is None:
            raise ValueError("No image.")

        gray = ensure_gray(img)

        # Otsu binarization: foreground tends to be white for typical masks
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if invert_input:
            bin_img = 255 - bin_img

        h, w = bin_img.shape[:2]
        total_px = int(h * w)
        if total_px <= 0:
            return OperationResult(ensure_bgr(bin_img), "CC Area Filter (empty image)")

        # Analyze black components inside the white foreground:
        # Invert so black regions become white for CC analysis
        inv = 255 - bin_img

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            inv, connectivity=connectivity
        )
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)

        # Border-touching components (usually the outside background in this inverted view)
        border_touch = [False] * n
        if ignore_border and n > 1:
            top = labels[0, :]
            bot = labels[h - 1, :]
            left = labels[:, 0]
            right = labels[:, w - 1]
            touched = np.unique(
                np.concatenate([top, bot, left, right]).astype(np.int32)
            )
            for idx in touched.tolist():
                if 0 <= idx < n:
                    border_touch[idx] = True

        out_bin = bin_img.copy()
        thr = int(max(0, min(int(area_thresh_px), total_px)))
        remove_larger = mode == "Remove Larger Than"

        for i in range(1, n):
            if ignore_border and border_touch[i]:
                continue

            area_i = int(areas[i])
            cond = (area_i > thr) if remove_larger else (area_i < thr)

            if cond:
                # Removing a black component means "filling it" -> set to white in original bin
                out_bin[labels == i] = 255

        if invert_input:
            out_bin = 255 - out_bin

        return OperationResult(
            ensure_bgr(out_bin),
            f"CC Area Filter (mode={mode}, thr={thr}px, conn={connectivity}, invert={invert_input}, ignore_border={ignore_border})",
        )

    @staticmethod
    def largest_cc_cluster(
        img: np.ndarray,
        eps_px: int,
        invert_input: bool,
        ignore_border: bool,
        connectivity: int,
    ) -> OperationResult:
        """
        DBSCAN-like clustering for black connected components (holes/speckles) inside a white foreground.
        Keep the cluster whose members have the largest **summed area**, and fill other black components
        to white.

        eps_px: centroid distance threshold (pixels).
        """
        if img is None:
            raise ValueError("No image.")

        gray = ensure_gray(img)
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if invert_input:
            bin_img = 255 - bin_img

        h, w = bin_img.shape[:2]
        if h <= 0 or w <= 0:
            return OperationResult(
                ensure_bgr(bin_img), "Largest CC Cluster (empty image)"
            )

        inv = 255 - bin_img
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inv, connectivity=connectivity
        )
        if n <= 1:
            out_bin = bin_img.copy()
            if invert_input:
                out_bin = 255 - out_bin
            return OperationResult(
                ensure_bgr(out_bin), f"Largest CC Cluster (eps={eps_px}px, no CC)"
            )

        border_touch = [False] * n
        if ignore_border:
            top = labels[0, :]
            bot = labels[h - 1, :]
            left = labels[:, 0]
            right = labels[:, w - 1]
            touched = np.unique(
                np.concatenate([top, bot, left, right]).astype(np.int32)
            )
            for idx in touched.tolist():
                if 0 <= idx < n:
                    border_touch[idx] = True

        cand_labels = []
        cand_pts = []
        cand_areas = []
        for lab in range(1, n):
            if ignore_border and border_touch[lab]:
                continue
            area = int(stats[lab, cv2.CC_STAT_AREA])
            cx, cy = centroids[lab]
            cand_labels.append(lab)
            cand_pts.append((float(cx), float(cy)))
            cand_areas.append(area)

        if len(cand_labels) <= 1:
            out_bin = bin_img.copy()
            if invert_input:
                out_bin = 255 - out_bin
            return OperationResult(
                ensure_bgr(out_bin),
                f"Largest CC Cluster (eps={eps_px}px, candidates<=1)",
            )

        eps = max(1, int(eps_px))
        eps2 = float(eps * eps)
        k = len(cand_labels)

        adj = [[] for _ in range(k)]
        for i in range(k):
            xi, yi = cand_pts[i]
            for j in range(i + 1, k):
                xj, yj = cand_pts[j]
                dx = xi - xj
                dy = yi - yj
                if (dx * dx + dy * dy) <= eps2:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * k
        best_members = None
        best_area_sum = -1
        for i in range(k):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            members = []
            area_sum = 0
            while stack:
                u = stack.pop()
                members.append(u)
                area_sum += int(cand_areas[u])
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            if area_sum > best_area_sum:
                best_area_sum = area_sum
                best_members = members

        keep_labels = (
            set(cand_labels[i] for i in best_members) if best_members else set()
        )

        out_bin = bin_img.copy()
        for lab in cand_labels:
            if lab not in keep_labels:
                out_bin[labels == lab] = 255

        if invert_input:
            out_bin = 255 - out_bin

        return OperationResult(
            ensure_bgr(out_bin),
            f"Largest CC Cluster (eps={eps}px, conn={connectivity}, invert={invert_input}, ignore_border={ignore_border})",
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
        blur = cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)
        out = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return OperationResult(
            out, f"Unsharp Mask (k={k}, sigma={sigma:.2f}, amount={amount:.2f})"
        )

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma: float) -> OperationResult:
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
        self.setWindowTitle("Image Pipeline Designer (PyQt5 + OpenCV)")
        self.resize(1600, 850)

        # Data model
        self.original_image: Optional[np.ndarray] = None
        self.states: List[np.ndarray] = (
            []
        )  # cached images for each step (len = steps+1)
        self.history_steps: List[str] = []  # step descriptions (len = steps)
        self.pipeline_ops: List[Dict[str, Any]] = (
            []
        )  # replayable steps: [{"op": str, "params": dict}, ...]
        self.state_index: int = -1  # current committed state index (0..len(states)-1)
        self.preview_image: Optional[np.ndarray] = None
        self.current_operation: str = "Average Blur"

        # For Difference operation (UI convenience)
        self.diff_image_raw: Optional[np.ndarray] = None
        self.diff_image_path: str = ""
        self.diff_folder: str = ""
        self.diff_folder_images: List[str] = []
        self.diff_folder_index: int = -1

        # Folder navigation state
        self.current_image_path: str = ""
        self.current_folder: str = ""
        self.current_folder_images: List[str] = []
        self.current_folder_index: int = -1

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

        self.orig_label = ImageLabel("Original")  # NEW pane
        self.before_label = ImageLabel("Current")
        self.after_label = ImageLabel("Preview")
        img_grid.addWidget(self.orig_label, 0, 0)
        img_grid.addWidget(self.before_label, 0, 1)
        img_grid.addWidget(self.after_label, 0, 2)

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
        func_layout.setSpacing(10)

        # --- Source / pipeline ---
        self.btn_load = QPushButton("Load Image")
        self.btn_load.setToolTip("Load an image and clear all history.")
        self.btn_apply_new = QPushButton("Apply Pipeline to New Image")
        self.btn_apply_new.setToolTip(
            "Pick another image and re-apply current pipeline."
        )

        # --- Actions ---
        self.btn_apply = QPushButton("Apply Step")
        self.btn_apply.setToolTip("Commit current preview as a pipeline step.")
        self.btn_save = QPushButton("Save Result")
        self.btn_save.setToolTip("Save current result image to disk.")

        # --- Folder navigator ---
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        for _b in (self.btn_prev, self.btn_next):
            _b.setFixedWidth(44)
            _b.setMinimumHeight(36)

        self.btn_prev.setToolTip("Previous image in this folder")
        self.btn_next.setToolTip("Next image in this folder")

        self.nav_label = QLabel("No folder")
        self.nav_label.setAlignment(Qt.AlignCenter)
        self.nav_label.setWordWrap(True)
        self.nav_label.setMinimumHeight(36)

        # --- History controls ---
        self.btn_undo = QPushButton("Undo")
        self.btn_redo = QPushButton("Redo")
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Reset to the original image and clear all history.")

        for _b in (
            self.btn_load,
            self.btn_apply_new,
            self.btn_apply,
            self.btn_save,
            self.btn_undo,
            self.btn_redo,
            self.btn_reset,
        ):
            _b.setMinimumHeight(36)

        # Row: load + apply pipeline to new image
        src_row = QHBoxLayout()
        src_row.addWidget(self.btn_load, 1)
        src_row.addWidget(self.btn_apply_new, 1)
        func_layout.addLayout(src_row)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setFrameShadow(QFrame.Sunken)
        func_layout.addWidget(sep1)

        # Row: navigator (◀ [index/name] ▶)
        nav_row = QHBoxLayout()
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.nav_label, 1)
        nav_row.addWidget(self.btn_next)
        func_layout.addLayout(nav_row)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setFrameShadow(QFrame.Sunken)
        func_layout.addWidget(sep2)

        # Row: apply + save
        act_row = QHBoxLayout()
        act_row.addWidget(self.btn_apply, 1)
        act_row.addWidget(self.btn_save, 1)
        func_layout.addLayout(act_row)

        # Row: undo + redo
        hist_row = QHBoxLayout()
        hist_row.addWidget(self.btn_undo, 1)
        hist_row.addWidget(self.btn_redo, 1)
        func_layout.addLayout(hist_row)

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
                "CC Area Filter",
                "Largest CC Cluster",
                "CLAHE",
                "Unsharp Mask",
                "Gamma",
            ]
        )
        op_layout.addWidget(QLabel("Operation"))
        op_layout.addWidget(self.op_selector)

        # Parameters: QStackedWidget
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
        self.btn_load.clicked.connect(self.load_image_with_warning)
        self.btn_apply_new.clicked.connect(self.apply_pipeline_to_new_image)
        self.btn_apply.clicked.connect(self.apply_step)
        self.btn_save.clicked.connect(self.save_result_image)
        self.btn_prev.clicked.connect(self.navigate_prev)
        self.btn_next.clicked.connect(self.navigate_next)

        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.btn_reset.clicked.connect(self.reset)
        self.op_selector.currentTextChanged.connect(self.on_operation_changed)

        self.on_operation_changed(self.op_selector.currentText())
        self.set_controls_enabled(False)
        self.refresh_history()
        self._update_nav_buttons()

    # ----------------------------
    # Keyboard navigation
    # ----------------------------

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.navigate_prev()
            return
        if event.key() == Qt.Key_Right:
            self.navigate_next()
            return
        super().keyPressEvent(event)

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

        # Threshold
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

        # CC Area Filter (pixel-based)
        self.params["CC Area Filter"] = {}
        w = QWidget()
        l = QVBoxLayout(w)

        cc_thr_px = QSpinBox()
        cc_thr_px.setMinimum(0)
        cc_thr_px.setMaximum(999999999)  # will be updated to H*W after loading image
        cc_thr_px.setValue(200)
        cc_thr_px.setSingleStep(10)

        cc_mode = QComboBox()
        cc_mode.addItems(["Remove Larger Than", "Remove Smaller Than"])
        cc_conn = QComboBox()
        cc_conn.addItems(["8", "4"])
        cc_inv = QCheckBox("Invert Input Binary")
        cc_ignore_border = QCheckBox("Ignore Border Components")
        cc_ignore_border.setChecked(True)

        l.addWidget(QLabel("Area Threshold (px)"))
        l.addWidget(cc_thr_px)
        l.addWidget(QLabel("Mode"))
        l.addWidget(cc_mode)
        l.addWidget(QLabel("Connectivity"))
        l.addWidget(cc_conn)
        l.addWidget(cc_inv)
        l.addWidget(cc_ignore_border)
        l.addStretch(1)

        self.params["CC Area Filter"].update(
            {
                "cc_thr_px": cc_thr_px,
                "cc_mode": cc_mode,
                "cc_conn": cc_conn,
                "cc_inv": cc_inv,
                "cc_ignore_border": cc_ignore_border,
            }
        )
        self._add_param_page("CC Area Filter", w)

        # Largest CC Cluster
        self.params["Largest CC Cluster"] = {}
        w = QWidget()
        l = QVBoxLayout(w)

        lcc_eps = QSpinBox()
        lcc_eps.setMinimum(1)
        lcc_eps.setMaximum(999999999)  # updated to max(H, W) after loading image
        lcc_eps.setValue(40)
        lcc_eps.setSingleStep(5)

        lcc_conn = QComboBox()
        lcc_conn.addItems(["8", "4"])
        lcc_inv = QCheckBox("Invert Input Binary")
        lcc_ignore_border = QCheckBox("Ignore Border Components")
        lcc_ignore_border.setChecked(True)

        l.addWidget(QLabel("Cluster Radius (px)"))
        l.addWidget(lcc_eps)
        l.addWidget(QLabel("Connectivity"))
        l.addWidget(lcc_conn)
        l.addWidget(lcc_inv)
        l.addWidget(lcc_ignore_border)
        l.addStretch(1)

        self.params["Largest CC Cluster"].update(
            {
                "lcc_eps": lcc_eps,
                "lcc_conn": lcc_conn,
                "lcc_inv": lcc_inv,
                "lcc_ignore_border": lcc_ignore_border,
            }
        )
        self._add_param_page("Largest CC Cluster", w)

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
        gamma = FloatSlider("Gamma", 0.10, 3.00, 1.00, scale=100)
        l.addWidget(gamma)
        l.addStretch(1)
        self.params["Gamma"].update({"gamma": gamma})
        self._add_param_page("Gamma", w)

        # Hook Difference button (needs self)
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
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self.schedule_preview_update)

        for op_name, pack in self.params.items():
            for k, v in pack.items():
                if k == "page":
                    continue
                if op_name == "Difference" and k in ("diff_btn", "diff_path"):
                    continue
                hook(v)

    # ----------------------------
    # Controls enable/disable
    # ----------------------------

    def set_controls_enabled(self, enabled: bool):
        self.btn_apply_new.setEnabled(enabled)
        self.btn_apply.setEnabled(enabled)
        self.btn_save.setEnabled(enabled)
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        self.btn_undo.setEnabled(enabled)
        self.btn_redo.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)
        self.op_selector.setEnabled(enabled)
        self.param_group.setEnabled(enabled)

    def _update_nav_buttons(self):
        has_list = (
            len(self.current_folder_images) >= 2 and self.current_folder_index >= 0
        )
        self.btn_prev.setEnabled(has_list and self.state_index >= 0)
        self.btn_next.setEnabled(has_list and self.state_index >= 0)

    def _update_nav_label(self):
        """Update navigator label: show index/total + filename."""
        try:
            if not hasattr(self, "nav_label") or self.nav_label is None:
                return
        except Exception:
            return

        if not self.current_image_path:
            self.nav_label.setText("No folder")
            return

        name = os.path.basename(self.current_image_path)
        if self.current_folder_index >= 0 and len(self.current_folder_images) > 0:
            self.nav_label.setText(
                f"{self.current_folder_index + 1}/{len(self.current_folder_images)}\n{name}"
            )
        else:
            self.nav_label.setText(name)

    def _update_lcc_eps_max(self):
        """Set Largest CC Cluster eps maximum to max(H, W)."""
        if self.original_image is None:
            return
        if "Largest CC Cluster" not in self.params:
            return
        sb = self.params["Largest CC Cluster"].get("lcc_eps", None)
        if sb is None:
            return
        h, w = self.original_image.shape[:2]
        mx = int(max(h, w))
        try:
            sb.setMaximum(mx)
            if sb.value() > mx:
                sb.setValue(mx)
        except Exception:
            pass

    def _update_cc_area_filter_max(self):
        """Set CC Area Filter threshold maximum to current image H*W (pixel count)."""
        if self.original_image is None:
            return
        if "CC Area Filter" not in self.params:
            return
        sb = self.params["CC Area Filter"].get("cc_thr_px", None)
        if sb is None:
            return
        h, w = self.original_image.shape[:2]
        mx = int(h * w)
        try:
            sb.setMaximum(mx)
            if sb.value() > mx:
                sb.setValue(mx)
        except Exception:
            pass

    # ----------------------------
    # Load image (reset) with warning
    # ----------------------------

    def load_image_with_warning(self):
        # Warn only when there exists at least one history operation
        if self.pipeline_ops or self.history_steps:
            reply = QMessageBox.question(
                self,
                "Warning",
                "Load new image will clear all history.\n\nContinue?",
                QMessageBox.Ok | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply != QMessageBox.Ok:
                return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)",
        )
        if not path:
            return

        img = read_image_any_path(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image.")
            return

        # Reset everything
        self.original_image = img
        self.states = [img.copy()]
        self.history_steps = []
        self.pipeline_ops = []
        self.state_index = 0
        self.preview_image = img.copy()

        # reset difference state
        self.diff_image_raw = None
        self.diff_image_path = ""
        self.params["Difference"]["diff_path"].setText("(No second image)")

        self._set_current_file(path)

        self.orig_label.set_cv_image(self.original_image)
        self._update_cc_area_filter_max()
        self._update_lcc_eps_max()
        self.before_label.set_cv_image(self.states[self.state_index])
        self.after_label.set_cv_image(self.preview_image)

        self.set_controls_enabled(True)
        self.refresh_history()
        self.schedule_preview_update()
        self._update_nav_buttons()

    # ----------------------------
    # Difference reference
    # ----------------------------

    def _is_difference_in_pipeline(self) -> bool:
        return any(step.get("op") == "Difference" for step in (self.pipeline_ops or []))

    def _set_difference_reference(
        self,
        path: str,
        *,
        update_pipeline: bool = True,
        schedule_preview: bool = True,
    ):
        """Set/remember the difference reference image path, build folder state, and optionally update pipeline."""
        if not path:
            return

        img2 = read_image_any_path(path)
        if img2 is None:
            raise ValueError("Failed to load second image.")

        self.diff_image_raw = img2
        self.diff_image_path = path

        # Folder state for auto prev/next syncing
        self.diff_folder = os.path.dirname(path)
        self.diff_folder_images = list_images_in_dir(self.diff_folder)
        try:
            self.diff_folder_index = self.diff_folder_images.index(path)
        except ValueError:
            rp = os.path.realpath(path)
            idx = -1
            for i, p in enumerate(self.diff_folder_images):
                if os.path.realpath(p) == rp:
                    idx = i
                    break
            self.diff_folder_index = idx

        # Update label on Difference param page
        try:
            self.params["Difference"]["diff_path"].setText(
                f"Second: {os.path.basename(path)}"
            )
        except Exception:
            pass

        # Keep pipeline Difference steps in sync (so replay uses the current reference)
        if update_pipeline and self.pipeline_ops:
            for step in self.pipeline_ops:
                if step.get("op") == "Difference":
                    step.setdefault("params", {})
                    step["params"]["path"] = path

        if schedule_preview:
            self.schedule_preview_update()

    def _prompt_difference_reference(self) -> bool:
        """Ask user to choose a new difference reference image."""
        start_dir = ""
        if self.diff_image_path:
            start_dir = os.path.dirname(self.diff_image_path)
        elif self.current_folder:
            start_dir = self.current_folder

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Difference Reference Image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)",
        )
        if not path:
            return False

        self._set_difference_reference(
            path, update_pipeline=True, schedule_preview=False
        )
        return True

    def _advance_difference_reference(self, delta: int):
        """When navigating base images (Prev/Next), also move the difference reference image within its own folder."""
        if not self._is_difference_in_pipeline():
            return
        if not self.diff_image_path:
            return

        # Ensure folder state exists
        if not self.diff_folder_images:
            self.diff_folder = os.path.dirname(self.diff_image_path)
            self.diff_folder_images = list_images_in_dir(self.diff_folder)
            try:
                self.diff_folder_index = self.diff_folder_images.index(
                    self.diff_image_path
                )
            except ValueError:
                self.diff_folder_index = -1

        if self.diff_folder_index < 0 or len(self.diff_folder_images) < 2:
            return

        new_idx = (self.diff_folder_index + int(delta)) % len(self.diff_folder_images)
        new_path = self.diff_folder_images[new_idx]
        self._set_difference_reference(
            new_path, update_pipeline=True, schedule_preview=False
        )

    def load_difference_image(self):
        if self.state_index < 0 or not self.states:
            QMessageBox.information(self, "Info", "Please load an image first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Difference Reference Image",
            self.current_folder if self.current_folder else "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)",
        )
        if not path:
            return

        try:
            self._set_difference_reference(
                path, update_pipeline=True, schedule_preview=True
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load second image.\n\n{e}")
            return

    def save_result_image(self):
        if self.state_index < 0 or not self.states:
            QMessageBox.information(self, "Info", "No result image to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result Image",
            "",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff);;WEBP (*.webp)",
        )
        if not path:
            return

        try:
            img = self.states[self.state_index]
            write_image_any_path(path, img)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image.\n\n{e}")

    # ----------------------------
    # Folder navigation
    # ----------------------------

    def _set_current_file(self, path: str):
        self.current_image_path = path
        self.current_folder = os.path.dirname(path)
        self.current_folder_images = list_images_in_dir(self.current_folder)
        try:
            self.current_folder_index = self.current_folder_images.index(path)
        except ValueError:
            # maybe path differs by case/realpath
            rp = os.path.realpath(path)
            idx = -1
            for i, p in enumerate(self.current_folder_images):
                if os.path.realpath(p) == rp:
                    idx = i
                    break
            self.current_folder_index = idx
        self._update_nav_label()
        self._update_nav_buttons()

    def navigate_prev(self):
        if self.current_folder_index < 0 or len(self.current_folder_images) < 2:
            return
        new_idx = (self.current_folder_index - 1) % len(self.current_folder_images)

        # Sync difference reference navigation, if needed
        self._advance_difference_reference(-1)

        self._load_image_keep_pipeline(self.current_folder_images[new_idx])

    def navigate_next(self):
        if self.current_folder_index < 0 or len(self.current_folder_images) < 2:
            return
        new_idx = (self.current_folder_index + 1) % len(self.current_folder_images)

        # Sync difference reference navigation, if needed
        self._advance_difference_reference(+1)

        self._load_image_keep_pipeline(self.current_folder_images[new_idx])

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
            if self.diff_image_raw is None and self.diff_image_path:
                maybe = read_image_any_path(self.diff_image_path)
                if maybe is not None:
                    self.diff_image_raw = maybe
                    self.params["Difference"]["diff_path"].setText(
                        f"Second: {os.path.basename(self.diff_image_path)}"
                    )

            if self.diff_image_raw is None:
                raise ValueError("Difference: please click 'Load Second Image' first.")

            ref = resize_to_match(self.diff_image_raw, base)
            use_abs = self.params[op]["diff_abs"].isChecked()
            normalize = self.params[op]["diff_norm"].isChecked()
            res = Processor.absdiff(base, ref, use_abs=use_abs, normalize=normalize)
            name = (
                os.path.basename(self.diff_image_path)
                if self.diff_image_path
                else "second"
            )
            res.description = f"{res.description} (ref={name})"
            return res

        if op == "CC Area Filter":
            area_thresh_px = int(self.params[op]["cc_thr_px"].value())
            mode = self.params[op]["cc_mode"].currentText()
            connectivity = int(self.params[op]["cc_conn"].currentText())
            inv = self.params[op]["cc_inv"].isChecked()
            ignore_border = self.params[op]["cc_ignore_border"].isChecked()
            return Processor.cc_area_filter(
                base,
                area_thresh_px=area_thresh_px,
                mode=mode,
                invert_input=inv,
                ignore_border=ignore_border,
                connectivity=connectivity,
            )

        if op == "Largest CC Cluster":
            eps_px = int(self.params[op]["lcc_eps"].value())
            connectivity = int(self.params[op]["lcc_conn"].currentText())
            inv = self.params[op]["lcc_inv"].isChecked()
            ignore_border = self.params[op]["lcc_ignore_border"].isChecked()
            return Processor.largest_cc_cluster(
                base,
                eps_px=eps_px,
                invert_input=inv,
                ignore_border=ignore_border,
                connectivity=connectivity,
            )

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
    # Pipeline replay (for new input)
    # ----------------------------

    def _collect_current_params(self, op: str) -> Dict[str, Any]:
        p = self.params[op]

        if op == "Average Blur":
            return {"k": p["avg_k"].value()}

        if op == "Gaussian Blur":
            return {"k": p["g_k"].value(), "sigma": p["g_sigma"].value()}

        if op == "Median Blur":
            return {"k": p["m_k"].value()}

        if op == "Threshold":
            return {"t": p["t_val"].value(), "invert": p["t_inv"].isChecked()}

        if op == "Adaptive Threshold":
            return {
                "block": p["at_block"].value(),
                "C": p["at_C"].value(),
                "method": p["at_method"].currentText(),
            }

        if op == "Otsu Threshold":
            return {"invert": p["otsu_inv"].isChecked()}

        if op == "Sobel":
            return {
                "ksize": int(p["s_ksize"].currentText()),
                "mode": p["s_mode"].currentText(),
            }

        if op == "Canny":
            return {
                "t1": p["c_t1"].value(),
                "t2": p["c_t2"].value(),
                "aperture": int(p["c_ap"].currentText()),
                "l2": p["c_l2"].isChecked(),
            }

        if op == "Morphology":
            return {
                "op": p["mo_op"].currentText(),
                "shape": p["mo_shape"].currentText(),
                "k": p["mo_k"].value(),
                "iters": p["mo_it"].value(),
            }

        if op == "Difference":
            return {
                "use_abs": p["diff_abs"].isChecked(),
                "normalize": p["diff_norm"].isChecked(),
                "path": self.diff_image_path,
            }

        if op == "CC Area Filter":
            return {
                "area_thresh_px": int(p["cc_thr_px"].value()),
                "mode": p["cc_mode"].currentText(),
                "connectivity": int(p["cc_conn"].currentText()),
                "invert_input": p["cc_inv"].isChecked(),
                "ignore_border": p["cc_ignore_border"].isChecked(),
            }

        if op == "Largest CC Cluster":
            return {
                "eps_px": int(p["lcc_eps"].value()),
                "connectivity": int(p["lcc_conn"].currentText()),
                "invert_input": p["lcc_inv"].isChecked(),
                "ignore_border": p["lcc_ignore_border"].isChecked(),
            }

        if op == "CLAHE":
            return {"clip": p["clahe_clip"].value(), "tile": p["clahe_tile"].value()}

        if op == "Unsharp Mask":
            return {
                "k": p["us_k"].value(),
                "sigma": p["us_sigma"].value(),
                "amount": p["us_amount"].value(),
            }

        if op == "Gamma":
            return {"gamma": p["gamma"].value()}

        return {}

    def _run_operation_with_params(
        self, op: str, base: np.ndarray, params: Dict[str, Any]
    ) -> OperationResult:
        if op == "Average Blur":
            return Processor.avg_blur(base, params["k"])

        if op == "Gaussian Blur":
            return Processor.gaussian_blur(base, params["k"], params["sigma"])

        if op == "Median Blur":
            return Processor.median_blur(base, params["k"])

        if op == "Threshold":
            return Processor.threshold(base, params["t"], params["invert"])

        if op == "Adaptive Threshold":
            return Processor.adaptive_threshold(
                base, params["block"], params["C"], params["method"]
            )

        if op == "Otsu Threshold":
            return Processor.otsu_threshold(base, params["invert"])

        if op == "Sobel":
            return Processor.sobel(base, params["ksize"], params["mode"])

        if op == "Canny":
            return Processor.canny(
                base, params["t1"], params["t2"], params["aperture"], params["l2"]
            )

        if op == "Morphology":
            return Processor.morphology(
                base, params["op"], params["shape"], params["k"], params["iters"]
            )

        if op == "Difference":
            ref_path = params.get("path", "")
            if not ref_path:
                raise ValueError("Difference step requires a reference image path.")
            ref = read_image_any_path(ref_path)
            if ref is None:
                raise ValueError(
                    f"Failed to load difference reference image: {os.path.basename(ref_path)}"
                )
            ref = resize_to_match(ref, base)
            res = Processor.absdiff(base, ref, params["use_abs"], params["normalize"])
            res.description = f"{res.description} (ref={os.path.basename(ref_path)})"
            return res

        if op == "CC Area Filter":
            return Processor.cc_area_filter(
                base,
                area_thresh_px=int(params["area_thresh_px"]),
                mode=params["mode"],
                invert_input=bool(params["invert_input"]),
                ignore_border=bool(params["ignore_border"]),
                connectivity=int(params["connectivity"]),
            )

        if op == "Largest CC Cluster":
            return Processor.largest_cc_cluster(
                base,
                eps_px=int(params["eps_px"]),
                invert_input=bool(params["invert_input"]),
                ignore_border=bool(params["ignore_border"]),
                connectivity=int(params["connectivity"]),
            )

        if op == "CLAHE":
            return Processor.clahe(base, params["clip"], params["tile"])

        if op == "Unsharp Mask":
            return Processor.unsharp_mask(
                base, params["k"], params["sigma"], params["amount"]
            )

        if op == "Gamma":
            return Processor.gamma_correction(base, params["gamma"])

        raise ValueError(f"Unknown op: {op}")

    def _recompute_states_for_image(
        self, img: np.ndarray
    ) -> Tuple[List[np.ndarray], Optional[str]]:
        """
        Recompute cached states by replaying pipeline_ops on img.
        Returns (states, error_message). error_message is None on success.
        """
        states = [img.copy()]
        try:
            for step in self.pipeline_ops:
                op = step["op"]
                params = step["params"]
                base = states[-1]
                result = self._run_operation_with_params(op, base, params)
                states.append(result.image.copy())
            return states, None
        except Exception as e:
            return [], str(e)

    def _load_image_keep_pipeline(self, path: str):
        img = read_image_any_path(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image.")
            return

        self._set_current_file(path)

        self.original_image = img
        self.orig_label.set_cv_image(self.original_image)
        self._update_cc_area_filter_max()
        self._update_lcc_eps_max()
        if self.pipeline_ops:
            states, err = self._recompute_states_for_image(img)
            if err is not None:
                QMessageBox.critical(
                    self, "Error", f"Failed to apply pipeline to new image.\n\n{err}"
                )
                return
            self.states = states
            self.state_index = len(self.states) - 1
        else:
            # No pipeline: behave like a simple load, but without clearing anything else (there's nothing to clear)
            self.states = [img.copy()]
            self.state_index = 0
            self.history_steps = []
            self.pipeline_ops = []

        cur = self.states[self.state_index]
        self.before_label.set_cv_image(cur)

        self.preview_image = cur.copy()
        self.after_label.set_cv_image(self.preview_image)

        self.set_controls_enabled(True)
        self.refresh_history()
        self.schedule_preview_update()
        self._update_nav_buttons()

    def apply_pipeline_to_new_image(self):
        if self.state_index < 0:
            QMessageBox.information(self, "Info", "Please load an image first.")
            return
        if not self.pipeline_ops:
            QMessageBox.information(self, "Info", "No pipeline steps to apply.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open New Image (Keep Pipeline)",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All Files (*)",
        )
        if not path:
            return

        # If pipeline contains Difference, let user pick a new reference image for this new base image
        if self._is_difference_in_pipeline():
            ok = self._prompt_difference_reference()
            if not ok:
                return

        self._load_image_keep_pipeline(path)

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
            self.pipeline_ops = self.pipeline_ops[: self.state_index]

        # Commit
        self.states.append(result.image.copy())
        self.history_steps.append(result.description)
        self.pipeline_ops.append(
            {
                "op": self.current_operation,
                "params": self._collect_current_params(self.current_operation),
            }
        )
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
        self._update_cc_area_filter_max()
        self._update_lcc_eps_max()
        self.states = [self.original_image.copy()]
        self.history_steps = []
        self.pipeline_ops = []
        self.state_index = 0
        self.preview_image = self.states[0].copy()

        self.before_label.set_cv_image(self.states[0])
        self.after_label.set_cv_image(self.preview_image)
        self.refresh_history()
        self.schedule_preview_update()

        # reset difference state
        self.diff_image_raw = None
        self.diff_image_path = ""
        self.params["Difference"]["diff_path"].setText("(No second image)")

    # ----------------------------
    # History
    # ----------------------------

    def refresh_history(self):
        if self.original_image is None or self.state_index < 0:
            self.history_text.setPlainText("")
            return

        steps_to_show = self.history_steps[: self.state_index]
        lines = []

        if self.current_image_path:
            lines.append(f"File: {os.path.basename(self.current_image_path)}")
            lines.append("")

        lines.append("0) Original")
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
