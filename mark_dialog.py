# -*- coding: utf-8 -*-
"""IDCardMuse marking / fine-tune dialog (PySide2).

Mirrors the Tk 1110-fix behavior: opens fit-to-window, green round handles,
full-width blue confirm button, no confirmation popups. Two entry flows:
- marking: a 12%-inset preset box is seeded immediately so the window is
  usable at once, and auto-detect runs on a background thread (S1 fix —
  it used to freeze the opening window for the whole rescue chain);
  when the result lands it replaces the preset unless the user has
  already started dragging.
- click-to-place: with fewer than 4 points, clicks place points in any order.
Wheel zoom is multiplicative and anchored at the cursor; dragging the
background pans when zoomed.
"""
import os, sys, threading, itertools
from PySide2.QtCore import Qt, Signal, QPointF, QObject
from PySide2.QtGui import QPixmap, QPainter, QColor, QPen, QBrush, QFont
from PySide2.QtWidgets import (QDialog, QWidget, QVBoxLayout,
                               QPushButton, QMessageBox, QLabel)
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app_qt import pil_to_qimage, BLUE

HANDLE_R = 11
GREEN = QColor('#2FBF71')


class _DetectBridge(QObject):
    """标角窗自动识别的跨线程信号桥（模块级长命对象）：工作线程只往
    它身上发信号，弹窗销毁后连接自动断开，迟到结果无人接收、安全
    丢弃——避免工作线程直接往可能已销毁的弹窗对象上 emit。"""
    done = Signal(int, object)


_DETECT_BRIDGE = _DetectBridge()
_DETECT_SEQ = itertools.count(1)


class MarkCanvas(QWidget):
    changed = Signal()

    def __init__(self, pil_img):
        super().__init__()
        self.img = pil_img
        self.qimg = pil_to_qimage(pil_img)
        self.points = []          # image coords [(x, y), ...] up to 4
        self.scale = 1.0
        self.off = QPointF(0, 0)  # view coords of image top-left
        self._drag_idx = None
        self._pan_start = None
        self._pan_off = None
        self._user_view = False   # becomes True after manual zoom/pan
        # S6: 缩放结果缓存（键 = 目标像素尺寸）。旧实现每次 paintEvent
        # 都对全分辨率原图做 Smooth 缩放，拖角时 mousemove 连续重绘
        # 即连续全图缩放——慢机上就是 Tk 线「整图重绘」卡顿的对应物。
        self._scaled_cache = {}
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)

    def _scaled(self, w, h):
        key = (w, h)
        hit = self._scaled_cache.get(key)
        if hit is None:
            if len(self._scaled_cache) > 16:
                self._scaled_cache.clear()
            hit = self.qimg.scaled(w, h, Qt.IgnoreAspectRatio,
                                   Qt.SmoothTransformation)
            self._scaled_cache[key] = hit
        return hit

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_view:
            self.fit()

    # ----- transforms -----
    def fit(self):
        w, h = self.width(), self.height()
        iw, ih = self.img.width, self.img.height
        pad = 16
        self.scale = min((w - 2 * pad) / iw, (h - 2 * pad) / ih)
        self.off = QPointF((w - iw * self.scale) / 2, (h - ih * self.scale) / 2)

    def to_view(self, p):
        return QPointF(self.off.x() + p[0] * self.scale,
                       self.off.y() + p[1] * self.scale)

    def to_image(self, v):
        return ((v.x() - self.off.x()) / self.scale,
                (v.y() - self.off.y()) / self.scale)

    # ----- painting -----
    def paintEvent(self, event):
        from PySide2.QtGui import QPainterPath
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QColor('white'))
        iw, ih = self.img.width * self.scale, self.img.height * self.scale
        x0, y0 = int(self.off.x()), int(self.off.y())
        path = QPainterPath()
        path.addRoundedRect(x0, y0, int(iw), int(ih), 12, 12)
        qp.save()
        qp.setClipPath(path)
        qp.drawImage(x0, y0, self._scaled(int(iw), int(ih)))
        qp.restore()
        if len(self.points) >= 2:
            qp.setPen(QPen(GREEN, 2))
            vs = [self.to_view(p) for p in self.points]
            n = len(vs)
            edges = n if n == 4 else n - 1
            for i in range(edges):
                qp.drawLine(vs[i], vs[(i + 1) % n])
        for p in self.points:
            v = self.to_view(p)
            qp.setPen(QPen(QColor('white'), 2))
            qp.setBrush(QBrush(GREEN))
            qp.drawEllipse(v, HANDLE_R, HANDLE_R)
        qp.end()

    # ----- interaction -----
    def _hit_handle(self, pos):
        for i, p in enumerate(self.points):
            v = self.to_view(p)
            if (v.x() - pos.x()) ** 2 + (v.y() - pos.y()) ** 2 <= (HANDLE_R + 7) ** 2:
                return i
        return None

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        idx = self._hit_handle(event.pos())
        if idx is not None:
            self._drag_idx = idx
            return
        self._user_view = True
        self._pan_start = event.pos()
        self._pan_off = QPointF(self.off)

    def mouseMoveEvent(self, event):
        if self._drag_idx is not None:
            x, y = self.to_image(event.pos())
            x = min(max(x, 0), self.img.width - 1)
            y = min(max(y, 0), self.img.height - 1)
            self.points[self._drag_idx] = (x, y)
            self.changed.emit()
            self.update()
        elif self._pan_start is not None:
            self.off = self._pan_off + (event.pos() - self._pan_start)
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag_idx = None
        self._pan_start = None

    def wheelEvent(self, event):
        self._user_view = True
        old = self.scale
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale = min(max(old * factor, 0.05), 12.0)
        # anchor zoom at cursor: keep the image point under cursor fixed
        ix, iy = self.to_image(event.pos()) if old == self.scale else (0, 0)
        if old != self.scale:
            ix, iy = ((event.pos().x() - self.off.x()) / old,
                      (event.pos().y() - self.off.y()) / old)
            self.off = QPointF(event.pos().x() - ix * self.scale,
                               event.pos().y() - iy * self.scale)
        self.update()


class MarkDialog(QDialog):
    def __init__(self, win, idx, side):
        super().__init__(win)
        self.win = win
        self.core = win.core
        self.idx = idx
        self.side = side
        data = self.core.pairs[idx][side]
        self.img = data['image']
        self.result_points = None
        self.setWindowTitle('手动微调边缘（拖拽四角）')
        # size the window to the photo's aspect, like the original
        iw, ih = self.img.width, self.img.height
        cw = 980
        ch = int(cw * ih / iw)
        if ch > 700:
            ch = 700
            cw = int(ch * iw / ih)
        if ch < 380:
            ch = 380
        self.resize(cw + 20, ch + 66)
        self.setStyleSheet('background: white;')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        self.canvas = MarkCanvas(self.img)
        self.canvas.changed.connect(self._sync_state)
        self.canvas.changed.connect(self._note_touched)
        lay.addWidget(self.canvas, 1)

        self.hint = QLabel('')
        self.hint.setStyleSheet('color: #5B6470; font-size: 13px;')
        lay.addWidget(self.hint)

        self.confirm_b = QPushButton('✓ 确认微调')
        self.confirm_b.setFixedHeight(42)
        self.confirm_b.setStyleSheet(
            f'QPushButton {{ background: {BLUE}; color: white; border: none;'
            ' border-radius: 4px; font-size: 15px; font-weight: bold; }')
        self.confirm_b.clicked.connect(self.confirm)
        lay.addWidget(self.confirm_b)

        # initial points: stored -> use directly. Otherwise seed the
        # preset box at once (window usable immediately) and run
        # auto-detect on a background thread (S1: the synchronous call
        # used to freeze the opening window for the whole rescue chain).
        self._alive = True
        self._touched = False
        self._detect_token = None
        if data.get('points'):
            self.canvas.points = [tuple(p) for p in data['points']]
        else:
            self.seed_preset()
            self._start_auto_detect()
        self.canvas.fit()
        self._sync_state()

    # ----- logic -----
    def seed_preset(self):
        w, h = self.img.width, self.img.height
        self.canvas.points = [(0.12 * w, 0.12 * h), (0.88 * w, 0.12 * h),
                              (0.88 * w, 0.88 * h), (0.12 * w, 0.88 * h)]
        self.canvas.update()

    def _note_touched(self):
        self._touched = True

    def _start_auto_detect(self):
        self._detect_token = next(_DETECT_SEQ)
        token = self._detect_token
        self.hint.setText('正在自动识别边框…')
        _DETECT_BRIDGE.done.connect(self._on_auto_detect_done)
        core, img = self.core, self.img

        def worker():
            corners = None
            try:
                corners = core.ai_detect_any(img)
            except Exception:
                corners = None
            _DETECT_BRIDGE.done.emit(token, corners)

        threading.Thread(target=worker, daemon=True).start()

    def _on_auto_detect_done(self, token, corners):
        if token != self._detect_token or not self._alive:
            return
        self._disconnect_detect()
        if corners and not self._touched:
            self.canvas.points = [tuple(p) for p in corners]
            self.canvas.update()
            self.hint.setText('已自动识别，可拖动四角微调')
        elif corners:
            self.hint.setText('已自动识别，但你已开始手动调整，按你的为准')
        elif getattr(self.core, '_ai_error', None) \
                and not getattr(self.core, '_ai_ran', False):
            self.hint.setText('识别模型不可用，请手动拖动四角对齐证件边缘')
        else:
            self.hint.setText('没自动认出，已放预置框，拖动四角对齐证件边缘')
        self._sync_state()

    def _disconnect_detect(self):
        try:
            _DETECT_BRIDGE.done.disconnect(self._on_auto_detect_done)
        except Exception:
            pass

    def closeEvent(self, event):
        self._alive = False      # 迟到的识别结果不再应用
        self._disconnect_detect()
        super().closeEvent(event)

    def _sync_state(self):
        self.confirm_b.setEnabled(len(self.canvas.points) == 4)

    def confirm(self):
        if len(self.canvas.points) != 4:
            return
        self.core.current_marking_side = self.side
        processed = self.core.process_card(self.img, list(self.canvas.points))
        if processed is None:
            QMessageBox.warning(self, '提示', '4 个角围的范围太小或交叉了，挪开一点再确认')
            return
        data = self.core.pairs[self.idx][self.side]
        data['points'] = [tuple(p) for p in self.canvas.points]
        data['processed'] = processed
        if hasattr(self.win, '_schedule_ocr'):
            self.win._schedule_ocr(self.idx, self.side, processed)
        row = self.win.rows[self.idx]
        (row.front if self.side == 'front' else row.back).set_processed(processed)
        name = '正面' if self.side == 'front' else '反面'
        self.win.set_status(f'第 {self.idx + 1} 组{name}微调完成')
        self.accept()
