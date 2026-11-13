# -*- coding: utf-8 -*-
"""营业执照模式（Qt 线）：导入 → lic_pipeline 处理 → 预览 → 导出。

算法全在 bigdoc.py / lic_enhance.py / lic_pipeline.py，本文件只做 UI 与接线：
- 单一规格（2026-11-12 青店实测裁决）：A4 横版 2339×1654 @200dpi，
  即青在 PS 里把执照拉正后的实际出图尺寸；A3 处理路已删（店里 A3
  正本直接上扫描仪、不走拍照识别，照片路只剩 A4 量级）；
- JPG 嵌 200dpi；PDF 在本文件内以 200dpi 定页（不能走
  core._pages_to_pdf_bytes：那是身份证 A4 页、按 core 的 300dpi
  盖章，会把 2339×1654 印成非 A4 尺寸），页面即 A4 297×210mm；
- 自动流程对不上纸边（fallback / no_quad）→ 手动标角窗（复用标角画布）
  框纸面四角后拉正调光；竖拍照片可用「旋转90°」转正后重处理。
"""
import os, sys, threading, tempfile, shutil
import numpy as np
import cv2
from PySide2.QtCore import Qt, QObject, Signal
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                               QLabel, QFileDialog, QMessageBox)
from PIL import Image
try:
    import img2pdf
except Exception:
    img2pdf = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app_qt import pil_to_qimage, BLUE, PAGE_BG, report_saved
from mark_dialog import MarkCanvas
from lic_pipeline import process_license, order_quad, OUT_A4, DEFAULT_MODEL
from bigdoc import dark_border_fraction
from lic_enhance import enhance_license

DARK_WARN = 0.08   # 任一边黑边占比超过此值 → 提示疑似黑边


class _ProcBridge(QObject):
    """执照管线跨线程信号桥（模块级长命对象）：工作线程只往它身上发
    信号，弹窗销毁后连接自动断开，迟到结果无人接收、安全丢弃。"""
    done = Signal(int, object)


_PROC_BRIDGE = _ProcBridge()


def imread_bgr(path):
    """cv2 读图，兼容中文路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _btn_style():
    return (f'QPushButton {{ background: {BLUE}; color: white; border: none;'
            ' border-radius: 4px; padding: 6px 14px; font-size: 13.5px; }'
            'QPushButton:hover { background: #2569D1; }'
            'QPushButton:disabled { background: #A9C6F5; }')


class LicenseMarkDialog(QDialog):
    """执照手动标角：拖动四角到纸面边缘，确认后回传四角（原图坐标）。"""

    def __init__(self, parent, pil_img, seed_points=None):
        super().__init__(parent)
        self.result_points = None
        self.setWindowTitle('手动标角（拖动四角到纸面边缘）')
        iw, ih = pil_img.width, pil_img.height
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
        self.canvas = MarkCanvas(pil_img)
        lay.addWidget(self.canvas, 1)
        if seed_points and len(seed_points) == 4:
            self.canvas.points = [tuple(map(float, p)) for p in seed_points]
        else:
            w, h = iw, ih
            self.canvas.points = [(0.12 * w, 0.12 * h), (0.88 * w, 0.12 * h),
                                  (0.88 * w, 0.88 * h), (0.12 * w, 0.88 * h)]
        confirm_b = QPushButton('✓ 确认标角')
        confirm_b.setFixedHeight(42)
        confirm_b.setStyleSheet(
            f'QPushButton {{ background: {BLUE}; color: white; border: none;'
            ' border-radius: 4px; font-size: 15px; font-weight: bold; }')
        confirm_b.clicked.connect(self._confirm)
        lay.addWidget(confirm_b)
        self.canvas.fit()

    def _confirm(self):
        if len(self.canvas.points) != 4:
            return
        self.result_points = [tuple(p) for p in self.canvas.points]
        self.accept()


class LicenseDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle('营业执照')
        self.resize(900, 780)
        self.setStyleSheet(f'background: {PAGE_BG};')
        self.src_bgr = None
        self.src_path = None
        self.result = None        # process_license 返回 dict 或手动流程同构 dict
        self.warped = None        # 当前可导出的 BGR 图
        # S1: 管线改后台线程跑；_proc_seq 每发起一次递增，完成回主
        # 线程时序号/源图对不上即弃（旋转、换图都会取代旧的）
        self._proc_seq = 0
        self._busy = False
        self._processing_src = None
        _PROC_BRIDGE.done.connect(self._on_proc_done)

        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.import_b = QPushButton('导入执照图')
        self.import_b.setStyleSheet(_btn_style())
        self.import_b.clicked.connect(self.import_image)
        top.addWidget(self.import_b)
        self.mark_b = QPushButton('手动标角')
        self.mark_b.setStyleSheet(_btn_style())
        self.mark_b.clicked.connect(self.manual_mark)
        top.addWidget(self.mark_b)
        self.rotate_b = QPushButton('旋转90°重处理')
        self.rotate_b.setStyleSheet(_btn_style())
        self.rotate_b.clicked.connect(self.rotate_source)
        top.addWidget(self.rotate_b)
        top.addStretch(1)
        self.jpg_b = QPushButton('导出 JPG')
        self.jpg_b.setStyleSheet(_btn_style())
        self.jpg_b.clicked.connect(self.export_jpg)
        top.addWidget(self.jpg_b)
        self.pdf_b = QPushButton('导出 PDF')
        self.pdf_b.setStyleSheet(_btn_style())
        self.pdf_b.clicked.connect(self.export_pdf)
        top.addWidget(self.pdf_b)
        lay.addLayout(top)

        self.view = QLabel('点「导入执照图」选择营业执照照片或电子档图片')
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet(
            'background: white; border: 1px solid #D5DBE3; border-radius: 5px;'
            ' color: #8A94A3; font-size: 14px;')
        lay.addWidget(self.view, 1)

        self.info = QLabel('就绪')
        self.info.setAlignment(Qt.AlignCenter)
        self.info.setStyleSheet('color: #5B6470; font-size: 13px;')
        lay.addWidget(self.info)
        self._sync_buttons()

    # ---------- helpers ----------
    def _out_size(self):
        return OUT_A4

    def _detector(self):
        """共用 core 的唯一检测器（S5 修复）：旧实现查的
        core.detector_ai 从未被赋值、永远落空，于是每次开执照窗都
        在 UI 线程自建第二个 40MB 模型，且构造在管线异常守卫之外、
        模型缺失能直接穿出槽函数。现在统一走 core.get_ai_detector
        （懒加载＋缓存＋失败闩锁＋留痕）；模型不可用时返回 None，
        管线按它自己的守卫降级。本方法会触发构造，只在后台线程
        调用（见 _process 的 worker）。"""
        get = getattr(self.win.core, 'get_ai_detector', None)
        if get is not None:
            return get()
        return getattr(self.win.core, 'detector_ai', None)

    def _sync_buttons(self):
        has_src = self.src_bgr is not None
        self.mark_b.setEnabled(has_src)
        self.rotate_b.setEnabled(has_src)
        self.jpg_b.setEnabled(self.warped is not None)
        self.pdf_b.setEnabled(self.warped is not None)

    def _show_bgr(self, bgr):
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pm = QPixmap.fromImage(pil_to_qimage(pil))
        pm = pm.scaled(840, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(pm)

    def _dark_note(self, dark):
        if not dark:
            return ''
        worst = max((v for k, v in dark.items() if not k.startswith('_')),
                    default=0.0)
        if worst > DARK_WARN:
            return f'边缘疑似有黑边（{worst:.0%}），可用手动标角微调'
        return ''

    # ---------- flow ----------
    def import_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择营业执照图片', self.win.last_dir,
            '图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)')
        if path:
            self.load(path)

    def load(self, path):
        bgr = imread_bgr(path)
        if bgr is None:
            QMessageBox.warning(self, '提示', '图片打开失败，换一张试试')
            return False
        self.src_path = path
        self.src_bgr = bgr
        self.win.last_dir = os.path.dirname(path)
        self.win._save_settings()
        self._process()
        return True

    def _set_busy(self, b):
        """处理中：全部动作钮禁用＋信息行提示（S1 的忙碌指示）；
        结束时按当前状态恢复（导出钮经 _sync_buttons 门闩）。"""
        self._busy = b
        if b:
            self.info.setText('正在处理，请稍候…')
            for w in (self.import_b, self.mark_b, self.rotate_b,
                      self.jpg_b, self.pdf_b):
                w.setEnabled(False)
        else:
            self.import_b.setEnabled(True)
            self._sync_buttons()

    def _process(self):
        """管线在后台线程跑（S1：旧实现同步跑完整管线＋首次构造
        模型，全程冻窗、无任何提示）。完成经信号回主线程，序号或
        源图对不上（已被旋转/换图的新一轮取代）即弃。"""
        if self.src_bgr is None:
            return
        self._proc_seq += 1
        seq = self._proc_seq
        src = self.src_bgr
        self._processing_src = src
        out_size = self._out_size()
        self._set_busy(True)

        def worker():
            try:
                res = process_license(src, detector=self._detector(),
                                      out_size=out_size)
            except Exception as exc:
                # process_license 本有守卫，这里是双保险：无论如何都
                # 要给主线程一个完成回执，否则忙碌态永不解除
                res = {'status': 'no_quad', 'route': None, 'flat': False,
                       'warped': None, 'paper_quad': None, 'fallback': None,
                       'dewarp_skipped_high_bow': False,
                       'info': {'reason': 'pipeline_error',
                                'error': f'{type(exc).__name__}: {exc}'}}
            _PROC_BRIDGE.done.emit(seq, res)

        threading.Thread(target=worker, daemon=True).start()

    def _on_proc_done(self, seq, res):
        if seq != self._proc_seq or self._processing_src is not self.src_bgr:
            return    # 迟到结果：已被新一轮处理取代，弃
        self.result = res
        if res['status'] != 'ok':
            self.warped = None
            self._show_bgr(self.src_bgr)
            # S2 口径：模型不可用与照片没认出分开说
            if getattr(self.win.core, '_ai_error', None):
                self.info.setText('识别模型不可用（原因记在程序旁边的 '
                                  'detect_error.log），请点「手动标角」'
                                  '框出纸面四角')
            elif res['info'].get('error'):
                self.info.setText(f"自动处理出错（{res['info']['error']}），"
                                  '请点「手动标角」框出纸面四角')
            else:
                self.info.setText('没认出执照边框，请点「手动标角」框出纸面四角')
            self._set_busy(False)
            return
        self.warped = res['warped']
        self._show_bgr(self.warped)
        notes = []
        if res['flat']:
            notes.append('电子档/平扫图：整图即纸面，已跳过外探与调光')
        if res['fallback'] == 'frame_quad':
            notes.append('外探没对上纸边，已按花边框拉正，建议手动标角核对')
        if res.get('dewarp_skipped_high_bow'):
            notes.append('这张照片卷曲较重，没能拉平，本次只做了透视拉正')
        note = self._dark_note(res['info'].get('dark_border'))
        if note:
            notes.append(note)
        self.info.setText('；'.join(notes) if notes else '处理完成，可导出')
        self._set_busy(False)

    def rotate_source(self):
        if self.src_bgr is None:
            return
        self.src_bgr = cv2.rotate(self.src_bgr, cv2.ROTATE_90_CLOCKWISE)
        self._process()

    def manual_mark(self):
        if self.src_bgr is None:
            return
        pil = Image.fromarray(cv2.cvtColor(self.src_bgr, cv2.COLOR_BGR2RGB))
        seed = None
        if self.result and self.result.get('paper_quad') is not None:
            seed = [tuple(map(float, p)) for p in self.result['paper_quad']]
        dlg = LicenseMarkDialog(self, pil, seed)
        if dlg.exec_() != QDialog.Accepted or not dlg.result_points:
            return
        quad = order_quad(dlg.result_points)
        warped = self._warp_from_quad(quad)
        self.warped = warped
        self.result = {'status': 'ok', 'route': 'manual', 'flat': False,
                       'warped': warped, 'paper_quad': quad, 'fallback': None,
                       'info': {'dark_border': dark_border_fraction(warped)}}
        self._show_bgr(warped)
        note = self._dark_note(self.result['info']['dark_border'])
        self.info.setText(note if note else '已按手动标角拉正，可导出')
        self._sync_buttons()

    def _warp_from_quad(self, quad):
        W, H = self._out_size()
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                       np.float32)
        M = cv2.getPerspectiveTransform(np.asarray(quad, np.float32), dst)
        warped = cv2.warpPerspective(self.src_bgr, M, (W, H))
        warped, _einfo = enhance_license(warped)
        return warped

    # ---------- export ----------
    def _pil(self):
        return Image.fromarray(cv2.cvtColor(self.warped, cv2.COLOR_BGR2RGB))

    def _after_save(self, path):
        folder = os.path.dirname(path)
        self.win.last_dir = folder
        self.win._save_settings()
        self.win.set_status(f'营业执照已导出：{path}')
        report_saved(self, folder, f'已保存：{os.path.basename(path)}')

    def export_jpg(self):
        if self.warped is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '导出 JPG',
            os.path.join(self.win.last_dir or os.path.expanduser('~'),
                         '营业执照.jpg'),
            'JPG (*.jpg)')
        if not path:
            return
        if not path.lower().endswith(('.jpg', '.jpeg')):
            path += '.jpg'
        # S4: 写文件守卫，失败弹错误框（与主窗信息导出口径一致）
        try:
            self._pil().save(path, quality=95, dpi=(200, 200))
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败：{e}')
            return
        self._after_save(path)

    def export_pdf(self):
        if self.warped is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '导出 PDF',
            os.path.join(self.win.last_dir or os.path.expanduser('~'),
                         '营业执照.pdf'),
            'PDF (*.pdf)')
        if not path:
            return
        if not path.lower().endswith('.pdf'):
            path += '.pdf'
        data = self._license_pdf_bytes()
        if data is None:
            QMessageBox.warning(self, '提示', 'PDF 生成失败，请改用导出 JPG')
            return
        # S4: 写文件守卫，失败弹错误框（与主窗信息导出口径一致）
        try:
            with open(path, 'wb') as f:
                f.write(data)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败：{e}')
            return
        self._after_save(path)

    def _license_pdf_bytes(self):
        """执照 PDF：2339×1654 以 200dpi 定页 = A4 横版 297×210mm。
        不走 core._pages_to_pdf_bytes（那是身份证页、按 300dpi 盖章，
        会把执照页印成约 198×140mm）。临时目录用完即清（同 core 的
        客户证件页纪律：不把整页渲染留在 %TEMP%）。"""
        if img2pdf is None:
            return None
        tmpdir = tempfile.mkdtemp()
        try:
            fp = os.path.join(tmpdir, 'license.jpg')
            self._pil().save(fp, quality=95, dpi=(200, 200))
            return img2pdf.convert(fp)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
