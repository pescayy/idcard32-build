# -*- coding: utf-8 -*-
"""IDCardMuse Qt main window (PySide2). UI layer only; all processing in core.py.
Layout follows IDCardMerge v2.2 (see IDCardMerge-v2.2-复刻对照表.md):
group rows as the main area, settings bar at the bottom, full-width orange
generate button, centered status line. Small radii: buttons 4px, slots 5px.
"""
import os, sys, json, threading
from PySide2.QtCore import Qt, Signal, QUrl
from PySide2.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QPen, QDesktopServices
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel,
                               QPushButton, QComboBox, QCheckBox, QLineEdit,
                               QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog,
                               QDialog, QMessageBox, QGridLayout)
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import Core
import ocr_qt

BLUE = '#2E7CF6'
ORANGE = '#F28C1E'
PAGE_BG = '#F5F6F8'
BAR_BG = '#EEF0F3'
SLOT_BG = '#FAFBFC'
ENH_MAP = {'原图': 'none', '彩色': 'color', '黑白': 'bw',
           '文档黑白': 'docbw', '文档彩色': 'doccolor'}
WM_COLORS = ['黑色', '蓝色', '红色', '灰色']
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.heif', '.tif', '.tiff')

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass


def open_folder(path):
    QDesktopServices.openUrl(QUrl.fromLocalFile(path))


def report_saved(parent, folder, message):
    """导出完成后报告保存位置，并给「打开所在文件夹」入口。"""
    box = QMessageBox(parent)
    box.setWindowTitle('导出成功')
    box.setText(f'{message}\n保存位置：{folder}')
    open_b = box.addButton('打开所在文件夹', QMessageBox.ActionRole)
    box.addButton('完成', QMessageBox.AcceptRole)
    box.exec_()
    if box.clickedButton() == open_b:
        open_folder(folder)


def pil_to_qimage(pil_img):
    img = pil_img.convert('RGBA')
    data = img.tobytes('raw', 'RGBA')
    return QImage(data, img.width, img.height, QImage.Format_RGBA8888).copy()


def new_slot_data():
    return {'image': None, 'processed': None, 'path': None, 'points': None,
            'ocr': {}, '_ocr_key': None}


def new_pair():
    return {'front': new_slot_data(), 'back': new_slot_data()}


class SlotWidget(QFrame):
    clicked = Signal(object)

    def __init__(self, side, label_text):
        super().__init__()
        self.side = side
        self.label_text = label_text
        self.setFixedSize(216, 136)
        self.setStyleSheet(
            f'QFrame {{ background: {SLOT_BG}; border: 1.5px dashed #B9C2CE;'
            ' border-radius: 5px; }')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.view = QLabel(label_text)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setWordWrap(True)
        self.view.setStyleSheet('color: #8A94A3; font-size: 13px; border: none;')
        lay.addWidget(self.view)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self)

    def set_processed(self, pil_img):
        if pil_img is None:
            self.view.clear()
            self.view.setText(self.label_text)
            self.setStyleSheet(
                f'QFrame {{ background: {SLOT_BG}; border: 1.5px dashed #B9C2CE;'
                ' border-radius: 5px; }')
            return
        pm = QPixmap.fromImage(pil_to_qimage(pil_img))
        pm = pm.scaled(206, 126, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(pm)
        self.setStyleSheet(
            f'QFrame {{ background: white; border: 1.5px solid {BLUE};'
            ' border-radius: 5px; }')


class PairRow(QFrame):
    def __init__(self, win, index):
        super().__init__()
        self.win = win
        self.index = index
        self.setStyleSheet('QFrame { background: white; border-radius: 6px; }')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)
        self.num = QLabel('')
        self.num.setStyleSheet('color: #5B6470; font-size: 13px;')
        self.num.setFixedWidth(52)
        lay.addWidget(self.num)
        self.front = SlotWidget('front', '正面（人像面）\n点击导入')
        self.back = SlotWidget('back', '反面（国徽面）\n点击导入')
        self.front.clicked.connect(win.on_slot_clicked)
        self.back.clicked.connect(win.on_slot_clicked)
        lay.addWidget(self.front)
        mid = QVBoxLayout()
        swap = QPushButton('⇄')
        swap.setFixedSize(34, 34)
        swap.setToolTip('正反互换')
        swap.setStyleSheet(self._btn_style())
        swap.clicked.connect(lambda: win.swap_pair(self.index))
        mid.addStretch(1)
        mid.addWidget(swap)
        mid.addStretch(1)
        lay.addLayout(mid)
        lay.addWidget(self.back)
        side = QVBoxLayout()
        clear_f = QPushButton('清空正面')
        clear_b = QPushButton('清空反面')
        del_row = QPushButton('删除本组')
        for b in (clear_f, clear_b, del_row):
            b.setStyleSheet(self._btn_style())
        clear_f.clicked.connect(lambda: win.clear_slot(self.index, 'front'))
        clear_b.clicked.connect(lambda: win.clear_slot(self.index, 'back'))
        del_row.clicked.connect(lambda: win.delete_pair(self.index))
        side.addWidget(clear_f)
        side.addWidget(clear_b)
        side.addWidget(del_row)
        side.addStretch(1)
        lay.addLayout(side)
        lay.addStretch(1)

    @staticmethod
    def _btn_style():
        return (f'QPushButton {{ background: {BAR_BG}; border: none; border-radius: 4px;'
                ' padding: 5px 10px; font-size: 12.5px; color: #333; }'
                f'QPushButton:hover {{ background: #E2E7EE; }}')


class MainWindow(QMainWindow):
    ocr_done = Signal(int, str, object, str)
    ocr_probe_done = Signal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('IDCardMuse 智能证件合并工具v1.0 by xm 2026-11')
        self.resize(880, 760)
        self.core = Core()
        self.rows = []
        self.settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'idcard_settings.json')
        self.last_dir = ''
        self._load_settings()
        self._build_ui()
        self.add_pair()
        self._restore_settings_ui()
        self.setAcceptDrops(True)
        # OCR（全能版并入）：后台线程回主线程走信号，不直接碰控件
        self.ocr_done.connect(self._on_ocr_done)
        self.ocr_probe_done.connect(self._on_ocr_probe)
        self._ocr_dead = False
        threading.Thread(target=self._probe_ocr, daemon=True).start()

    # ---------- OCR (full-edition merge; logic in ocr_qt.py) ----------
    def _probe_ocr(self):
        self.ocr_probe_done.emit(ocr_qt.ocr_available())

    def _on_ocr_probe(self, ok):
        if not ok:
            self._ocr_dead = True
            self.set_status('文字识别组件不可用，已停用读字（其余功能正常）')

    def _schedule_ocr(self, idx, side, pil_img):
        """处理结果变化后排队读字；同一张图只读一次（与全能版 _ocr_key 口径一致）。"""
        if pil_img is None or self._ocr_dead:
            return
        if not ocr_qt.ocr_available():
            self._ocr_dead = True
            self.set_status('文字识别组件不可用，已停用读字（其余功能正常）')
            return
        data = self.core.pairs[idx][side]
        if data.get('_ocr_key') == id(pil_img):
            return
        data['_ocr_key'] = id(pil_img)

        def worker():
            err_text = ''
            try:
                info = ocr_qt.read_card(pil_img, side)
            except Exception as e:
                info = {}
                err_text = repr(e)
                ocr_qt.log_ocr_error(e)
            self.ocr_done.emit(idx, side, info, err_text)

        threading.Thread(target=worker, daemon=True).start()

    def _on_ocr_done(self, idx, side, info, err_text):
        if not info and err_text:
            self._ocr_dead = True
            self.set_status('读字没跑起来，原因记在程序旁边的 ocr_error.log')
            return
        if idx < 0 or idx >= len(self.core.pairs):
            return
        self.core.pairs[idx][side]['ocr'] = info
        if not info:
            return
        if side == 'front' and info.get('姓名'):
            self.set_status('已识别：%s %s' % (info.get('姓名', ''),
                                               info.get('公民身份号码', '')))
        elif side == 'back' and info.get('签发机关'):
            self.set_status('已识别签发机关：%s' % info.get('签发机关', ''))

    # ---------- UI construction ----------
    def _build_ui(self):
        central = QWidget()
        central.setStyleSheet(f'background: {PAGE_BG};')
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet('QScrollArea { border: none; }')
        self.pairs_host = QWidget()
        self.pairs_host.setStyleSheet(f'background: {PAGE_BG};')
        self.pairs_lay = QVBoxLayout(self.pairs_host)
        self.pairs_lay.setContentsMargins(2, 2, 2, 2)
        self.pairs_lay.setSpacing(8)
        self.pairs_lay.addStretch(1)
        self.scroll.setWidget(self.pairs_host)
        root.addWidget(self.scroll, 1)

        bar = QFrame()
        bar.setStyleSheet(f'QFrame {{ background: {BAR_BG}; border-radius: 6px; }}')
        bl = QVBoxLayout(bar)
        bl.setContentsMargins(10, 8, 10, 8)
        bl.setSpacing(7)

        row1 = QHBoxLayout()
        add_btn = QPushButton('＋ 新增一行')
        add_btn.setStyleSheet(self._primary_style())
        add_btn.clicked.connect(self.add_pair)
        batch_btn = QPushButton('批量导入（两张一组）')
        batch_btn.setStyleSheet(self._primary_style())
        batch_btn.clicked.connect(self.batch_import)
        lic_btn = QPushButton('营业执照')
        lic_btn.setStyleSheet(self._primary_style())
        lic_btn.clicked.connect(self.open_license_dialog)
        row1.addWidget(add_btn)
        row1.addWidget(batch_btn)
        row1.addWidget(lic_btn)
        row1.addStretch(1)
        row1.addWidget(QLabel('增强：'))
        self.enh_combo = QComboBox()
        self.enh_combo.addItems(list(ENH_MAP.keys()))
        self.enh_combo.setCurrentText('彩色')
        self.enh_combo.setStyleSheet('QComboBox { border-radius: 4px; padding: 3px 6px; background: white; }')
        row1.addWidget(self.enh_combo)
        bl.addLayout(row1)

        row2 = QHBoxLayout()
        self.wm_check = QCheckBox('安全水印')
        row2.addWidget(self.wm_check)
        self.wm_text = QLineEdit()
        self.wm_text.setPlaceholderText('水印内容（可空）')
        self.wm_text.setFixedWidth(150)
        self.wm_text.setStyleSheet('QLineEdit { background: white; border-radius: 4px; padding: 3px 6px; }')
        row2.addWidget(self.wm_text)
        row2.addWidget(QLabel('字号'))
        self.wm_size = self._combo([str(i) for i in range(4, 15)], '7')
        row2.addWidget(self.wm_size)
        row2.addWidget(QLabel('颜色'))
        self.wm_color = self._combo(WM_COLORS, '黑色')
        row2.addWidget(self.wm_color)
        row2.addWidget(QLabel('透明度%'))
        self.wm_opacity = self._combo(['5', '10', '15', '18', '20', '25', '30', '40', '50', '60'], '18')
        row2.addWidget(self.wm_opacity)
        row2.addWidget(QLabel('行数'))
        self.wm_rows = self._combo(['1', '2', '3', '4', '5', '6'], '3')
        row2.addWidget(self.wm_rows)
        row2.addStretch(1)
        row2.addWidget(QLabel('模板'))
        self.template_combo = self._combo(['标准（每组一页）', '紧凑拼版'], '标准（每组一页）')
        row2.addWidget(self.template_combo)
        bl.addLayout(row2)
        root.addWidget(bar)

        self.status = QLabel('就绪')
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet('color: #5B6470; font-size: 13px;')
        root.addWidget(self.status)

        gen = QPushButton('生成预览')
        gen.setFixedHeight(44)
        gen.setStyleSheet(
            f'QPushButton {{ background: {ORANGE}; color: white; border: none;'
            ' border-radius: 4px; font-size: 16px; font-weight: bold; }'
            'QPushButton:hover { background: #E07F14; }')
        gen.clicked.connect(self.generate_preview)
        root.addWidget(gen)

    @staticmethod
    def _primary_style():
        return (f'QPushButton {{ background: {BLUE}; color: white; border: none;'
                ' border-radius: 4px; padding: 6px 14px; font-size: 13.5px; }'
                'QPushButton:hover { background: #2569D1; }')

    @staticmethod
    def _combo(items, current):
        c = QComboBox()
        c.addItems(items)
        c.setCurrentText(current)
        c.setStyleSheet('QComboBox { border-radius: 4px; padding: 3px 4px; background: white; }')
        return c

    # ---------- pairs ----------
    def add_pair(self):
        idx = len(self.rows)
        self.core.pairs.append(new_pair())
        row = PairRow(self, idx)
        self.rows.append(row)
        self.pairs_lay.insertWidget(self.pairs_lay.count() - 1, row)
        self._renumber()
        return idx

    def delete_pair(self, idx):
        if idx < 0 or idx >= len(self.rows):
            return
        if len(self.rows) == 1:
            # 组列表永远至少留一行：删最后一行改为清空该行（ying 2026-11-10 定）
            self.core.pairs[idx] = new_pair()
            self._refresh_row(idx)
            self.set_status(f'第 {idx + 1} 组已清空')
            return
        row = self.rows.pop(idx)
        self.core.pairs.pop(idx)
        row.setParent(None)
        row.deleteLater()
        for i, r in enumerate(self.rows):
            r.index = i
        self._renumber()

    def _renumber(self):
        for i, r in enumerate(self.rows):
            r.num.setText(f'第 {i + 1} 组')

    def clear_slot(self, idx, side):
        self.core.pairs[idx][side] = new_slot_data()
        row = self.rows[idx]
        (row.front if side == 'front' else row.back).set_processed(None)
        self.set_status(f'第 {idx + 1} 组{"正面" if side == "front" else "反面"}已清空')

    def swap_pair(self, idx):
        pair = self.core.pairs[idx]
        pair['front'], pair['back'] = pair['back'], pair['front']
        self._refresh_row(idx)
        self.set_status(f'第 {idx + 1} 组正反已互换')

    def _refresh_row(self, idx):
        row = self.rows[idx]
        row.front.set_processed(self.core.pairs[idx]['front']['processed'])
        row.back.set_processed(self.core.pairs[idx]['back']['processed'])

    # ---------- import & processing ----------
    def on_slot_clicked(self, slot):
        for i, r in enumerate(self.rows):
            if slot in (r.front, r.back):
                if self.core.pairs[i][slot.side]['image'] is not None:
                    self.open_mark_dialog(i, slot.side)
                    return
                path, _ = QFileDialog.getOpenFileName(
                    self, '选择图片', self.last_dir,
                    '图片 (*.jpg *.jpeg *.png *.bmp *.webp *.heic *.heif *.tif *.tiff)')
                if path:
                    if not self.import_into_slot(i, slot.side, path):
                        self.open_mark_dialog(i, slot.side)
                return

    def open_mark_dialog(self, idx, side):
        from mark_dialog import MarkDialog
        if self.core.pairs[idx][side]['image'] is None:
            return
        MarkDialog(self, idx, side).exec_()

    def import_into_slot(self, idx, side, path):
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            self.set_status(f'图片打开失败：{e}')
            return False
        self.last_dir = os.path.dirname(path)
        self._sync_core_settings()
        self.core.current_marking_side = side
        corners = self.core.ai_detect_any(img)
        processed = self.core.process_card(img, corners) if corners else None
        data = self.core.pairs[idx][side]
        data['image'] = img
        data['path'] = path
        data['points'] = [tuple(p) for p in corners] if corners else None
        data['processed'] = processed
        (self.rows[idx].front if side == 'front' else self.rows[idx].back).set_processed(processed)
        name = '正面' if side == 'front' else '反面'
        if processed is not None:
            self.set_status(f'第 {idx + 1} 组{name}导入成功，已自动处理')
            self._schedule_ocr(idx, side, processed)
            return True
        self.set_status(f'第 {idx + 1} 组{name}未检测到高精度边框，请换一张或稍后手动标角')
        return False

    def batch_import(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, '批量导入（两张一组）', self.last_dir,
            '图片 (*.jpg *.jpeg *.png *.bmp *.webp *.heic *.heif *.tif *.tiff)')
        if paths:
            self.import_files_as_pairs(paths)

    def import_files_as_pairs(self, paths):
        paths = [p for p in paths if p.lower().endswith(IMG_EXTS)]
        ok = fail = 0
        for k in range(0, len(paths), 2):
            idx = self.add_pair()
            for side, p in zip(('front', 'back'), paths[k:k + 2]):
                if self.import_into_slot(idx, side, p):
                    ok += 1
                else:
                    fail += 1
        self.set_status(f'批量导入完成：成功 {ok} 张，失败 {fail} 张')

    # ---------- drag & drop ----------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if p.lower().endswith(IMG_EXTS)]
        if not paths:
            return
        # fill empty slots first, then new pairs
        rest = list(paths)
        for i, pair in enumerate(self.core.pairs):
            for side in ('front', 'back'):
                if rest and pair[side]['image'] is None:
                    self.import_into_slot(i, side, rest.pop(0))
        if rest:
            self.import_files_as_pairs(rest)

    # ---------- settings ----------
    def _sync_core_settings(self):
        c = self.core
        c.enhance_mode = ENH_MAP.get(self.enh_combo.currentText(), 'color')
        c.wm_enabled = self.wm_check.isChecked()
        c.wm_text = self.wm_text.text()
        c.wm_size = int(self.wm_size.currentText())
        c.wm_color = self.wm_color.currentText()
        c.wm_opacity = int(self.wm_opacity.currentText())
        c.wm_rows = int(self.wm_rows.currentText())
        c.template = self.template_combo.currentText()

    def _save_settings(self):
        self._sync_core_settings()
        data = {'last_dir': self.last_dir, 'enhance': self.enh_combo.currentText(),
                'wm_enabled': self.wm_check.isChecked(), 'wm_text': self.wm_text.text(),
                'wm_size': self.wm_size.currentText(), 'wm_color': self.wm_color.currentText(),
                'wm_opacity': self.wm_opacity.currentText(), 'wm_rows': self.wm_rows.currentText(),
                'template': self.template_combo.currentText()}
        try:
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

    def _load_settings(self):
        try:
            with open(self.settings_path, encoding='utf-8') as f:
                self._saved = json.load(f)
            self.last_dir = self._saved.get('last_dir', '')
        except Exception:
            self._saved = {}

    def _restore_settings_ui(self):
        s = getattr(self, '_saved', {})
        if not s:
            return
        if s.get('enhance') in ENH_MAP:
            self.enh_combo.setCurrentText(s['enhance'])
        self.wm_check.setChecked(bool(s.get('wm_enabled', False)))
        self.wm_text.setText(s.get('wm_text', ''))
        for combo, key in ((self.wm_size, 'wm_size'), (self.wm_color, 'wm_color'),
                           (self.wm_opacity, 'wm_opacity'), (self.wm_rows, 'wm_rows'),
                           (self.template_combo, 'template')):
            v = s.get(key)
            if v is not None and combo.findText(str(v)) >= 0:
                combo.setCurrentText(str(v))
        self._sync_core_settings()

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)

    def set_status(self, text):
        self.status.setText(text)

    def open_license_dialog(self):
        from license_dialog import LicenseDialog
        LicenseDialog(self).exec_()

    # ---------- preview & export ----------
    def generate_preview(self):
        self._sync_core_settings()
        layout = self.core.compute_layout()
        if not layout:
            self.set_status('还没有处理好的图片，先导入证件图')
            return
        dlg = PreviewDialog(self, layout)
        dlg.exec_()


class PageView(QWidget):
    """A4 页交互视图：灰底 + 白页 + 布局里的卡片缩略图，卡片可拖。

    坐标系与 core.compute_layout 一致（A4 页 300dpi 全分辨率）；拖动只限
    页内，松手即写回本窗持有的布局副本——导出/打印按拖后位置渲染，
    关窗不保存拖动结果（与现版第 27 次构建口径一致）。
    """

    def __init__(self, core, layout):
        super().__init__()
        self.core = core
        self.layout = layout            # [[[img, x, y], ...], ...]
        self.idx = 0
        s = core.mm_to_pixel
        self.pw, self.ph = int(core.a4_width * s), int(core.a4_height * s)
        self.scale = 1.0
        self.ox = self.oy = 0.0
        self._thumbs = {}
        self._drag = None               # (card_idx, grab_dx, grab_dy) 视图像素
        self.setMinimumSize(430, 580)
        self.setStyleSheet('border-radius: 5px;')

    # ----- geometry -----
    def _fit(self):
        if self.width() <= 0 or self.height() <= 0:
            return
        self.scale = min((self.width() - 20) / self.pw,
                         (self.height() - 20) / self.ph)
        self.ox = (self.width() - self.pw * self.scale) / 2
        self.oy = (self.height() - self.ph * self.scale) / 2

    def _card_rect(self, img, x, y):
        return (self.ox + x * self.scale, self.oy + y * self.scale,
                img.width * self.scale, img.height * self.scale)

    def _thumb(self, img):
        key = id(img)
        if key not in self._thumbs:
            self._thumbs[key] = QPixmap.fromImage(pil_to_qimage(img))
        return self._thumbs[key]

    # ----- painting -----
    def paintEvent(self, event):
        self._fit()
        qp = QPainter(self)
        qp.setRenderHint(QPainter.SmoothPixmapTransform)
        qp.fillRect(self.rect(), QColor('#808080'))
        px, py = int(self.ox), int(self.oy)
        pw, ph = int(self.pw * self.scale), int(self.ph * self.scale)
        qp.fillRect(px, py, pw, ph, QColor('white'))
        qp.setPen(QPen(QColor('black'), 1))
        qp.drawRect(px, py, pw, ph)
        for img, x, y in self.layout[self.idx]:
            rx, ry, rw, rh = self._card_rect(img, x, y)
            qp.drawPixmap(int(rx), int(ry), int(rw), int(rh), self._thumb(img))
            qp.setPen(QPen(QColor('#999999'), 1))
            qp.drawRect(int(rx), int(ry), int(rw), int(rh))
        qp.end()

    # ----- dragging (helpers kept separate so tests can drive them) -----
    def _press(self, vx, vy):
        cards = self.layout[self.idx]
        for i in range(len(cards) - 1, -1, -1):
            img, x, y = cards[i]
            rx, ry, rw, rh = self._card_rect(img, x, y)
            if rx <= vx <= rx + rw and ry <= vy <= ry + rh:
                self._drag = (i, vx - rx, vy - ry)
                return True
        return False

    def _move(self, vx, vy):
        if not self._drag:
            return
        i, gdx, gdy = self._drag
        img, _x, _y = self.layout[self.idx][i]
        nx = (vx - gdx - self.ox) / self.scale
        ny = (vy - gdy - self.oy) / self.scale
        nx = min(max(nx, 0.0), float(self.pw - img.width))
        ny = min(max(ny, 0.0), float(self.ph - img.height))
        self.layout[self.idx][i] = [img, nx, ny]
        self.update()

    def _release(self):
        self._drag = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press(event.x(), event.y())

    def mouseMoveEvent(self, event):
        self._move(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        self._release()


class PreviewDialog(QDialog):
    """可交互排版预览：五钮齐平（导出裁剪/JPG/单页 PDF/合并 PDF/系统打印），
    页内拖卡；导出一律走系统标准保存框，存完报告位置并给打开文件夹入口。
    排版按主窗底部模板设置（双模板已在主窗切换，此处不另设选择）。"""

    def __init__(self, win, layout):
        super().__init__(win)
        self.win = win
        self.core = win.core
        # 布局副本：拖动只改这份，关窗即弃
        self.layout = [[[img, float(x), float(y)] for img, x, y in page]
                       for page in layout]
        self.idx = 0
        self.setWindowTitle('可交互排版预览（支持拖放）')
        self.resize(720, 940)
        self.setStyleSheet(f'background: {PAGE_BG};')
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        for text, fn in (('导出裁剪', self.export_crops),
                         ('导出为 JPG', self.export_jpg),
                         ('导出单页 PDF', self.export_pdf_single),
                         ('导出合并 PDF', self.export_pdf_merged),
                         ('系统直接打印', self.print_pages)):
            b = QPushButton(text)
            b.setStyleSheet(MainWindow._primary_style())
            b.clicked.connect(fn)
            top.addWidget(b)
        top.addStretch(1)
        lay.addLayout(top)
        # 全能版并入：读字结果导出（口径照全能版，列与格式在 ocr_qt.py）
        ocr_row = QHBoxLayout()
        for text, fn in (('导出信息Excel', self.export_info_excel),
                         ('导出信息TXT', self.export_info_txt)):
            b = QPushButton(text)
            b.setStyleSheet(MainWindow._primary_style())
            b.clicked.connect(fn)
            ocr_row.addWidget(b)
        ocr_row.addStretch(1)
        lay.addLayout(ocr_row)
        self.title = QLabel('')
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet('font-size: 15px; font-weight: bold; color: #333;')
        lay.addWidget(self.title)
        self.view = PageView(self.core, self.layout)
        lay.addWidget(self.view, 1)
        nav = QHBoxLayout()
        prev_b = QPushButton('上一页')
        next_b = QPushButton('下一页')
        for b in (prev_b, next_b):
            b.setStyleSheet(PairRow._btn_style())
        prev_b.clicked.connect(lambda: self.show_page(self.idx - 1))
        next_b.clicked.connect(lambda: self.show_page(self.idx + 1))
        self.page_label = QLabel('')
        self.page_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(prev_b)
        nav.addWidget(self.page_label, 1)
        nav.addWidget(next_b)
        lay.addLayout(nav)
        self.hint = QLabel('拖动卡片可调整位置，保存/打印按新位置')
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet('color: #5B6470; font-size: 12.5px;')
        lay.addWidget(self.hint)
        self.show_page(0)

    def show_page(self, i):
        self.idx = max(0, min(i, len(self.layout) - 1))
        self.view.idx = self.idx
        self.view.update()
        self.title.setText(f'— 第 {self.idx + 1} 页 A4 画布 —')
        self.page_label.setText(f'第 {self.idx + 1} / {len(self.layout)} 页')

    # ----- export plumbing -----
    def _default_dir(self):
        return self.win.last_dir or os.path.expanduser('~')

    def _remember(self, folder):
        self.win.last_dir = folder
        self.win._save_settings()

    def _pages(self):
        """按当前（可能拖过的）布局渲染最终页，含水印。"""
        return self.core.render_pages(self.layout)

    def _ask_folder(self, title):
        d = QFileDialog.getExistingDirectory(self, title, self._default_dir())
        return d or None

    def export_crops(self):
        items = []
        for gi, pair in enumerate(self.core.pairs):
            for side, sname in (('front', '正面'), ('back', '反面')):
                img = pair[side].get('processed')
                if img is not None:
                    items.append((gi + 1, sname, img))
        if not items:
            QMessageBox.warning(self, '提示', '还没有已处理的证件可导出')
            return
        d = self._ask_folder('选择保存文件夹')
        if not d:
            return
        for gi, sname, img in items:
            img.convert('RGB').save(os.path.join(d, f'第{gi}组-{sname}.jpg'),
                                    quality=95)
        self._remember(d)
        self.win.set_status(f'裁剪图已导出到 {d}')
        report_saved(self, d, f'已导出 {len(items)} 张裁剪图')

    def export_jpg(self):
        pages = self._pages()
        if not pages:
            return
        d = self._ask_folder('选择保存文件夹')
        if not d:
            return
        for i, p in enumerate(pages):
            p.save(os.path.join(d, f'证件拼版_{i + 1:02d}.jpg'),
                   quality=95, dpi=(300, 300))
        self._remember(d)
        self.win.set_status(f'JPG 已导出到 {d}')
        report_saved(self, d, f'已导出 {len(pages)} 页 JPG')

    def export_pdf_single(self):
        pages = self._pages()
        if not pages:
            return
        d = self._ask_folder('选择保存文件夹')
        if not d:
            return
        for i, p in enumerate(pages):
            data = self.core._pages_to_pdf_bytes([p])
            with open(os.path.join(d, f'证件拼版_{i + 1:02d}.pdf'), 'wb') as f:
                f.write(data)
        self._remember(d)
        self.win.set_status(f'单页 PDF 已导出到 {d}')
        report_saved(self, d, f'已导出 {len(pages)} 个单页 PDF')

    def export_pdf_merged(self):
        path, _ = QFileDialog.getSaveFileName(
            self, '保存合并 PDF',
            os.path.join(self._default_dir(), '证件拼版.pdf'), 'PDF (*.pdf)')
        if not path:
            return
        if not path.lower().endswith('.pdf'):
            path += '.pdf'
        pages = self._pages()
        if not pages:
            return
        data = self.core._pages_to_pdf_bytes(pages)
        with open(path, 'wb') as f:
            f.write(data)
        self._remember(os.path.dirname(path))
        self.win.set_status(f'合并 PDF 已导出：{path}')
        report_saved(self, os.path.dirname(path),
                     f'已导出合并 PDF：{os.path.basename(path)}')

    # ----- OCR info export (full-edition) -----
    def _info_rows_or_warn(self):
        rows = ocr_qt.pair_info_rows(self.core.pairs)
        if not rows:
            QMessageBox.information(self, '提示',
                                    '还没有识别到身份证信息，请先处理身份证照片')
            return None
        return rows

    def export_info_excel(self):
        rows = self._info_rows_or_warn()
        if rows is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '导出信息Excel',
            os.path.join(self._default_dir(), '证件信息.xlsx'),
            'Excel 工作簿 (*.xlsx)')
        if not path:
            return
        if not path.lower().endswith('.xlsx'):
            path += '.xlsx'
        try:
            ocr_qt.export_info_excel(rows, path)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败：{e}')
            return
        self._remember(os.path.dirname(path))
        self.win.set_status(f'已导出 Excel：{path}')
        report_saved(self, os.path.dirname(path),
                     f'已导出 Excel：{os.path.basename(path)}')

    def export_info_txt(self):
        rows = self._info_rows_or_warn()
        if rows is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '导出信息TXT',
            os.path.join(self._default_dir(), '证件信息.txt'),
            '文本文件 (*.txt)')
        if not path:
            return
        if not path.lower().endswith('.txt'):
            path += '.txt'
        try:
            ocr_qt.export_info_txt(rows, path)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败：{e}')
            return
        self._remember(os.path.dirname(path))
        self.win.set_status(f'已导出 TXT：{path}')
        report_saved(self, os.path.dirname(path),
                     f'已导出 TXT：{os.path.basename(path)}')

    def print_pages(self):
        from PySide2.QtPrintSupport import QPrinter, QPrintDialog
        printer = QPrinter(QPrinter.HighResolution)
        printer.setPageSize(QPrinter.A4)
        printer.setFullPage(True)
        dlg = QPrintDialog(printer, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        pages = self._pages()
        if not pages:
            return
        painter = QPainter(printer)
        rect = printer.pageRect()
        for i, page in enumerate(pages):
            if i:
                printer.newPage()
            painter.drawImage(rect, pil_to_qimage(page))
        painter.end()
        self.win.set_status('打印任务已发送')


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont('Microsoft YaHei', 10))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
