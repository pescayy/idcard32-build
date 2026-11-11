# -*- coding: utf-8 -*-
"""IDCardMuse Qt main window (PySide2). UI layer only; all processing in core.py.
Layout follows IDCardMerge v2.2 (see IDCardMerge-v2.2-复刻对照表.md):
group rows as the main area, settings bar at the bottom, full-width orange
generate button, centered status line. Small radii: buttons 4px, slots 5px.
"""
import os, sys, json, threading
from PySide2.QtCore import Qt, Signal, QUrl, QRectF
from PySide2.QtGui import (QPixmap, QImage, QFont, QPainter, QColor, QPen,
                         QDesktopServices, QPainterPath)
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel,
                               QPushButton, QComboBox, QCheckBox, QLineEdit,
                               QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog,
                               QDialog, QMessageBox, QGridLayout, QMenu, QAction)
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import Core
import ocr_qt

BLUE = '#2E7CF6'
GREEN = '#3DB549'
SLATE = '#5B84AE'
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

# 全局尺寸（2026-11-11 午后五，对版 v2.2 原版的大钮/大菜单）：应用级样式表，
# 主窗与各对话框（执照/标角窗、文件框右键菜单等）统一吃到；控件本地样式表
# 只覆盖它写明的属性，其余尺寸属性仍由此表兜底。
GLOBAL_QSS = (
    'QPushButton { min-height: 34px; }'
    'QComboBox { min-height: 32px; font-size: 14px; }'
    'QLineEdit { min-height: 32px; font-size: 14px; }'
    'QMenu { font-size: 14px; padding: 4px 0; min-width: 220px; }'
    'QMenu::item { padding: 8px 26px; }'
    f'QMenu::item:selected {{ background: {BLUE}; color: white; }}'
)


def btn_qss(bg, hover, size=14, bold=False):
    """彩色大按钮样式：padding 9px 18px、圆角 4px（与全局 min-height 34 叠加）。"""
    weight = ' font-weight: bold;' if bold else ''
    return (f'QPushButton {{ background: {bg}; color: white; border: none;'
            f' border-radius: 4px; padding: 9px 18px; font-size: {size}px;{weight} }}'
            f'QPushButton:hover {{ background: {hover}; }}'
            'QPushButton:disabled { background: #B9C6D6; }')


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
                ' padding: 8px 14px; font-size: 14px; color: #333; }'
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
        # 设置窗五项（对照表 §7）：默认值 = 现行行为，改了才生效
        _sv = getattr(self, '_saved', {})
        self.canvas_w = float(_sv.get('canvas_w', 210))
        self.canvas_h = float(_sv.get('canvas_h', 297))
        self.card_w = float(_sv.get('card_w', 85.6))
        self.card_h = float(_sv.get('card_h', 54.0))
        self.print_scale = float(_sv.get('print_scale', 100))
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
            self._refresh_info_btns()
            self.set_status('文字识别组件不可用，已停用读字（其余功能正常）')

    def _schedule_ocr(self, idx, side, pil_img):
        """处理结果变化后排队读字；同一张图只读一次（与全能版 _ocr_key 口径一致）。"""
        if pil_img is None or self._ocr_dead:
            return
        if not ocr_qt.ocr_available():
            self._ocr_dead = True
            self._refresh_info_btns()
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
            self._refresh_info_btns()
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
        clear_btn = QPushButton('清空数据')
        clear_btn.setStyleSheet(PairRow._btn_style())
        clear_btn.clicked.connect(self.clear_all)
        settings_btn = QPushButton('设置')
        settings_btn.setStyleSheet(self._primary_style())
        settings_btn.clicked.connect(self.open_settings_dialog)
        row1.addWidget(add_btn)
        row1.addWidget(batch_btn)
        row1.addWidget(lic_btn)
        row1.addWidget(clear_btn)
        row1.addWidget(settings_btn)
        row1.addStretch(1)
        row1.addWidget(QLabel('增强：'))
        self.enh_combo = QComboBox()
        self.enh_combo.addItems(list(ENH_MAP.keys()))
        self.enh_combo.setCurrentText('彩色')
        self.enh_combo.setStyleSheet('QComboBox { border-radius: 4px; padding: 5px 8px;'
                                     ' background: white; font-size: 14px; }')
        row1.addWidget(self.enh_combo)
        bl.addLayout(row1)

        row2 = QHBoxLayout()
        self.wm_check = QCheckBox('安全水印')
        row2.addWidget(self.wm_check)
        self.wm_text = QLineEdit()
        self.wm_text.setPlaceholderText('请输入自定义水印内容...')
        self.wm_text.setFixedWidth(170)
        self.wm_text.setStyleSheet('QLineEdit { background: white; border-radius: 4px;'
                                   ' padding: 6px 8px; font-size: 14px; }')
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
        self.template_combo = self._combo(
            ['标准（每组一页）', '紧凑拼版', '单列居中（清单）'], '标准（每组一页）')
        row2.addWidget(self.template_combo)
        bl.addLayout(row2)
        root.addWidget(bar)

        # 信息导出两钮（原在预览窗，按 v2.2 原版钮数挪回主窗、贴生成预览钮）
        info_row = QHBoxLayout()
        info_row.addStretch(1)
        self.info_excel_btn = QPushButton('导出信息 Excel')
        self.info_txt_btn = QPushButton('导出信息 TXT')
        for b in (self.info_excel_btn, self.info_txt_btn):
            b.setStyleSheet(btn_qss(SLATE, '#4A719B'))
        self.info_excel_btn.clicked.connect(self.export_info_excel)
        self.info_txt_btn.clicked.connect(self.export_info_txt)
        info_row.addWidget(self.info_excel_btn)
        info_row.addWidget(self.info_txt_btn)
        root.addLayout(info_row)

        self.status = QLabel('就绪')
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet('color: #5B6470; font-size: 14px;')
        root.addWidget(self.status)

        gen = QPushButton('生成标准 A4 并预览')
        gen.setFixedHeight(48)
        gen.setStyleSheet(
            f'QPushButton {{ background: {ORANGE}; color: white; border: none;'
            ' border-radius: 4px; font-size: 17px; font-weight: bold; }'
            'QPushButton:hover { background: #E07F14; }')
        gen.clicked.connect(self.generate_preview)
        root.addWidget(gen)

    @staticmethod
    def _primary_style():
        return (f'QPushButton {{ background: {BLUE}; color: white; border: none;'
                ' border-radius: 4px; padding: 9px 18px; font-size: 14px; }'
                'QPushButton:hover { background: #2569D1; }')

    @staticmethod
    def _combo(items, current):
        c = QComboBox()
        c.addItems(items)
        c.setCurrentText(current)
        c.setStyleSheet('QComboBox { border-radius: 4px; padding: 5px 8px;'
                        ' background: white; font-size: 14px; }')
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

    def _confirm(self, title, text):
        """原版式确认弹窗（对照表 §2「清空确认」）：默认焦点在「取消」。"""
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        ok_b = box.addButton('确定', QMessageBox.AcceptRole)
        cancel_b = box.addButton('取消', QMessageBox.RejectRole)
        box.setDefaultButton(cancel_b)
        box.exec_()
        return box.clickedButton() == ok_b

    def _row_has_data(self, idx):
        return any(self.core.pairs[idx][s]['image'] is not None
                   for s in ('front', 'back'))

    def clear_all(self):
        """「清空数据」（对照表 §2）：有数据才弹「清空确认」。"""
        if not any(self._row_has_data(i) for i in range(len(self.rows))):
            self.set_status('没有可清空的数据')
            return
        if not self._confirm('清空确认', '确定要清空所有数据吗？清空后不可恢复。'):
            return
        while len(self.rows) > 1:
            row = self.rows.pop()
            self.core.pairs.pop()
            row.setParent(None)
            row.deleteLater()
        self.core.pairs[0] = new_pair()
        self.rows[0].index = 0
        self._refresh_row(0)
        self._renumber()
        self.set_status('所有数据已清空。')

    def delete_pair(self, idx):
        if idx < 0 or idx >= len(self.rows):
            return
        if len(self.rows) == 1:
            # 组列表永远至少留一行：删最后一行改为清空该行（ying 2026-11-10 定）
            if self._row_has_data(idx) and not self._confirm(
                    '清空确认', f'确定要清空第 {idx + 1} 组吗？清空后不可恢复。'):
                return
            self.core.pairs[idx] = new_pair()
            self._refresh_row(idx)
            self.set_status(f'第 {idx + 1} 组已清空')
            return
        if self._row_has_data(idx) and not self._confirm(
                '删除确认', f'确定要删除第 {idx + 1} 组吗？删除后不可恢复。'):
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
        if self.core.pairs[idx][side]['image'] is not None and not self._confirm(
                '清空确认',
                f'确定要清空第 {idx + 1} 组{"正面" if side == "front" else "反面"}吗？'
                '清空后不可恢复。'):
            return
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
        # 设置窗五项（§7）：画布/卡片规格写进 core，默认值与 core 出厂一致
        c.a4_width = float(getattr(self, 'canvas_w', 210))
        c.a4_height = float(getattr(self, 'canvas_h', 297))
        c.card_width = float(getattr(self, 'card_w', 85.6))
        c.card_height = float(getattr(self, 'card_h', 54.0))

    def _save_settings(self):
        self._sync_core_settings()
        data = {'last_dir': self.last_dir, 'enhance': self.enh_combo.currentText(),
                'wm_enabled': self.wm_check.isChecked(), 'wm_text': self.wm_text.text(),
                'wm_size': self.wm_size.currentText(), 'wm_color': self.wm_color.currentText(),
                'wm_opacity': self.wm_opacity.currentText(), 'wm_rows': self.wm_rows.currentText(),
                'template': self.template_combo.currentText(),
                'canvas_w': self.canvas_w, 'canvas_h': self.canvas_h,
                'card_w': self.card_w, 'card_h': self.card_h,
                'print_scale': self.print_scale}
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

    def open_settings_dialog(self):
        from settings_dialog import SettingsDialog
        SettingsDialog(self).exec_()

    # ---------- OCR info export (buttons sit by the generate button) ----------
    def _refresh_info_btns(self):
        dead = getattr(self, '_ocr_dead', False)
        for b in (getattr(self, 'info_excel_btn', None),
                  getattr(self, 'info_txt_btn', None)):
            if b is not None:
                b.setEnabled(not dead)

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
            os.path.join(self.last_dir or os.path.expanduser('~'), '证件信息.xlsx'),
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
        self.last_dir = os.path.dirname(path)
        self._save_settings()
        self.set_status(f'已导出 Excel：{path}')
        report_saved(self, os.path.dirname(path),
                     f'已导出 Excel：{os.path.basename(path)}')

    def export_info_txt(self):
        rows = self._info_rows_or_warn()
        if rows is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, '导出信息TXT',
            os.path.join(self.last_dir or os.path.expanduser('~'), '证件信息.txt'),
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
        self.last_dir = os.path.dirname(path)
        self._save_settings()
        self.set_status(f'已导出 TXT：{path}')
        report_saved(self, os.path.dirname(path),
                     f'已导出 TXT：{os.path.basename(path)}')

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
        self.setMinimumSize(430, 500)
        self.setStyleSheet('border-radius: 5px;')

    # ----- geometry -----
    def _fit(self):
        if self.width() <= 0 or self.height() <= 0:
            return
        self.scale = min((self.width() - 16) / self.pw,
                         (self.height() - 16) / self.ph)
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
        # 页外底色用近白（与窗体底色同族）：原版画布区是亮底、页面适窗
        # 铺开看着舒展，中灰大边显得缩（ying 2026-11-11 拿原版照片指出）。
        qp.fillRect(self.rect(), QColor('#EDF0F3'))
        px, py = int(self.ox), int(self.oy)
        pw, ph = int(self.pw * self.scale), int(self.ph * self.scale)
        qp.fillRect(px, py, pw, ph, QColor('white'))
        qp.setPen(QPen(QColor('#AEB6C2'), 1))
        qp.drawRect(px, py, pw, ph)
        for img, x, y in self.layout[self.idx]:
            rx, ry, rw, rh = self._card_rect(img, x, y)
            # 圆角仅显示层：半径按真实身份证圆角比例（约卡宽 3.5%）裁剪，
            # 导出/打印仍走 core.render_pages 的直角原图，不受影响。
            rect = QRectF(rx, ry, rw, rh)
            path = QPainterPath()
            path.addRoundedRect(rect, rw * 0.035, rw * 0.035)
            thumb = self._thumb(img)
            qp.save()
            qp.setClipPath(path)
            qp.drawPixmap(rect, thumb, QRectF(thumb.rect()))
            qp.restore()
            qp.setPen(QPen(QColor('#999999'), 1))
            qp.drawPath(path)
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
    """可交互排版预览：顶排照 IDCardMerge v2.2 原版——单排 5 个长钮等宽
    拉长撑满窗宽（导出裁剪 / 导出为 JPG / 导出单页 PDF / 导出合并 PDF /
    系统直接打印），页内拖卡；信息 Excel/TXT 不在此窗（已挪主窗生成预览
    钮旁）。导出一律走系统标准保存框，存完报告位置并给打开文件夹入口。
    排版按主窗底部模板设置（三模板已在主窗切换，此处不另设选择）。"""

    def __init__(self, win, layout):
        super().__init__(win)
        self.win = win
        self.core = win.core
        # 布局副本：拖动只改这份，关窗即弃
        self.layout = [[[img, float(x), float(y)] for img, x, y in page]
                       for page in layout]
        self.idx = 0
        # 每张布局卡对应的 (组序, 面)：compute_layout 按组序、正反序摊平，
        # 与布局顺序严格一致；供原版口径导出命名取源图基底（对照表 §5）。
        self._src_map = []
        _flat = [(gi, side) for gi, pair in enumerate(self.core.pairs)
                 for side in ('front', 'back')
                 if pair[side].get('processed') is not None]
        if len(_flat) == sum(len(p) for p in self.layout):
            _it = iter(_flat)
            self._src_map = [[next(_it) for _ in page] for page in self.layout]
        self.setWindowTitle('可交互排版预览（支持拖放）')
        self.resize(1180, 760)
        self.setStyleSheet(f'background: {PAGE_BG};')
        lay = QVBoxLayout(self)
        # 原版式单排长钮：5 钮等宽拉长撑满窗宽——裁剪素色（剪刀语义）、
        # JPG/单页 PDF 蓝、合并 PDF 绿、系统直接打印同裁剪素色（浅底深字、
        # 不作主钮——原版 v2.2 如此，2026-11-11 午后小修更正）。
        top = QHBoxLayout()
        self.top_buttons = {}
        plain_qss = (
            f'QPushButton {{ background: {BAR_BG}; color: #333; border: none;'
            ' border-radius: 4px; padding: 9px 18px; font-size: 14px; }'
            'QPushButton:hover { background: #E2E7EE; }')
        crop_b = QPushButton('✂ 导出裁剪')
        crop_b.setStyleSheet(plain_qss)
        crop_b.clicked.connect(self.export_crops)
        top.addWidget(crop_b, 1)
        self.top_buttons['导出裁剪'] = crop_b
        for text, fn, style in (
                ('导出为 JPG', self.export_jpg, btn_qss(BLUE, '#2569D1')),
                ('导出单页 PDF', self.export_pdf_single, btn_qss(BLUE, '#2569D1')),
                ('导出合并 PDF', self.export_pdf_merged, btn_qss(GREEN, '#2F9E3E')),
                ('系统直接打印', self.print_pages, plain_qss)):
            b = QPushButton(text)
            b.setStyleSheet(style)
            b.clicked.connect(fn)
            top.addWidget(b, 1)
            self.top_buttons[text] = b
        self.print_btn = self.top_buttons['系统直接打印']
        lay.addLayout(top)
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
        self.page_label.setStyleSheet('color: #5B6470; font-size: 12.5px;')
        nav.addWidget(prev_b)
        nav.addWidget(self.page_label, 1)
        nav.addWidget(next_b)
        lay.addLayout(nav)
        self.show_page(0)

    def show_page(self, i):
        self.idx = max(0, min(i, len(self.layout) - 1))
        self.view.idx = self.idx
        self.view.update()
        self.title.setText(f'— 第 {self.idx + 1} 页 A4 画布 —')
        # 底部文案照对照表 §5 原文一行：页码 + 拖动提示合并
        self.page_label.setText(
            f'第 {self.idx + 1}/{len(self.layout)} 页 · '
            '拖动卡片可调整位置，保存/打印按新位置')

    # ----- export plumbing -----
    def _stem(self, gi, side):
        """源图名基底（原版命名 %s）；无名时以组号兜底。"""
        path = self.core.pairs[gi][side].get('path')
        if path:
            stem = os.path.splitext(os.path.basename(path))[0].strip()
            if stem:
                return stem
        return str(gi + 1)

    def _page_base(self, i):
        if self._src_map and self._src_map[i]:
            gi, side = self._src_map[i][0]
            return self._stem(gi, side)
        return str(i + 1)

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
        # 命名照原版 §5：%s_Crop_P{页}_{页内序号}.jpg，%s 为该卡源图基底
        items = []
        if self._src_map:
            for pno, page in enumerate(self._src_map):
                for sno, (gi, side) in enumerate(page):
                    img = self.core.pairs[gi][side].get('processed')
                    if img is not None:
                        items.append(
                            (f'{self._stem(gi, side)}_Crop_P{pno + 1}_{sno + 1}.jpg',
                             img))
        else:
            for gi, pair in enumerate(self.core.pairs):
                k = 0
                for side in ('front', 'back'):
                    img = pair[side].get('processed')
                    if img is not None:
                        k += 1
                        items.append(
                            (f'{self._stem(gi, side)}_Crop_P{gi + 1}_{k}.jpg', img))
        if not items:
            QMessageBox.warning(self, '提示', '还没有已处理的证件可导出')
            return
        d = self._ask_folder('选择保存文件夹')
        if not d:
            return
        for name, img in items:
            img.convert('RGB').save(os.path.join(d, name), quality=95)
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
            p.save(os.path.join(d, f'{self._page_base(i)}_IDCard_{i + 1}.jpg'),
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
            with open(os.path.join(
                    d, f'{self._page_base(i)}_IDCard_{i + 1}.pdf'), 'wb') as f:
                f.write(data)
        self._remember(d)
        self.win.set_status(f'单页 PDF 已导出到 {d}')
        report_saved(self, d, f'已导出 {len(pages)} 个单页 PDF')

    def export_pdf_merged(self):
        path, _ = QFileDialog.getSaveFileName(
            self, '保存合并 PDF',
            os.path.join(self._default_dir(),
                         f'{self._page_base(0)}_IDCard.pdf'), 'PDF (*.pdf)')
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

    @staticmethod
    def _calibrated(page, pct):
        """物理尺寸微调（对照表 §6/§7）：整页绕中心按 pct% 缩放贴回同尺寸
        白页；只供打印路径调用，导出文件不走这里。"""
        f = pct / 100.0
        w, h = page.size
        nw, nh = max(1, round(w * f)), max(1, round(h * f))
        out = Image.new('RGB', (w, h), (255, 255, 255))
        out.paste(page.resize((nw, nh), Image.Resampling.LANCZOS),
                  ((w - nw) // 2, (h - nh) // 2))
        return out

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
        scale = float(getattr(self.win, 'print_scale', 100) or 100)
        if scale != 100.0:
            pages = [self._calibrated(p, scale) for p in pages]
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
    app.setStyleSheet(GLOBAL_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
