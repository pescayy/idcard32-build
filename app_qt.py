# -*- coding: utf-8 -*-
"""IDCardMuse Qt main window (PySide2). UI layer only; all processing in core.py.
Layout follows IDCardMerge v2.2 (see IDCardMerge-v2.2-复刻对照表.md):
group rows as the main area, settings bar at the bottom, full-width orange
generate button, centered status line. Small radii: buttons 4px, slots 5px.
"""
import os, sys, json, threading
from PySide2.QtCore import Qt, Signal, QUrl, QRectF, QTimer
from PySide2.QtGui import (QPixmap, QImage, QFont, QPainter, QColor, QPen,
                         QDesktopServices, QPainterPath, QGuiApplication)
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
ENH_MAP = {'原图': 'none', '彩色': 'color', '清晰': 'clarity', '黑白': 'bw',
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
        processed = self.core.process_card(img, corners, points_ordered=True) if corners else None
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
                'enhance_id_default': self.enh_combo.currentText() == '清晰',
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
        elif s.get('enhance_id_default'):
            # 清晰档粘性标记（出厂默认 False，见 idcard_settings.json）：
            # enhance 键缺失时凭它恢复上次的清晰档选择
            self.enh_combo.setCurrentText('清晰')
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
    """A4 页交互视图：亮底 + 白页 + 布局里的卡片缩略图，卡片可拖。

    几何照原版 v2.2 本体真数（PREVIEW-GEOMETRY.md，本体常量＋照片互证）：
    页显示宽定版为对话框宽 × 0.567（原版 595/1050），水平居中（左右各
    ≈21.65%），上边距 0——页顶与画布顶齐平；页高按 A4 比例（宽×297/210）
    自然溢出画布，由外层 QScrollArea 纵向滚动查看（原版即如此，照片实证
    页下边被截断），不为塞下整页缩小页宽占比。

    坐标系与 core.compute_layout 一致（A4 页 300dpi 全分辨率）；拖动写回
    本窗持有的布局副本——导出/打印按拖后位置渲染，关窗不保存拖动结果
    （与现版第 27 次构建口径一致）。本控件是滚动区的
    内容控件，控件坐标即内容坐标，滚动不改变 _press/_move 的坐标语义。

    自由排版（照原版 v2.2 语义复刻，PREVIEW-GEOMETRY.md §七/§八）：
    默认锁定——卡片不可选中、不可拖动，点按无反应；经对话框第六钮
    开启后，先点选一张卡（冷蓝描边为选中态）才能拖被选中的那张，点
    另一张改选另一张；拖动任意像素自由放——无吸附、无页界钳制，卡可
    拖出页外，坐标照存、导出/打印照吃同一套坐标；拖动中显示过被拖卡
    中心的整页横/竖参考线，松手即隐藏（纯显示）。「全选图片」勾选时
    本页全部卡视为已选中、可直接拖任意一张，取消则塌回当前单张；
    退出自由排版连带清选中、收全选。开关状态每次开窗新建、不落盘。

    缩放层（2026-11-11 用户拍板的行业式补齐，非原版口径、与对版清单
    分开记）：100% 即上述原版几何（页宽 = 对话框宽 × 0.567）；Ctrl+滚
    轮以光标点为锚在 25%–300% 间缩放，另有「适页」档把整页塞进视口。
    缩放只改显示倍率——页宽基准、边距、拖动坐标语义一律不变，导出与
    打印走 core 渲染、不受影响；每次开窗新建本控件，缩放不持久化。
    """

    PAGE_W_RATIO = 0.567  # 页宽/对话框宽（原版 595/1050，本体常量）
    # 缩放层参数：100% = 上行原版口径；倍率范围与步进照行业通行
    ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.25, 3.0, 1.25
    zoom_changed = Signal(float)

    def __init__(self, core, layout):
        super().__init__()
        self.core = core
        self.layout = layout            # [[[img, x, y], ...], ...]
        self.idx = 0
        s = core.mm_to_pixel
        self.pw, self.ph = int(core.a4_width * s), int(core.a4_height * s)
        self.ref_w = 1050.0             # 参照宽 = 对话框宽，对话框 resize 时同步
        self.scale = 1.0
        self.ox = 0.0
        self.oy = 0.0                   # 恒 0：页上边距 0
        self._thumbs = {}
        self._drag = None               # (card_idx, grab_dx, grab_dy) 内容像素
        self.free_mode = False          # 自由排版开关：默认锁定（原版语义）
        self.sel = None                 # 当前选中卡（本页序号）；None = 未选
        self.all_sel = False            # 「全选图片」勾选态：全卡视为已选中
        self.zoom = 1.0                 # 1.0 = 原版口径；开窗新建、不持久化
        self._scroll_area = None        # 对话框建好滚动区后 attach_scroll 注入
        self._pending_anchor = None     # 缩放后待落位的 (滚动x, 滚动y)
        self.setMinimumSize(200, 120)
        self.setStyleSheet('border-radius: 5px;')
        self._recalc()

    # ----- geometry -----
    def set_ref_width(self, w):
        """对话框宽度变化时同步参照宽，页宽占比（0.567）随之恒定。"""
        if w and abs(float(w) - self.ref_w) > 0.5:
            self.ref_w = float(w)
            self._recalc()

    def _recalc(self):
        # 页宽只由参照宽与缩放倍率定（× 0.567 × zoom），不随控件高度
        # 变；zoom=1 即原版口径（整页适高只作缩放的「适页」档存在）。
        page_w = self.ref_w * self.PAGE_W_RATIO * self.zoom
        self.scale = page_w / self.pw
        self.oy = 0.0
        self.ox = (self.width() - page_w) / 2 if self.width() > 0 else 0.0
        # 内容尺寸 = 页全尺寸（A4 比例）：宽/高超出视口的部分由滚动区
        # 滚动（纵向原版即如此；横向仅在放大到页宽超视口时出现）。
        self.setMinimumWidth(int(round(page_w)))
        self.setMinimumHeight(int(round(self.ph * self.scale)))
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        page_w = self.ref_w * self.PAGE_W_RATIO * self.zoom
        self.ox = (self.width() - page_w) / 2 if self.width() > 0 else 0.0

    # ----- zoom（行业式补齐层：100% = 原版口径，见类 docstring） -----
    def attach_scroll(self, scroll):
        self._scroll_area = scroll

    def set_zoom(self, z, anchor=None):
        """设缩放倍率（25%–300%）。anchor 为控件（内容）坐标：缩放
        前后，该点所指的页内容在视口中的位置保持不变（页面不跳位）；
        不给 anchor 时以视口中心为锚。滚动偏移待布局落定后落位。"""
        z = min(max(float(z), self.ZOOM_MIN), self.ZOOM_MAX)
        if abs(z - self.zoom) < 1e-9:
            return
        if self._pending_anchor is not None:
            self._apply_pending_anchor()
        sa = self._scroll_area
        if sa is not None and sa.viewport().width() > 0:
            vp = sa.viewport()
            hb, vb = sa.horizontalScrollBar(), sa.verticalScrollBar()
            if anchor is None:
                anchor = (hb.value() + vp.width() / 2.0,
                          vb.value() + vp.height() / 2.0)
            pgx = (anchor[0] - self.ox) / self.scale
            pgy = (anchor[1] - self.oy) / self.scale
            vx, vy = anchor[0] - hb.value(), anchor[1] - vb.value()
            self.zoom = z
            self._recalc()
            # 目标几何按确定式算，与滚动区（resizable＋最小尺寸）最终
            # 落定的内容尺寸一致，不依赖中间帧的控件宽高。
            page_w = self.ref_w * self.PAGE_W_RATIO * z
            content_w = max(float(vp.width()), page_w)
            content_h = max(float(vp.height()), self.ph * self.scale)
            ox_new = (content_w - page_w) / 2.0
            tx = ox_new + pgx * self.scale - vx
            ty = pgy * self.scale - vy          # oy 恒 0
            self._pending_anchor = (
                min(max(tx, 0.0), max(0.0, content_w - vp.width())),
                min(max(ty, 0.0), max(0.0, content_h - vp.height())))
            QTimer.singleShot(0, self._apply_pending_anchor)
        else:
            self.zoom = z
            self._recalc()
        self.zoom_changed.emit(self.zoom)

    def _apply_pending_anchor(self):
        if self._pending_anchor is None or self._scroll_area is None:
            return
        tx, ty = self._pending_anchor
        self._pending_anchor = None
        self._scroll_area.horizontalScrollBar().setValue(int(round(tx)))
        self._scroll_area.verticalScrollBar().setValue(int(round(ty)))

    def zoom_step(self, steps):
        self.set_zoom(self.zoom * (self.ZOOM_STEP ** steps))

    def zoom_reset(self):
        self.set_zoom(1.0)

    def zoom_fit(self):
        """适页：整页恰好塞进视口（按高适配），水平居中、无需滚动。"""
        sa = self._scroll_area
        if sa is None or sa.viewport().height() <= 0:
            return
        base = self.ref_w * self.PAGE_W_RATIO / self.pw   # 100% 时 scale
        self.set_zoom(sa.viewport().height() / (self.ph * base))

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
        if self.free_mode:
            # 选中态：既有冷蓝色描边（克制）。全选时全卡描边，否则只描
            # 当前选中卡——与原版 +0x89 选中位驱动逐卡 Refresh 同义。
            if self.all_sel:
                marked = range(len(self.layout[self.idx]))
            elif self.sel is not None:
                marked = (self.sel,)
            else:
                marked = ()
            for i in marked:
                img, x, y = self.layout[self.idx][i]
                rx, ry, rw, rh = self._card_rect(img, x, y)
                spath = QPainterPath()
                spath.addRoundedRect(QRectF(rx, ry, rw, rh),
                                     rw * 0.035, rw * 0.035)
                qp.setPen(QPen(QColor(BLUE), 2))
                qp.drawPath(spath)
            if self._drag:
                # 拖动参考线：过被拖卡中心的整页横/竖线（原版 func23），
                # 纯显示、不回馈坐标；松手（_release 清 _drag）即隐藏。
                img, x, y = self.layout[self.idx][self._drag[0]]
                cx = self.ox + (x + img.width / 2.0) * self.scale
                cy = self.oy + (y + img.height / 2.0) * self.scale
                guide = QColor(BLUE)
                guide.setAlpha(150)
                qp.setPen(QPen(guide, 1))
                qp.drawLine(int(px), int(cy), int(px + pw), int(cy))
                qp.drawLine(int(cx), int(py), int(cx), int(py + ph))
        qp.end()

    # ----- free layout & dragging (helpers kept separate for tests) -----
    def set_free_mode(self, on):
        """自由排版开关（原版 func27 语义）：开＝在模板现位置上原地解锁；
        关＝原地锁定，且连带清当前选中、收「全选图片」态（原版退出时
        SetChecked(false) 把多选塌回，净效果即全清）。不重排、不落盘。"""
        self.free_mode = bool(on)
        if not self.free_mode:
            self._drag = None
            self.sel = None
            self.all_sel = False
        self.update()

    def _press(self, vx, vy):
        # 锁定态：点按无反应（原版关态 Tapped/Dragged 双检许可位直接返回）
        if not self.free_mode:
            return False
        cards = self.layout[self.idx]
        for i in range(len(cards) - 1, -1, -1):
            img, x, y = cards[i]
            rx, ry, rw, rh = self._card_rect(img, x, y)
            if rx <= vx <= rx + rw and ry <= vy <= ry + rh:
                # 先点选、再拖动：按下即选中该卡并抓起它；点另一张即改选
                # 另一张。全选态下任意卡本就已选中，同路径直接拖。
                self.sel = i
                self._drag = (i, vx - rx, vy - ry)
                self.update()
                return True
        return False

    def _move(self, vx, vy):
        if not self._drag:
            return
        i, gdx, gdy = self._drag
        img, _x, _y = self.layout[self.idx][i]
        # 任意像素自由放：无吸附、无页界钳制（原版 Dragged 原样累加增量，
        # 卡可拖出页外、坐标照存）——旧版自加的页内 clamp 已按原版撤除。
        nx = (vx - gdx - self.ox) / self.scale
        ny = (vy - gdy - self.oy) / self.scale
        self.layout[self.idx][i] = [img, nx, ny]
        self.update()

    def _release(self):
        self._drag = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press(event.x(), event.y())

    def mouseMoveEvent(self, event):
        self._move(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        self._release()

    def wheelEvent(self, event):
        # 行业式：普通滚轮平移（不接受事件、传播给滚动区，保持原状）；
        # Ctrl+滚轮以光标点为锚缩放。
        if event.modifiers() & Qt.ControlModifier:
            steps = event.angleDelta().y() / 120.0
            if steps:
                self.set_zoom(self.zoom * (self.ZOOM_STEP ** steps),
                              anchor=(float(event.x()), float(event.y())))
            event.accept()
            return
        super().wheelEvent(event)


class PreviewDialog(QDialog):
    """可交互排版预览：顶排照 IDCardMerge v2.2 原版——单排 6 个长钮等宽
    拉长撑满窗宽（导出裁剪 / 导出为 JPG / 导出单页 PDF / 导出合并 PDF /
    系统直接打印 / 🛠 开启自由排版），自由排版语义照原版复刻（默认锁定、
    开启后先点选再拖、无页界钳制、拖动参考线、文案双态恒橙，详见
    PageView docstring 与 PREVIEW-GEOMETRY.md §七）；标题行右端有
    「全选图片」勾选（原版置于「铺满 A4 全页」钮行右端，我方无该钮，
    置此等价右侧位，§八）。信息 Excel/TXT 不在此窗（已挪主窗生成预览
    钮旁）。导出一律走系统标准保存框，存完报告位置并给打开文件夹入口。
    排版按主窗底部模板设置（三模板已在主窗切换，此处不另设选择）。
    几何照原版真数（PREVIEW-GEOMETRY.md）：窗 1050×650 居中、页宽 =
    窗宽 × 0.567、上边距 0、页高溢出纵向滚动——此前近全屏＋左栏版已撤，
    多页导航只用底部上一页/下一页钮。底部行右侧另有缩放簇（−／百分比／
    ＋／适页），系 2026-11-11 用户拍板的行业式补齐，非原版口径、与对版
    清单分开记（见 PORT-STATUS）。"""

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
        # 尺寸照原版 v2.2 本体真数：1050×650 居中开窗（PREVIEW-GEOMETRY.md
        # §一，Resize 常量实锤；亦合「预览页比主窗小一号」）。窗可缩放，
        # PageView 的页宽占比由对话框 resize 时同步参照宽恒定保持。
        self.resize(1050, 650)
        scr = QGuiApplication.primaryScreen()
        if scr is not None and QGuiApplication.platformName() != 'offscreen':
            geo = scr.availableGeometry()
            self.move(geo.x() + (geo.width() - self.width()) // 2,
                      geo.y() + (geo.height() - self.height()) // 2)
        self.setStyleSheet(f'background: {PAGE_BG};')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(6)
        # 原版式单排长钮：6 钮等宽拉长撑满窗宽——裁剪素色（剪刀语义）、
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
        # 第六钮「开启自由排版」：原版恒 warning 橙（Importance=4，§六），
        # 与既有五钮同高同排版规则、贴右；开启后文案变「🛑 锁定当前
        # 排版」，颜色不变（原版切换不改 Importance）。
        self.free_btn = QPushButton('🛠 开启自由排版')
        self.free_btn.setStyleSheet(btn_qss(ORANGE, '#E07F14'))
        self.free_btn.clicked.connect(self.toggle_free)
        top.addWidget(self.free_btn, 1)
        self.top_buttons['开启自由排版'] = self.free_btn
        # 钮带高度照真数核对（PREVIEW-GEOMETRY.md §二：原版钮带 ≈33、
        # 含间隙工具区 ≈42）：全局大钮口径会把本窗六钮撑到 52，仅本窗
        # 覆写回紧凑高（实测 ≈37），文案/数目/配色一律不动。
        for b in self.top_buttons.values():
            b.setStyleSheet(b.styleSheet() +
                            'QPushButton { min-height: 25px;'
                            ' padding: 6px 18px; }')
        # 六钮严格同高：第六钮的 emoji 字形度量可能比纯文字钮高 1px
        # （环境字体而定），以五钮的 sizeHint 最大值为准把六钮钉同高，
        # 数值随平台字体自适应、不写死。
        _h = max(b.sizeHint().height()
                 for k, b in self.top_buttons.items() if k != '开启自由排版')
        for b in self.top_buttons.values():
            b.setFixedHeight(_h)
        lay.addLayout(top)
        title_row = QHBoxLayout()
        self.title = QLabel('')
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet('font-size: 15px; font-weight: bold; color: #333;')
        title_row.addWidget(self.title, 1)
        # 「全选图片」勾选（原版 §八）：原版位于「铺满 A4 全页」钮行最
        # 右端；我方无该钮，置于标题行右端这一等价的预览控制区右侧位。
        self.all_sel_chk = QCheckBox('全选图片')
        self.all_sel_chk.setStyleSheet('color: #333; font-size: 13px;')
        self.all_sel_chk.stateChanged.connect(self._on_all_sel)
        title_row.addWidget(self.all_sel_chk)
        lay.addLayout(title_row)
        self.view = PageView(self.core, self.layout)
        self.view.set_ref_width(self.width())
        # 页高按 A4 比例溢出画布 → 纵向滚动（原版式，照片实证页下边被
        # 截断）：QScrollArea 常规实现，滚轮与竖直拖拽条均可用；左栏
        # 缩略图已撤，多页导航只用底部上一页/下一页钮。
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.view)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        # 横向条按需：≤100%/适页时内容不溢出、不出现；放大到页宽超
        # 视口时出现以便横向查看（缩放层所需，原版口径下恒不触发）。
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setStyleSheet('background: #EDF0F3;')
        self.view.attach_scroll(self.scroll)
        lay.addWidget(self.scroll, 1)
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
        # 缩放簇（行业式补齐，用户 2026-11-11 拍板，非原版口径）：素色
        # 与上下页钮同族、紧凑定宽，与中间文案分置两端——文案仍居中、
        # 一行原文不变，不被挤压。百分比钮点击回 100%。
        zoom_box = QHBoxLayout()
        zoom_box.setSpacing(4)
        self.zoom_out_b = QPushButton('−')
        self.zoom_pct_b = QPushButton('100%')
        self.zoom_in_b = QPushButton('+')
        self.zoom_fit_b = QPushButton('适页')
        for b, w in ((self.zoom_out_b, 34), (self.zoom_pct_b, 58),
                     (self.zoom_in_b, 34), (self.zoom_fit_b, 56)):
            b.setStyleSheet(PairRow._btn_style() +
                            'QPushButton { padding: 8px 4px; }')
            b.setFixedWidth(w)
            zoom_box.addWidget(b)
        self.zoom_out_b.clicked.connect(lambda: self.view.zoom_step(-1))
        self.zoom_in_b.clicked.connect(lambda: self.view.zoom_step(1))
        self.zoom_pct_b.clicked.connect(self.view.zoom_reset)
        self.zoom_fit_b.clicked.connect(self.view.zoom_fit)
        self.view.zoom_changed.connect(self._on_zoom_changed)
        nav.addLayout(zoom_box)
        nav.addWidget(next_b)
        lay.addLayout(nav)
        self.show_page(0)

    def toggle_free(self):
        """自由排版开关（原版 func27）：文案双态、恒橙；退出时连带把
        「全选图片」勾选收掉（原版 SetChecked(false) 联动）。"""
        on = not self.view.free_mode
        self.view.set_free_mode(on)
        self.free_btn.setText('🛑 锁定当前排版' if on else '🛠 开启自由排版')
        if not on and self.all_sel_chk.isChecked():
            self.all_sel_chk.blockSignals(True)
            self.all_sel_chk.setChecked(False)
            self.all_sel_chk.blockSignals(False)

    def _on_all_sel(self, state):
        # 勾选＝全卡已选中（可直接拖任意一张）；取消＝塌回当前单张
        # （view.sel 本就只记当前卡，语义自动合拢）。
        self.view.all_sel = (state == Qt.Checked)
        self.view.update()

    def show_page(self, i):
        self.idx = max(0, min(i, len(self.layout) - 1))
        self.view.idx = self.idx
        # 选中是页内 affordance：换页清当前选中，避免在新页错描另一张
        # （全选态跨页仍按页生效，与原版逐卡置位同义）。
        self.view.sel = None
        self.view._drag = None
        self.view.update()
        self.title.setText(f'— 第 {self.idx + 1} 页 A4 画布 —')
        # 底部文案照对照表 §5 原文一行：页码 + 拖动提示合并
        self.page_label.setText(
            f'第 {self.idx + 1}/{len(self.layout)} 页 · '
            '拖动卡片可调整位置，保存/打印按新位置')

    def _on_zoom_changed(self, z):
        self.zoom_pct_b.setText(f'{int(round(z * 100))}%')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 页宽占比锚定对话框宽：缩放窗体时同步参照宽（0.567 恒定）
        if getattr(self, 'view', None) is not None:
            self.view.set_ref_width(self.width())

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
