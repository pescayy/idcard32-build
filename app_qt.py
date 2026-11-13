# -*- coding: utf-8 -*-
"""IDCardMuse Qt main window (PySide2). UI layer only; all processing in core.py.
Layout follows IDCardMerge v2.2 (see IDCardMerge-v2.2-复刻对照表.md):
group rows as the main area, settings bar at the bottom, full-width orange
generate button, centered status line. Small radii: buttons 4px, slots 5px.
"""
import os, sys, json, threading, queue
import numpy as np
from PySide2.QtCore import Qt, Signal, QUrl, QRectF, QTimer, QSize, QEvent
from PySide2.QtGui import (QPixmap, QImage, QFont, QPainter, QColor, QPen,
                         QDesktopServices, QPainterPath, QGuiApplication,
                         QIcon, QFontDatabase)
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel,
                               QPushButton, QComboBox, QCheckBox, QLineEdit,
                               QVBoxLayout, QHBoxLayout, QScrollArea, QFileDialog,
                               QDialog, QMessageBox, QGridLayout, QMenu, QAction,
                               QRadioButton, QButtonGroup)
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import Core, _apply_watermark
import ocr_qt

# 色板照 Merge v2.2 实测（MERGE-UI-MEASURE.md 色板总表 + 预览窗截图采样）：
# 蓝 #296FF6、橙 #FF9800、绿 #43F436（合并 PDF 与自由排版选中描边同色）、
# 灰钮 #F5F5F5 无边、下拉 #F3F3F3、正文墨 #565656、分隔线 #E3E3E3、
# 预览画布 #E6E6E6（09-preview.png 多点采样）。
BLUE = '#296FF6'
BLUE_HOVER = '#1E5FE1'
GREEN = '#43F436'
SLATE = '#5B84AE'
ORANGE = '#FF9800'
ORANGE_HOVER = '#FB8C00'
PAGE_BG = '#F5F6F8'
BAR_BG = '#EEF0F3'
SLOT_BG = '#FAFBFC'
INK = '#565656'
LINE = '#E3E3E3'
GRAY_BTN = '#F5F5F5'
COMBO_BG = '#F3F3F3'
CANVAS_BG = '#E6E6E6'
ENH_MAP = {'原图': 'none', '彩色': 'color', '清晰': 'clarity', '黑白': 'bw',
           '文档黑白': 'docbw', '文档彩色': 'doccolor'}
WM_COLORS = ['黑色', '蓝色', '红色', '灰色']
# 参数行五下拉（MERGE-FULL-SPEC 补取专节，选项/默认照 exe 原文）：
# 安全水印预设的逗号为半角、下划线为 ASCII（预设二 6 个、预设三 4+4 个）。
WM_PRESETS = ['无水印',
              '本复印件仅供______办理业务使用，他用无效',
              '本复印件仅供____年____月办理业务使用，他用无效',
              '自定义']
WM_SIZE_ITEMS = ['小 (36)', '中 (56)', '大 (72)']
# 界面字号档 → core.wm_size（core 口径 font_px = size × 300dpi 系数，
# 与 Merge 的 36/56/72 单位不同，UI 层做近似映射，默认档与旧默认 7 对齐）
WM_SIZE_MAP = {'小 (36)': 7, '中 (56)': 11, '大 (72)': 14}
WM_COLOR_ITEMS = ['黑色 （默认）', '蓝色', '红色', '灰色']
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.heic', '.heif', '.tif', '.tiff')

ICONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons')
_icon_cache = {}


def icon(name):
    """icons/ 下的 SVG 小图标（Material 一脉），挂 QIcon 用；缺失返回空图标。"""
    if name not in _icon_cache:
        _icon_cache[name] = QIcon(os.path.join(ICONS_DIR, name + '.svg'))
    return _icon_cache[name]

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
    """一组里的一列（照 Merge 实拍）：上＝导入灰钮 212×36（文件夹图标＋
    「导入证件 N」），下＝缩略图区 212×133（导入后出图、双击进微调窗）。
    空态只见钮、下方留白，组高恒定。钮点击＝导入/换图；缩略图双击＝微调。"""
    import_clicked = Signal(object)
    mark_clicked = Signal(object)

    def __init__(self, side, num):
        super().__init__()
        self.side = side
        self.setFixedWidth(212)
        self.setStyleSheet('QFrame { background: white; border: none; }')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)
        self.button = QPushButton(f'导入证件 {num}')
        self.button.setIcon(icon('folder_dark'))
        self.button.setIconSize(QSize(17, 17))
        self.button.setFixedSize(212, 36)
        self.button.setStyleSheet(
            f'QPushButton {{ background: {GRAY_BTN}; color: {INK}; border: none;'
            ' border-radius: 4px; font-size: 14px; min-height: 36px; }'
            'QPushButton:hover { background: #ECECEC; }')
        self.button.clicked.connect(lambda: self.import_clicked.emit(self))
        lay.addWidget(self.button)
        self.view = QLabel('')
        self.view.setFixedSize(212, 133)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setCursor(Qt.PointingHandCursor)
        self.view.installEventFilter(self)
        lay.addWidget(self.view)
        lay.addStretch(1)

    def eventFilter(self, obj, event):
        if obj is self.view and event.type() == QEvent.MouseButtonDblClick \
                and event.button() == Qt.LeftButton:
            self.mark_clicked.emit(self)
            return True
        return super().eventFilter(obj, event)

    def set_processed(self, pil_img):
        if pil_img is None:
            self.view.clear()
            return
        pm = QPixmap.fromImage(pil_to_qimage(pil_img))
        pm = pm.scaled(212, 133, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.view.setPixmap(pm)


class PairRow(QFrame):
    """一整组（照 Merge 实拍横排）：左轨＝组号竖排＋其下垃圾桶删除钮；
    两列 SlotWidget 之间一条竖分隔线，⇄ 对调钮压在线上居中；
    控制列数＝1 时只剩第一列（第二列与分隔/对调一并隐藏）。"""

    def __init__(self, win, index):
        super().__init__()
        self.win = win
        self.index = index
        self.setStyleSheet('QFrame { background: white; border: none; }')
        self.setFixedHeight(190)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 0, 0, 0)
        lay.setSpacing(0)
        rail = QVBoxLayout()
        rail.setContentsMargins(0, 0, 0, 0)
        rail.setSpacing(0)
        rail.addSpacing(46)
        self.num = QLabel('')
        self.num.setAlignment(Qt.AlignCenter)
        self.num.setStyleSheet(f'color: {INK}; font-size: 14px;')
        num_row = QHBoxLayout()
        num_row.addStretch(1)
        num_row.addWidget(self.num)
        num_row.addStretch(1)
        rail.addLayout(num_row)
        rail.addSpacing(6)
        self.del_btn = QPushButton()
        self.del_btn.setIcon(icon('trash_dark'))
        self.del_btn.setIconSize(QSize(16, 16))
        self.del_btn.setFixedSize(36, 36)
        self.del_btn.setStyleSheet(
            f'QPushButton {{ background: {GRAY_BTN}; border: none;'
            ' border-radius: 4px; min-height: 36px; }'
            'QPushButton:hover { background: #ECECEC; }')
        self.del_btn.clicked.connect(lambda: win.delete_pair(self.index))
        rail.addWidget(self.del_btn)
        rail.addStretch(1)
        rail_w = QWidget()
        rail_w.setLayout(rail)
        rail_w.setFixedWidth(36)
        lay.addWidget(rail_w)
        lay.addSpacing(8)
        self.front = SlotWidget('front', 1)
        self.back = SlotWidget('back', 2)
        for s in (self.front, self.back):
            s.import_clicked.connect(win.on_import_button)
            s.mark_clicked.connect(win.on_thumb_mark)
        lay.addWidget(self.front)
        self.mid = QWidget()
        self.mid.setFixedWidth(33)
        grid = QGridLayout(self.mid)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)
        line_wrap = QWidget()
        ll = QHBoxLayout(line_wrap)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addStretch(1)
        ln = QFrame()
        ln.setFixedWidth(1)
        ln.setStyleSheet(f'background: {LINE}; border: none;')
        ll.addWidget(ln)
        ll.addStretch(1)
        grid.addWidget(line_wrap, 0, 0)
        self.swap = QPushButton('⇄')
        self.swap.setFixedSize(30, 30)
        self.swap.setStyleSheet(
            f'QPushButton {{ background: {GRAY_BTN}; color: {INK}; border: none;'
            ' border-radius: 4px; font-size: 15px; min-height: 30px;'
            ' padding: 0px; }'
            'QPushButton:hover { background: #ECECEC; }')
        self.swap.clicked.connect(lambda: win.swap_pair(self.index))
        grid.addWidget(self.swap, 0, 0, Qt.AlignCenter)
        lay.addWidget(self.mid)
        lay.addWidget(self.back)
        lay.addStretch(1)

    def set_cols(self, cols):
        vis = cols >= 2
        self.back.setVisible(vis)
        self.mid.setVisible(vis)

    @staticmethod
    def _btn_style():
        return (f'QPushButton {{ background: {BAR_BG}; border: none; border-radius: 4px;'
                ' padding: 8px 14px; font-size: 14px; color: #333; }'
                f'QPushButton:hover {{ background: #E2E7EE; }}')


class MainWindow(QMainWindow):
    ocr_done = Signal(int, str, object, str, object, object, object)
    ocr_probe_done = Signal(bool)
    detect_done = Signal(object)
    lic_done = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('IDCardMuse 智能证件合并工具v1.0 by xm 2026-11')
        # 客户区照 Merge 实测 1000×650（MERGE-UI-MEASURE.md §1/§11）
        self.resize(1000, 650)
        self.setMinimumSize(900, 620)
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
        # 控制列数（1/2，默认 2）与增强/模板的现行值：主窗不再摆 Merge
        # 没有的下拉，两值由单选（彩色/黑白）与保存文件承载，_sync 时
        # 写进 core（见 _sync_core_settings）。
        self.cols = 2
        self._enh_mode = '彩色'
        self._template_value = '标准（每组一页）'
        self._build_ui()
        self.add_pair()
        self._restore_settings_ui()
        self.setAcceptDrops(True)
        # OCR（全能版并入）：后台线程回主线程走信号，不直接碰控件
        self.ocr_done.connect(self._on_ocr_done)
        self.ocr_probe_done.connect(self._on_ocr_probe)
        self._ocr_dead = False
        threading.Thread(target=self._probe_ocr, daemon=True).start()
        # 边框识别（S1 修复）：与 OCR 同模式——单个后台线程顺序执行，
        # 结果走 detect_done 信号回主线程应用；识别期间窗口保持响应，
        # 以等待光标＋状态行作忙碌指示。队列里的任务只携带 (core, 图)
        # 与回写坐标，回主线程时按图对象身份校验槽位是否仍是那张图。
        self.detect_done.connect(self._on_detect_done)
        self.lic_done.connect(self._on_lic_done)
        self._detect_queue = queue.Queue()
        self._detect_worker = None
        self._detect_pending = 0        # 已排队未完成的识别数（光标/进度）
        self._mark_open = False         # 标角窗正开着（防自动再开一个）
        self._closing = False

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
        # S3 修复：随信号带上槽位 dict 与图对象本身，回写时按身份校验
        # 「这还是排队时的那个槽、那张图」——只验组号下标时，删组/
        # 清槽/换图后结果会落到另一张卡头上，导出把甲卡信息记给乙卡。
        slot = data
        img = pil_img

        def worker():
            err_text = ''
            texts = []
            try:
                info, texts = ocr_qt.read_card_full(img, side)
            except Exception as e:
                info = {}
                err_text = repr(e)
                ocr_qt.log_ocr_error(e)
            self.ocr_done.emit(idx, side, info, err_text, slot, img, texts)

        threading.Thread(target=worker, daemon=True).start()

    def _on_ocr_done(self, idx, side, info, err_text, slot, img, texts):
        if not info and err_text:
            self._ocr_dead = True
            self._refresh_info_btns()
            self.set_status('读字没跑起来，原因记在程序旁边的 ocr_error.log')
            return
        if idx < 0 or idx >= len(self.core.pairs):
            return
        pair = self.core.pairs[idx]
        # 身份校验（必须用 is：dict 按值比较会把内容相同的两个槽误判
        # 为同一个）。槽位 dict 已不在本组（删组/清槽/换槽）→ 弃；
        # 本组内正反互换是整 dict 搬家，结果跟着图走、写回原 dict 即
        # 正确落位。槽里的图已不是排队那张（被换掉）→ 弃。
        if not (slot is pair['front'] or slot is pair['back']):
            return
        if slot.get('processed') is not img or slot.get('_ocr_key') != id(img):
            return
        slot['ocr'] = info
        # 面别纠正：原始行与判定跟着槽位 dict 走（整 dict 搬家时随图
        # 归位）；两面都判出且正好相反时由 _maybe_swap_pair_sides 对调。
        # 检查放在本函数末尾，让对调文案不被「已识别」文案盖掉。
        slot['_ocr_texts'] = texts
        slot['_side_verdict'] = ocr_qt.detect_side_by_ocr(texts)
        if not info:
            self._maybe_swap_pair_sides(idx)
            return
        if info.get('姓名'):
            self.set_status('已识别：%s %s' % (info.get('姓名', ''),
                                               info.get('公民身份号码', '')))
        elif info.get('签发机关'):
            self.set_status('已识别签发机关：%s' % info.get('签发机关', ''))
        self._maybe_swap_pair_sides(idx)

    def _maybe_swap_pair_sides(self, idx):
        """导入面别纠正（与全能版 _maybe_swap_pair_sides 同口径）：一组
        两面都已处理且都判出面别、且判定与槽位正好相反时，整组对调
        摆正；任一面判不准就不动、不猜。对调走 swap_pair 同一套整
        dict 搬家，结果跟着图走。"""
        if idx < 0 or idx >= len(self.core.pairs):
            return False
        pair = self.core.pairs[idx]
        verdicts = {}
        for side in ('front', 'back'):
            slot = pair[side]
            if slot.get('processed') is None:
                return False
            v = slot.get('_side_verdict')
            if v is None:
                return False
            verdicts[side] = v
        if verdicts.get('front') == 'back' and verdicts.get('back') == 'front':
            # 先清判定防重复触发：搬家后判定随 dict 走，留着会让后续
            # 任一次回写再查时误判已对调的组。
            pair['front']['_side_verdict'] = None
            pair['back']['_side_verdict'] = None
            self.swap_pair(idx)
            self.set_status(f'第{idx + 1}组正反面放反了，已自动对调摆正')
            return True
        return False

    # ---------- UI construction ----------
    def _build_ui(self):
        # 主窗照 Merge v2.2 实测规格（MERGE-UI-MEASURE.md §8/§11 与
        # MERGE-FULL-SPEC.md 落位专节/补取专节）：纵向分段＝顶行→分隔线→
        # 提示行→分隔线→导入区→分隔线→底部按钮行→参数行→生成钮→分隔线
        # →状态行。界面不写说明书：除原版有的提示行/状态行不加解释小字。
        central = QWidget()
        central.setStyleSheet('background: white;')
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 顶行：左蓝钮「营业执照处理」142×37；右「设置」＋「清空数据」96×36
        top = QHBoxLayout()
        top.setContentsMargins(7, 7, 8, 0)
        top.setSpacing(6)
        self.lic_btn = QPushButton('营业执照处理')
        self.lic_btn.setIcon(icon('card_white'))
        self.lic_btn.setIconSize(QSize(20, 20))
        self.lic_btn.setFixedSize(142, 37)
        self.lic_btn.setStyleSheet(
            f'QPushButton {{ background: {BLUE}; color: white; border: none;'
            ' border-radius: 4px; font-size: 14px; min-height: 37px; }'
            f'QPushButton:hover {{ background: {BLUE_HOVER}; }}')
        self.lic_btn.clicked.connect(self.open_license_flow)
        top.addWidget(self.lic_btn)
        top.addStretch(1)
        self.settings_btn = QPushButton('设置')
        self.settings_btn.setFixedHeight(36)
        self.settings_btn.setStyleSheet(self._gray_qss())
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        top.addWidget(self.settings_btn)
        self.clear_btn = QPushButton('清空数据')
        self.clear_btn.setIcon(icon('trash_dark'))
        self.clear_btn.setIconSize(QSize(15, 15))
        self.clear_btn.setFixedSize(96, 36)
        self.clear_btn.setStyleSheet(self._gray_qss())
        self.clear_btn.clicked.connect(self.clear_all)
        top.addWidget(self.clear_btn)
        root.addLayout(top)
        root.addSpacing(7)
        root.addWidget(self._sep())

        # 提示行（带高 40，居中；文案照 exe 原文取证）
        self.hint = QLabel('请分别添加证件正面和背面，支持批量导入。')
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setFixedHeight(40)
        self.hint.setStyleSheet(f'color: {INK}; font-size: 14px;')
        root.addWidget(self.hint)
        root.addWidget(self._sep())

        # 导入区：左＝组列表（竖向叠，滚动），右＝控制列数固定轨
        mid = QHBoxLayout()
        mid.setContentsMargins(0, 0, 0, 0)
        mid.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet('QScrollArea { background: white; border: none; }')
        self.pairs_host = QWidget()
        self.pairs_host.setStyleSheet('background: white;')
        self.pairs_lay = QVBoxLayout(self.pairs_host)
        self.pairs_lay.setContentsMargins(0, 2, 0, 2)
        self.pairs_lay.setSpacing(4)
        self.pairs_lay.addStretch(1)
        self.scroll.setWidget(self.pairs_host)
        mid.addWidget(self.scroll, 1)
        ctrl = QVBoxLayout()
        ctrl.setContentsMargins(0, 0, 8, 0)
        ctrl.setSpacing(0)
        ctrl.addSpacing(57)
        lab = QLabel('控制列数')
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet(f'color: {INK}; font-size: 14px;')
        lab.setFixedWidth(72)
        ctrl.addWidget(lab)
        ctrl.addSpacing(14)
        self.col_plus = QPushButton()
        self.col_plus.setIcon(icon('plus_dark'))
        self.col_plus.setIconSize(QSize(14, 14))
        self.col_plus.setFixedSize(72, 36)
        self.col_plus.setStyleSheet(self._gray_qss())
        self.col_plus.clicked.connect(lambda: self.set_cols(self.cols + 1))
        ctrl.addWidget(self.col_plus)
        ctrl.addSpacing(4)
        self.col_minus = QPushButton()
        self.col_minus.setIcon(icon('minus_dark'))
        self.col_minus.setIconSize(QSize(14, 14))
        self.col_minus.setFixedSize(72, 36)
        self.col_minus.setStyleSheet(self._gray_qss())
        self.col_minus.clicked.connect(lambda: self.set_cols(self.cols - 1))
        ctrl.addWidget(self.col_minus)
        ctrl.addStretch(1)
        ctrl_w = QWidget()
        ctrl_w.setLayout(ctrl)
        ctrl_w.setFixedWidth(80)
        mid.addWidget(ctrl_w)
        root.addLayout(mid, 1)
        root.addWidget(self._sep())

        # 底部按钮行：新增一行 113×36 / 批量导入（按列数分组）蓝 197×38 /
        # 右端 彩色模式·黑白模式 单选
        brow = QHBoxLayout()
        brow.setContentsMargins(8, 8, 8, 0)
        brow.setSpacing(3)
        self.add_btn = QPushButton('新增一行')
        self.add_btn.setIcon(icon('plus_dark'))
        self.add_btn.setIconSize(QSize(12, 12))
        self.add_btn.setFixedSize(113, 36)
        self.add_btn.setStyleSheet(self._gray_qss())
        self.add_btn.clicked.connect(self.add_pair)
        brow.addWidget(self.add_btn)
        self.batch_btn = QPushButton('批量导入（按列数分组）')
        self.batch_btn.setIcon(icon('folder_white'))
        self.batch_btn.setIconSize(QSize(16, 16))
        self.batch_btn.setFixedSize(197, 38)
        self.batch_btn.setStyleSheet(
            f'QPushButton {{ background: {BLUE}; color: white; border: none;'
            ' border-radius: 4px; font-size: 14px; min-height: 38px; }'
            f'QPushButton:hover {{ background: {BLUE_HOVER}; }}')
        self.batch_btn.clicked.connect(self.batch_import)
        brow.addWidget(self.batch_btn)
        brow.addStretch(1)
        self.radio_color = QRadioButton('彩色模式')
        self.radio_bw = QRadioButton('黑白模式')
        for r in (self.radio_color, self.radio_bw):
            r.setStyleSheet(self._radio_qss())
        self.radio_color.setChecked(True)
        self.radio_color.toggled.connect(self._sync_mode_from_radios)
        self.radio_bw.toggled.connect(self._sync_mode_from_radios)
        brow.addWidget(self.radio_color)
        brow.addSpacing(5)
        brow.addWidget(self.radio_bw)
        root.addLayout(brow)

        # 参数行：五组「标签＋下拉 125×32」（选项/默认照补取专节）
        prow = QHBoxLayout()
        prow.setContentsMargins(8, 11, 8, 0)
        prow.setSpacing(0)
        self.wm_preset = self._param_combo(prow, '安全水印：', WM_PRESETS, '无水印')
        self.wm_preset.currentIndexChanged.connect(self._on_wm_preset)
        self.wm_size_c = self._param_combo(prow, '字号：', WM_SIZE_ITEMS, '小 (36)')
        self.wm_color_c = self._param_combo(prow, '颜色：', WM_COLOR_ITEMS,
                                           '黑色 （默认）')
        self.wm_opacity_c = self._param_combo(
            prow, '透明度(%)：', [str(i) for i in range(10, 101, 10)], '50')
        self.wm_rows_c = self._param_combo(
            prow, '行数：', [str(i) for i in range(1, 11)], '3', last=True)
        prow.addStretch(1)
        root.addLayout(prow)

        # 自定义水印输入行：仅选「自定义」时出现（照补取专节图 43）
        self.wm_custom_row = QWidget()
        crow = QHBoxLayout(self.wm_custom_row)
        crow.setContentsMargins(8, 6, 8, 0)
        crow.setSpacing(0)
        clab = QLabel('自定义内容：')
        clab.setStyleSheet(f'color: {INK}; font-size: 14px;')
        crow.addWidget(clab)
        self.wm_custom = QLineEdit()
        self.wm_custom.setPlaceholderText('请输入自定义水印内容...')
        self.wm_custom.setFixedSize(360, 32)
        self.wm_custom.setStyleSheet(
            f'QLineEdit {{ background: {COMBO_BG}; color: {INK}; border: none;'
            ' border-radius: 4px; padding-left: 9px; font-size: 14px; }')
        crow.addWidget(self.wm_custom)
        crow.addStretch(1)
        self.wm_custom_row.setVisible(False)
        root.addWidget(self.wm_custom_row)

        # 信息导出两钮（咱们独有，位置照现行：生成钮旁、右对齐）
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 6, 8, 0)
        info_row.addStretch(1)
        self.info_excel_btn = QPushButton('导出信息 Excel')
        self.info_txt_btn = QPushButton('导出信息 TXT')
        for b in (self.info_excel_btn, self.info_txt_btn):
            b.setFixedHeight(32)
            b.setStyleSheet(self._gray_qss(32))
        self.info_excel_btn.clicked.connect(self.export_info_excel)
        self.info_txt_btn.clicked.connect(self.export_info_txt)
        info_row.addWidget(self.info_excel_btn)
        info_row.addWidget(self.info_txt_btn)
        root.addLayout(info_row)
        root.addSpacing(4)

        # 橙色生成钮（近满宽 ×38）
        grow = QHBoxLayout()
        grow.setContentsMargins(7, 0, 7, 0)
        self.gen_btn = QPushButton('生成标准 A4 并预览')
        self.gen_btn.setFixedHeight(38)
        self.gen_btn.setStyleSheet(
            f'QPushButton {{ background: {ORANGE}; color: white; border: none;'
            ' border-radius: 4px; font-size: 14px; min-height: 38px; }'
            f'QPushButton:hover {{ background: {ORANGE_HOVER}; }}')
        self.gen_btn.clicked.connect(self.generate_preview)
        grow.addWidget(self.gen_btn)
        root.addLayout(grow)
        root.addSpacing(7)
        root.addWidget(self._sep())

        # 状态行（带高 47，居中）
        self.status = QLabel('就绪')
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setFixedHeight(47)
        self.status.setStyleSheet(f'color: {INK}; font-size: 14px;')
        root.addWidget(self.status)

    @staticmethod
    def _sep():
        ln = QFrame()
        ln.setFixedHeight(1)
        ln.setStyleSheet(f'background: {LINE}; border: none;')
        return ln

    @staticmethod
    def _gray_qss(h=36):
        # Merge 灰钮：#F5F5F5 填充、无边、圆角 4、墨字。min-height 必须
        # 钉到目标高：QSS 的 min-height 会覆盖 setFixedHeight 的下限，
        # 写 0 会被布局按 sizeHint 压矮（实测 37→22）。
        return (f'QPushButton {{ background: {GRAY_BTN}; color: {INK}; border: none;'
                f' border-radius: 4px; font-size: 14px; min-height: {h}px;'
                ' padding-left: 10px; padding-right: 10px; }'
                'QPushButton:hover { background: #ECECEC; }')

    @staticmethod
    def _radio_qss():
        # 单选照 Merge 实拍：未选＝浅灰环、已选＝灰环＋蓝盘；间距视觉对齐
        return (
            f'QRadioButton {{ color: {INK}; font-size: 14px; spacing: 9px; }}'
            'QRadioButton::indicator { width: 18px; height: 18px; }'
            f'QRadioButton::indicator:unchecked {{ image: url({ICONS_DIR}/radio_unchecked.svg); }}'
            f'QRadioButton::indicator:checked {{ image: url({ICONS_DIR}/radio_checked.svg); }}')

    def _param_combo(self, prow, label, items, current, last=False):
        lab = QLabel(label)
        lab.setStyleSheet(f'color: {INK}; font-size: 14px;')
        prow.addWidget(lab)
        prow.addSpacing(4)
        combo = QComboBox()
        combo.addItems(items)
        if current in items:
            combo.setCurrentText(current)
        combo.setFixedSize(125, 32)
        combo.setStyleSheet(
            f'QComboBox {{ background: {COMBO_BG}; color: {INK}; border: none;'
            ' border-radius: 4px; padding-left: 9px; font-size: 14px;'
            ' min-height: 32px; }'
            'QComboBox::drop-down { border: none; width: 26px; }'
            f'QComboBox::down-arrow {{ image: url({ICONS_DIR}/down_arrow.svg);'
            ' width: 12px; height: 12px; }'
            f'QComboBox QAbstractItemView {{ background: white; color: {INK};'
            f' selection-background-color: {BLUE}; selection-color: white; }}')
        prow.addWidget(combo)
        if not last:
            prow.addSpacing(14)
        return combo

    def _on_wm_preset(self, *_):
        self.wm_custom_row.setVisible(
            self.wm_preset.currentText() == '自定义')

    def _sync_mode_from_radios(self, *_):
        self._enh_mode = '黑白' if self.radio_bw.isChecked() else '彩色'

    def set_cols(self, cols):
        """控制列数 1/2：2＝组内左右两列；1＝每组只剩一列（第二张先藏
        不删，切回来还在）。批量/拖入的分组口径随之走。"""
        cols = max(1, min(2, int(cols)))
        if cols == self.cols:
            return
        self.cols = cols
        for r in self.rows:
            r.set_cols(cols)

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
        row.set_cols(self.cols)
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
        """「清空数据」：有数据才弹确认——标题/正文/钮文案照 Merge 实拍
        （补取专节图 45/46/47：标题「清空确认」、正文「确定要清空所有
        图片和内存吗？」、钮 ✗ No／✓ Yes），无数据只报状态。"""
        if not any(self._row_has_data(i) for i in range(len(self.rows))):
            self.set_status('当前没有可清空的数据')
            return
        box = QMessageBox(self)
        box.setWindowTitle('清空确认')
        box.setText('确定要清空所有图片和内存吗？')
        yes_b = box.addButton('✓ Yes', QMessageBox.AcceptRole)
        no_b = box.addButton('✗ No', QMessageBox.RejectRole)
        box.setDefaultButton(no_b)
        box.exec_()
        if box.clickedButton() != yes_b:
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
            # 组号竖排照 Merge 实拍（第／N／组 三字纵列）
            r.num.setText(f'第\n{i + 1}\n组')

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
                    # 识别改走后台队列；认不出时由完成回调开标角窗
                    self.import_into_slot(i, slot.side, path,
                                          open_mark_on_fail=True)
                return

    def _slot_pos(self, slot):
        for i, r in enumerate(self.rows):
            if slot is r.front:
                return i, 'front'
            if slot is r.back:
                return i, 'back'
        return None

    def on_import_button(self, slot):
        """槽上「导入证件 N」灰钮（照 Merge 实拍）：点即选图导入；
        槽里已有图再点＝换图（导入管线自带占槽覆盖）。"""
        pos = self._slot_pos(slot)
        if pos is None:
            return
        i, side = pos
        path, _ = QFileDialog.getOpenFileName(
            self, '选择图片', self.last_dir,
            '图片 (*.jpg *.jpeg *.png *.bmp *.webp *.heic *.heif *.tif *.tiff)')
        if path:
            self.import_into_slot(i, side, path, open_mark_on_fail=True)

    def on_thumb_mark(self, slot):
        """缩略图双击＝手动微调边缘（照 Merge「双击图片可微调边缘」）。"""
        pos = self._slot_pos(slot)
        if pos is None:
            return
        i, side = pos
        if self.core.pairs[i][side]['image'] is not None:
            self.open_mark_dialog(i, side)

    def open_mark_dialog(self, idx, side):
        from mark_dialog import MarkDialog
        if self.core.pairs[idx][side]['image'] is None:
            return
        self._mark_open = True
        try:
            MarkDialog(self, idx, side).exec_()
        finally:
            self._mark_open = False

    # ----- S1: 后台识别队列（单线程顺序执行，信号回主线程） -----
    def _enqueue_detect(self, idx, side, img, stats=None,
                        open_mark_on_fail=False):
        job = {'idx': idx, 'side': side, 'img': img, 'stats': stats,
               'open_mark': open_mark_on_fail}
        if stats is not None:
            stats['pending'] += 1
        if self._detect_pending == 0:
            QApplication.setOverrideCursor(Qt.WaitCursor)
        self._detect_pending += 1
        name = '正面' if side == 'front' else '反面'
        self.set_status(f'正在识别第 {idx + 1} 组{name}，请稍候…')
        self._detect_queue.put(job)
        if self._detect_worker is None or not self._detect_worker.is_alive():
            self._detect_worker = threading.Thread(
                target=self._detect_worker_loop, daemon=True)
            self._detect_worker.start()

    def _detect_worker_loop(self):
        while True:
            job = self._detect_queue.get()
            res = dict(job)
            res['corners'] = None
            res['processed'] = None
            try:
                self.core.current_marking_side = job['side']
                corners = self.core.ai_detect_any(job['img'])
                res['corners'] = corners
                if corners:
                    # AI 检出的角点已按证件本身排序（左上→右上→右下→
                    # 左下），必须 points_ordered=True 原样拉正才会自动
                    # 转正；几何重排只许用于手工标角（FIX3 修 2）。
                    res['processed'] = self.core.process_card(
                        job['img'], corners, points_ordered=True)
            except Exception as e:
                res['error'] = repr(e)
            try:
                self.detect_done.emit(res)
            except RuntimeError:
                pass

    def _detect_bookkeeping(self, stats):
        """每个识别任务收尾时调用（无论结果是否被采用）：回收忙碌
        计数/光标，批量统计 pending 归零时播报汇总。返回 True 表示
        已播报汇总（调用方不要再用单张状态覆盖它）。"""
        self._detect_pending = max(0, self._detect_pending - 1)
        if self._detect_pending == 0:
            QApplication.restoreOverrideCursor()
        if stats is not None:
            stats['pending'] -= 1
            if stats['pending'] <= 0:
                msg = (f"{stats['label']}完成：成功 {stats['ok']} 张，"
                       f"失败 {stats['fail']} 张")
                if (getattr(self.core, '_ai_error', None)
                        and not getattr(self.core, '_ai_ran', False)):
                    msg += '；识别模型不可用，失败的请逐张手动标角'
                self.set_status(msg)
                return True
        return False

    def _on_detect_done(self, res):
        if getattr(self, '_closing', False):
            return
        idx, side, img = res['idx'], res['side'], res['img']
        stats = res.get('stats')
        if idx < 0 or idx >= len(self.core.pairs):
            self._detect_bookkeeping(stats)
            return
        data = self.core.pairs[idx][side]
        if data.get('image') is not img:
            # 识别期间该槽已被清掉/换了图：结果作废，不落到新图头上
            self._detect_bookkeeping(stats)
            return
        corners, processed = res.get('corners'), res.get('processed')
        data['points'] = [tuple(p) for p in corners] if corners else None
        data['processed'] = processed
        (self.rows[idx].front if side == 'front'
         else self.rows[idx].back).set_processed(processed)
        name = '正面' if side == 'front' else '反面'
        if processed is not None:
            if stats is not None:
                stats['ok'] += 1
            summarized = self._detect_bookkeeping(stats)
            if not summarized:
                self.set_status(f'第 {idx + 1} 组{name}导入成功，已自动处理')
            self._schedule_ocr(idx, side, processed)
            return
        if stats is not None:
            stats['fail'] += 1
        summarized = self._detect_bookkeeping(stats)
        # S2 口径：模型缺失/损坏与「照片没认出」分开说，不再冒充照片问题
        if not summarized:
            if (getattr(self.core, '_ai_error', None)
                    and not getattr(self.core, '_ai_ran', False)):
                self.set_status(
                    f'第 {idx + 1} 组{name}：识别模型不可用'
                    '（原因记在程序旁边的 detect_error.log），请手动标角')
            else:
                self.set_status(
                    f'第 {idx + 1} 组{name}未检测到高精度边框，'
                    '请换一张或稍后手动标角')
        if res.get('open_mark') and not self._mark_open:
            self.open_mark_dialog(idx, side)

    def import_into_slot(self, idx, side, path, stats=None,
                         open_mark_on_fail=False):
        """同步只做开图与占槽（快），识别排队到后台线程。返回 True=
        已受理（识别结果异步到达），False=图片打开失败。"""
        try:
            img = Image.open(path)
            img.load()
        except Exception as e:
            self.set_status(f'图片打开失败：{e}')
            if stats is not None:
                stats['fail'] += 1
            return False
        self.last_dir = os.path.dirname(path)
        self._sync_core_settings()
        data = self.core.pairs[idx][side]
        data['image'] = img
        data['path'] = path
        data['points'] = None
        data['processed'] = None
        data['ocr'] = {}
        data['_ocr_key'] = None
        (self.rows[idx].front if side == 'front'
         else self.rows[idx].back).set_processed(None)
        self._enqueue_detect(idx, side, img, stats=stats,
                             open_mark_on_fail=open_mark_on_fail)
        return True

    def batch_import(self):
        # 文件框标题照 Merge exe 原文（%d = 当前列数）
        paths, _ = QFileDialog.getOpenFileNames(
            self, f'批量选择照片（将以 {self.cols} 张为一组）', self.last_dir,
            '图片 (*.jpg *.jpeg *.png *.bmp *.webp *.heic *.heif *.tif *.tiff)')
        if paths:
            self.import_files_as_pairs(paths)

    def _first_free_pair(self):
        """第一个整组空着的组号（FIX3 修 1：批量导入先填已有空组，
        不再无条件新建组、把第 1 组晾着）。列数=1 时只看第一槽。"""
        sides = ('front',) if self.cols == 1 else ('front', 'back')
        for i, pair in enumerate(self.core.pairs):
            if all(pair[s]['image'] is None for s in sides):
                return i
        return None

    def import_files_as_pairs(self, paths, stats=None):
        paths = [p for p in paths if p.lower().endswith(IMG_EXTS)]
        if stats is None:
            stats = {'ok': 0, 'fail': 0, 'pending': 0, 'label': '批量导入'}
        if self.cols == 1:
            # 单列口径：一组一张（照 Merge 列数=1 实拍）
            for p in paths:
                idx = self._first_free_pair()
                if idx is None:
                    idx = self.add_pair()
                self.import_into_slot(idx, 'front', p, stats=stats)
        else:
            for k in range(0, len(paths), 2):
                idx = self._first_free_pair()
                if idx is None:
                    idx = self.add_pair()
                for side, p in zip(('front', 'back'), paths[k:k + 2]):
                    self.import_into_slot(idx, side, p, stats=stats)
        if stats['pending'] == 0:
            # 全部在开图阶段就失败了：没有异步回调会来播报，这里补上
            self.set_status(f"{stats['label']}完成：成功 {stats['ok']} 张，"
                            f"失败 {stats['fail']} 张")
        return stats

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
        stats = {'ok': 0, 'fail': 0, 'pending': 0, 'label': '拖入导入'}
        rest = list(paths)
        sides = ('front',) if self.cols == 1 else ('front', 'back')
        for i, pair in enumerate(self.core.pairs):
            for side in sides:
                if rest and pair[side]['image'] is None:
                    self.import_into_slot(i, side, rest.pop(0), stats=stats)
        if rest:
            self.import_files_as_pairs(rest, stats=stats)
        elif stats['pending'] == 0 and (stats['ok'] or stats['fail']):
            self.set_status(f"拖入导入完成：成功 {stats['ok']} 张，"
                            f"失败 {stats['fail']} 张")

    # ---------- settings ----------

    @staticmethod
    def _primary_style():
        return (f'QPushButton {{ background: {BLUE}; color: white; border: none;'
                ' border-radius: 4px; padding: 9px 18px; font-size: 14px; }'
                'QPushButton:hover { background: #2569D1; }')

    def _sync_core_settings(self):
        c = self.core
        c.enhance_mode = ENH_MAP.get(getattr(self, '_enh_mode', '彩色'), 'color')
        # 水印四件按补取专节联动：预设档位直接给出 开关+文案（无独立窗、
        # 无「启用水印」勾选）；字号档经 WM_SIZE_MAP 映射到 core 口径。
        preset = self.wm_preset.currentText()
        if preset == '无水印':
            c.wm_enabled = False
            c.wm_text = ''
        elif preset == '自定义':
            c.wm_enabled = True
            c.wm_text = self.wm_custom.text()
        else:
            c.wm_enabled = True
            c.wm_text = preset
        c.wm_size = WM_SIZE_MAP.get(self.wm_size_c.currentText(), 7)
        c.wm_color = self.wm_color_c.currentText().replace(' （默认）', '')
        c.wm_opacity = int(self.wm_opacity_c.currentText())
        c.wm_rows = int(self.wm_rows_c.currentText())
        c.template = getattr(self, '_template_value', '标准（每组一页）')
        # 设置窗五项（§7）：画布/卡片规格写进 core，默认值与 core 出厂一致
        c.a4_width = float(getattr(self, 'canvas_w', 210))
        c.a4_height = float(getattr(self, 'canvas_h', 297))
        c.card_width = float(getattr(self, 'card_w', 85.6))
        c.card_height = float(getattr(self, 'card_h', 54.0))

    def _save_settings(self):
        self._sync_core_settings()
        data = {'last_dir': self.last_dir, 'enhance': self._enh_mode,
                'enhance_id_default': self._enh_mode == '清晰',
                'wm_preset': self.wm_preset.currentText(),
                'wm_enabled': self.core.wm_enabled,
                'wm_text': self.wm_custom.text(),
                'wm_size': self.wm_size_c.currentText(),
                'wm_color': self.wm_color_c.currentText(),
                'wm_opacity': self.wm_opacity_c.currentText(),
                'wm_rows': self.wm_rows_c.currentText(),
                'template': self._template_value,
                'cols': self.cols,
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
            self._enh_mode = s['enhance']
        elif s.get('enhance_id_default'):
            # 清晰档粘性标记（出厂默认 False，见 idcard_settings.json）：
            # enhance 键缺失时凭它恢复上次的清晰档选择
            self._enh_mode = '清晰'
        # 单选只承载 彩色/黑白 两档；其余增强档仍按保存值经 _enh_mode
        # 生效（core 侧），单选停在彩色位。屏蔽信号免覆盖 _enh_mode。
        for r in (self.radio_color, self.radio_bw):
            r.blockSignals(True)
        self.radio_bw.setChecked(self._enh_mode == '黑白')
        self.radio_color.setChecked(self._enh_mode != '黑白')
        for r in (self.radio_color, self.radio_bw):
            r.blockSignals(False)
        preset = s.get('wm_preset')
        if preset in WM_PRESETS:
            self.wm_preset.setCurrentText(preset)
        if s.get('wm_text'):
            # 老存档（仅 wm_text）或新存档的自定义文案都落回输入框，
            # 并把档位切到「自定义」以免文字丢弃
            self.wm_custom.setText(s.get('wm_text', ''))
            if preset not in WM_PRESETS[1:3]:
                self.wm_preset.setCurrentText('自定义')
        self._on_wm_preset()
        size_v = s.get('wm_size')
        if size_v is not None:
            sv = str(size_v)
            if sv in WM_SIZE_ITEMS:
                self.wm_size_c.setCurrentText(sv)
            elif sv.isdigit():
                n = int(sv)
                label = '小 (36)' if n <= 8 else ('中 (56)' if n <= 12 else '大 (72)')
                self.wm_size_c.setCurrentText(label)
        color_v = s.get('wm_color')
        if color_v is not None:
            cv = str(color_v)
            if cv == '黑色':
                cv = '黑色 （默认）'
            if cv in WM_COLOR_ITEMS:
                self.wm_color_c.setCurrentText(cv)
        for combo, key in ((self.wm_opacity_c, 'wm_opacity'),
                           (self.wm_rows_c, 'wm_rows')):
            v = s.get(key)
            if v is not None and combo.findText(str(v)) >= 0:
                combo.setCurrentText(str(v))
        if s.get('template'):
            self._template_value = s['template']
        if s.get('cols') in (1, 2):
            self.cols = int(s['cols'])
            for r in self.rows:
                r.set_cols(self.cols)
        self._sync_core_settings()

    def closeEvent(self, event):
        self._closing = True     # 后台识别/OCR 的迟到结果一律不再应用
        self._save_settings()
        super().closeEvent(event)

    def set_status(self, text):
        self.status.setText(text)

    # ---------- 营业执照：共用 A4 预览窗（定案 3/4，不另设二级界面） ----
    def open_license_flow(self):
        """顶行蓝钮「营业执照处理」：选图 → 后台跑 lic_pipeline →
        处理完直接开与身份证一模一样的 A4 预览窗（横版页）。管线只
        调用公开接口 process_license（LIC-RUN 的成熟接法：detector 在
        worker 线程内取、状态回主线程应用）。手动标角/单独打印钮等
        执照页零碎按定案 4 并进预览窗六钮，不另设窗口。"""
        path, _ = QFileDialog.getOpenFileName(
            self, '选择营业执照图片', self.last_dir,
            '图片 (*.jpg *.jpeg *.png *.bmp *.webp *.heic *.heif *.tif *.tiff)')
        if not path:
            return
        try:
            img = Image.open(path)
            img.load()
            rgb = np.asarray(img.convert('RGB'))
        except Exception as e:
            QMessageBox.warning(self, '提示', f'图片打开失败：{e}')
            return
        self.last_dir = os.path.dirname(path)
        self._save_settings()
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        stem = os.path.splitext(os.path.basename(path))[0].strip() or '执照'
        self.set_status('正在处理营业执照，请稍候…')
        QApplication.setOverrideCursor(Qt.WaitCursor)

        def worker():
            res = None
            try:
                from lic_pipeline import process_license, OUT_A4
                get = getattr(self.core, 'get_ai_detector', None)
                det = get() if get is not None else getattr(
                    self.core, 'detector_ai', None)
                res = process_license(bgr, detector=det, out_size=OUT_A4)
            except Exception as e:
                res = {'status': 'error', 'error': repr(e)}
            res['_stem'] = stem
            try:
                self.lic_done.emit(res)
            except RuntimeError:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _on_lic_done(self, res):
        QApplication.restoreOverrideCursor()
        if getattr(self, '_closing', False):
            return
        if not res or res.get('status') != 'ok' or res.get('warped') is None:
            if (getattr(self.core, '_ai_error', None)
                    and not getattr(self.core, '_ai_ran', False)):
                msg = ('识别模型不可用（原因记在程序旁边的 '
                       'detect_error.log），这张没处理成')
            elif res and res.get('error'):
                msg = f"自动处理出错（{res['error']}），这张没处理成"
            else:
                msg = '没认出执照边框，换一张拍正一点的试试'
            self.set_status(msg)
            QMessageBox.warning(self, '提示', msg)
            return
        warped = res['warped']      # BGR ndarray，OUT_A4 = 2339×1654
        pil = Image.fromarray(np.ascontiguousarray(warped[:, :, ::-1]))
        stem = res.get('_stem', '执照')
        self.set_status('营业执照处理完成')
        dlg = PreviewDialog(self, [[(pil, 0.0, 0.0)]],
                            page_px=pil.size, page_mm=(297.0, 210.0),
                            base_name=stem,
                            crop_items=[[(f'{stem}_Crop_P1_1.jpg', pil)]],
                            images_mode=True)
        dlg.exec_()

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
        stash = []
        if self.cols == 1:
            # 单列口径（照 Merge 列数=1 实拍）：一面一页、单卡居中。
            # core.compute_layout 按组收集两面且 core 本期不许改，故在
            # 生成期间把第二槽数据暂存挪开、关窗即原样还原。
            for pair in self.core.pairs:
                if (pair['back']['image'] is not None
                        or pair['back']['processed'] is not None):
                    stash.append((pair, pair['back']))
                    pair['back'] = new_slot_data()
        try:
            layout = self.core.compute_layout()
            if not layout:
                # 照 Merge 实拍（补取 44）：无数据点生成弹「提示」窗
                QMessageBox.information(self, '提示', '没有可生成的图片数据')
                return
            # 自由排版手动结果留存（按 Merge：锁定/关窗不丢）：上次
            # 预览关窗时按「页内图身份签名」存下的布局副本，图集未
            # 变（签名逐图 id 一致且页/卡数同构）时原样带回——拖动
            # 位置与面板的改尺寸/旋转/增强一并保留；任一图被重处理
            # 或增删，签名即不符，自动回落到 core 新算的布局。
            sig = tuple(id(img) for page in layout for img, _x, _y in page)
            saved = getattr(self, '_free_saved', None)
            if (saved is not None and saved[0] == sig
                    and len(saved[1]) == len(layout)
                    and all(len(a) == len(b)
                            for a, b in zip(saved[1], layout))):
                layout = [[[img, float(x), float(y)] for img, x, y in page]
                          for page in saved[1]]
            dlg = PreviewDialog(self, layout)
            dlg.exec_()
            self._free_saved = (
                sig, [[[img, float(x), float(y)] for img, x, y in page]
                      for page in dlg.layout])
        finally:
            for pair, back in stash:
                pair['back'] = back


class PageView(QWidget):
    """A4 页交互视图（照 Merge v2.2 预览实拍）：#E6E6E6 灰底平铺，
    全部页纵向叠放，每页上方一条页标带「--- 第 N 页 A4 画布 ---」，
    白页内按布局放卡片图（整页图免圆角裁剪、原样贴）。

    页显示宽 = 对话框宽 × 0.567（原版 595/1050，PREVIEW-GEOMETRY.md），
    水平居中；坐标系与 core.compute_layout 一致（页全分辨率像素）；
    拖动写回本窗持有的布局副本，导出/打印按拖后位置渲染。身份证
    模式下关窗后由 MainWindow 按页内图身份签名留存本副本，再次生成
    时原样带回（按 Merge：锁定/关窗后手动排版不丢）。

    自由排版（原版语义）：默认锁定、点按无反应；开启后先点选一张卡
    才能拖被选中的那张，拖动任意像素自由放、无吸附无页界钳制；选中
    描边与合并 PDF 钮同为原版绿 #43F436，选中卡底部带尺寸签
    「85.6 x 54.0 mm」。「全选图片」由展开面板勾选接入（set_all_sel）：
    全选时全卡描边、面板的尺寸/旋转/增强操作作用于全部卡，拖动仍
    只动按下的那张。选中变化经 sel_changed 通知对话框同步面板。

    缩放：Ctrl+滚轮以光标点为锚在 25%–300% 间缩放（行业式补齐，
    非原版口径）；普通滚轮交给滚动区。

    横版页（执照 2339×1654）经 page_px/page_mm 注入同一控件显示，
    几何规则与竖版页一致。"""

    PAGE_W_RATIO = 0.567  # 页宽/对话框宽（原版 595/1050，本体常量）
    LABEL_H = 30          # 页标带高（Merge 实拍页上方灰带量级）
    ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 0.25, 3.0, 1.25
    zoom_changed = Signal(float)
    sel_changed = Signal()

    def __init__(self, core, layout, page_px=None, page_mm=None):
        super().__init__()
        self.core = core
        self.layout = layout            # [[[img, x, y], ...], ...]
        self.idx = 0                    # 兼容旧属性（现为多页同屏，未用）
        s = core.mm_to_pixel
        if page_px is None:
            page_px = (int(core.a4_width * s), int(core.a4_height * s))
        self.pw, self.ph = int(page_px[0]), int(page_px[1])
        self.page_mm = page_mm or (core.a4_width, core.a4_height)
        self.ref_w = 1050.0             # 参照宽 = 对话框宽，对话框 resize 时同步
        self.scale = 1.0
        self.ox = 0.0
        self._thumbs = {}
        self._drag = None               # (page, card, grab_dx, grab_dy) 内容像素
        self.free_mode = False          # 自由排版开关：默认锁定（原版语义）
        self.sel = None                 # 当前选中卡 (page, card)；None = 未选
        self.all_sel = False            # 全选态（面板「全选图片」勾选接入）
        self.zoom = 1.0                 # 1.0 = 原版口径；开窗新建、不持久化
        self._scroll_area = None        # 对话框建好滚动区后 attach_scroll 注入
        self._pending_anchor = None     # 缩放后待落位的 (滚动x, 滚动y)
        self.setMinimumSize(200, 120)
        self._recalc()

    # ----- geometry -----
    def _page_h(self):
        return self.ph * self.scale

    def _page_top(self, i):
        return i * (self.LABEL_H + self._page_h())

    def _content_size(self, vp_w=0, vp_h=0):
        page_w = self.ref_w * self.PAGE_W_RATIO * self.zoom
        w = max(float(vp_w), page_w)
        h = max(float(vp_h), len(self.layout) * (self.LABEL_H + self._page_h()))
        return w, h

    def set_ref_width(self, w):
        """对话框宽度变化时同步参照宽，页宽占比（0.567）随之恒定。"""
        if w and abs(float(w) - self.ref_w) > 0.5:
            self.ref_w = float(w)
            self._recalc()

    def _recalc(self):
        page_w = self.ref_w * self.PAGE_W_RATIO * self.zoom
        self.scale = page_w / self.pw
        self.ox = (self.width() - page_w) / 2 if self.width() > 0 else 0.0
        self.setMinimumWidth(int(round(page_w)))
        self.setMinimumHeight(
            int(round(len(self.layout) * (self.LABEL_H + self._page_h()))))
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
        前后该点在内容中的相对位置保持不变（页面不跳位）；不给
        anchor 时以视口中心为锚。"""
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
            old_w = max(float(self.width()), 1.0)
            old_h = max(float(self.height()), 1.0)
            rx = anchor[0] / old_w
            ry = anchor[1] / old_h
            vx, vy = anchor[0] - hb.value(), anchor[1] - vb.value()
            self.zoom = z
            self._recalc()
            new_w, new_h = self._content_size(vp.width(), vp.height())
            self._pending_anchor = (
                min(max(rx * new_w - vx, 0.0), max(0.0, new_w - vp.width())),
                min(max(ry * new_h - vy, 0.0), max(0.0, new_h - vp.height())))
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
        """适页：第一页（页标带＋整页）恰好塞进视口高，水平居中。"""
        sa = self._scroll_area
        if sa is None or sa.viewport().height() <= 0:
            return
        base = self.ref_w * self.PAGE_W_RATIO / self.pw   # 100% 时 scale
        target = max(1.0, sa.viewport().height() - self.LABEL_H) / self.ph
        self.set_zoom(target / base)

    def _card_rect(self, pi, img, x, y):
        top = self._page_top(pi) + self.LABEL_H
        return (self.ox + x * self.scale, top + y * self.scale,
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
        # PS 式平铺灰底（Merge 预览实拍多点采样 #E6E6E6）
        qp.fillRect(self.rect(), QColor(CANVAS_BG))
        page_w = self.pw * self.scale
        page_h = self._page_h()
        label_font = QFont()
        label_font.setPixelSize(13)
        for pi, page in enumerate(self.layout):
            top = self._page_top(pi)
            px, py = int(self.ox), int(top + self.LABEL_H)
            qp.setFont(label_font)
            qp.setPen(QColor(INK))
            qp.drawText(int(self.ox), int(top), int(page_w), self.LABEL_H,
                        Qt.AlignCenter, f'--- 第 {pi + 1} 页 A4 画布 ---')
            qp.fillRect(px, py, int(page_w), int(page_h), QColor('white'))
            qp.setPen(QPen(QColor('#BFBFBF'), 1))
            qp.drawRect(px, py, int(page_w), int(page_h))
            for img, x, y in page:
                rx, ry, rw, rh = self._card_rect(pi, img, x, y)
                thumb = self._thumb(img)
                qp.drawPixmap(QRectF(rx, ry, rw, rh), thumb,
                              QRectF(thumb.rect()))
        if self.free_mode:
            # 选中描边＝原版绿 #43F436（自由排版实拍采样，与合并 PDF
            # 钮同色）；全选时全卡描边，否则只描当前选中卡。
            if self.all_sel:
                marked = [(pi, ci) for pi, page in enumerate(self.layout)
                          for ci in range(len(page))]
            elif self.sel is not None:
                marked = (self.sel,)
            else:
                marked = ()
            for pi, ci in marked:
                img, x, y = self.layout[pi][ci]
                rx, ry, rw, rh = self._card_rect(pi, img, x, y)
                qp.setPen(QPen(QColor(GREEN), 2))
                qp.setBrush(Qt.NoBrush)
                qp.drawRect(QRectF(rx + 1, ry + 1, rw - 2, rh - 2))
            if self.sel is not None and not self.all_sel:
                # 选中卡底部尺寸签（照 Merge 实拍：深底白字居中，
                # 文案「85.6 x 54.0 mm」格式）
                pi, ci = self.sel
                img, x, y = self.layout[pi][ci]
                rx, ry, rw, rh = self._card_rect(pi, img, x, y)
                s = self.core.mm_to_pixel
                tag = f'{img.width / s:.1f} x {img.height / s:.1f} mm'
                tf = QFont()
                tf.setPixelSize(11)
                qp.setFont(tf)
                fm = qp.fontMetrics()
                tw = fm.horizontalAdvance(tag) + 12
                th = fm.height() + 5
                tx = rx + (rw - tw) / 2.0
                ty = ry + rh - th - 4
                bg = QColor(0, 0, 0)
                bg.setAlpha(170)
                qp.setPen(Qt.NoPen)
                qp.setBrush(bg)
                qp.drawRect(QRectF(tx, ty, tw, th))
                qp.setPen(QColor('white'))
                qp.drawText(QRectF(tx, ty, tw, th), Qt.AlignCenter, tag)
        qp.end()

    # ----- free layout & dragging -----
    def set_free_mode(self, on):
        """自由排版开关（原版 func27 语义）：开＝在模板现位置上原地
        解锁；关＝原地锁定，且连带清当前选中、全选态。位置不回滚：
        拖过的卡留在新位置（对话框持有的布局副本即工作状态）。"""
        self.free_mode = bool(on)
        if not self.free_mode:
            self._drag = None
            self.sel = None
            self.all_sel = False
        self.update()
        self.sel_changed.emit()

    def set_all_sel(self, on):
        """全选态（面板「全选图片」勾选）：全卡绿描边；面板的尺寸/
        旋转/增强操作由对话框作用于全部卡。拖动仍只动按下的那张
        （按下即把 sel 指向它，字段回填以它为准）。"""
        self.all_sel = bool(on)
        self.update()
        self.sel_changed.emit()

    def set_page_px(self, pw, ph, page_mm=None):
        """画布基础设置（面板纵/横切换）：换页像素尺寸与页毫米数；
        卡片坐标由对话框按新页宽高重排后写入布局，本方法只管画布
        几何（页标、缩放基准随新页尺寸重算）。"""
        self.pw, self.ph = int(pw), int(ph)
        if page_mm is not None:
            self.page_mm = page_mm
        self._recalc()
        self.update()

    def _press(self, vx, vy):
        # 锁定态：点按无反应（原版关态许可位直接返回）
        if not self.free_mode:
            return False
        for pi in range(len(self.layout) - 1, -1, -1):
            cards = self.layout[pi]
            for ci in range(len(cards) - 1, -1, -1):
                img, x, y = cards[ci]
                rx, ry, rw, rh = self._card_rect(pi, img, x, y)
                if rx <= vx <= rx + rw and ry <= vy <= ry + rh:
                    # 先点选、再拖动：按下即选中该卡并抓起它
                    self.sel = (pi, ci)
                    self._drag = (pi, ci, vx - rx, vy - ry)
                    self.update()
                    self.sel_changed.emit()
                    return True
        return False

    def _move(self, vx, vy):
        if not self._drag:
            return
        pi, ci, gdx, gdy = self._drag
        img, _x, _y = self.layout[pi][ci]
        # 任意像素自由放：无吸附、无页界钳制（原版原样累加增量，
        # 卡可拖出页外、坐标照存）
        nx = (vx - gdx - self.ox) / self.scale
        ny = (vy - gdy - (self._page_top(pi) + self.LABEL_H)) / self.scale
        self.layout[pi][ci] = [img, nx, ny]
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
        # 普通滚轮平移（不接受事件、传播给滚动区）；Ctrl+滚轮缩放。
        if event.modifiers() & Qt.ControlModifier:
            steps = event.angleDelta().y() / 120.0
            if steps:
                self.set_zoom(self.zoom * (self.ZOOM_STEP ** steps),
                              anchor=(float(event.x()), float(event.y())))
            event.accept()
            return
        super().wheelEvent(event)


class PreviewDialog(QDialog):
    """A4 预览窗（照 Merge v2.2 实拍）：顶排六钮按内容定宽、居中成组
    （导出裁剪／导出为 JPG／导出单页 PDF／导出合并 PDF／系统直接
    打印／开启自由排版），带同源 Material 一脉小图标；颜色照实测
    （JPG/单页 PDF 蓝 #296FF6、合并 PDF 绿 #43F436、自由排版橙
    #FF9800、裁剪/打印灰）。窗内不设 Merge 没有的钮与框：无页导航、
    无缩放簇、无标题行（缩放只留 Ctrl+滚轮）。页面 PS 式平铺灰底、
    多页纵叠、页标「--- 第 N 页 A4 画布 ---」。

    身份证页走 core 渲染（含水印与拖后坐标）；执照横版页走页图模式
    （images_mode）：整页图即页面内容，六钮流程与身份证页完全一致
    （导出/打印复用同一套实现，只接线不改算法）。

    自由排版展开面板（二期，照 Merge v2.2 补取专节）：开启自由排版
    时右侧滑出——画布基础设置（纵/横）、页面自动排版（exe 取证四
    模式、公式照 merge-layout-geometry.md）、🏷️ 常见尺寸快捷设置
    （含全选图片）、物理尺寸微调、操作图像旋转 90°、📄 文档模式
    增强、💡 提示：三行原文。锁定后被拖的卡留在新位置。"""

    def __init__(self, win, layout, page_px=None, page_mm=None,
                 base_name=None, crop_items=None, images_mode=False):
        super().__init__(win)
        self.win = win
        self.core = win.core
        # 布局副本：拖动与面板操作都改这份，即本窗的工作状态；
        # 身份证模式关窗时由 MainWindow 按图身份签名留存、再次生成
        # 原样带回（按 Merge：手动排版不丢）。
        self.layout = [[[img, float(x), float(y)] for img, x, y in page]
                       for page in layout]
        # 面板卡片操作状态（按 (页, 卡) 记）：base＝未经理操作的源图，
        # enh/rot/size 为面板三类操作的目标态，_render_card 统一由
        # base 重算显示图（旋转→增强→缩放，同 core 管线序），不叠加
        # 重采样。size=None 表示自然尺寸（px，旋转后的最终显示尺寸）。
        self._card_state = {}
        self._layout_mode = '上下居中（推荐）'
        self._sel_base = None           # 面板缩放联动的基准尺寸 (mm, mm)
        self._syncing = False           # 面板回填/联动期间的信号闸
        self.idx = 0
        self.images_mode = images_mode
        self.base_name = base_name
        self.crop_items_override = crop_items
        s = self.core.mm_to_pixel
        if page_px is None:
            page_px = (int(self.core.a4_width * s),
                       int(self.core.a4_height * s))
        self.page_px = (int(page_px[0]), int(page_px[1]))
        self.page_mm = page_mm or (self.core.a4_width, self.core.a4_height)
        # 画布方向（面板「画布基础设置」）：与初始页宽高一致——身份
        # 证页纵向、执照页横向，面板单选初始态即当前方向。
        self.orientation = ('landscape' if self.page_px[0] > self.page_px[1]
                            else 'portrait')
        # 每张布局卡对应的 (组序, 面)：compute_layout 按组序、正反序
        # 摊平，与布局顺序严格一致；供导出命名取源图基底（对照表 §5）。
        self._src_map = []
        if not images_mode:
            _flat = [(gi, side) for gi, pair in enumerate(self.core.pairs)
                     for side in ('front', 'back')
                     if pair[side].get('processed') is not None]
            if len(_flat) == sum(len(p) for p in self.layout):
                _it = iter(_flat)
                self._src_map = [[next(_it) for _ in page]
                                 for page in self.layout]
        self.setWindowTitle('可交互排版预览（支持拖放）')
        # 窗 1050×650 居中（原版 v2.2 本体真数，PREVIEW-GEOMETRY.md）
        self.resize(1050, 650)
        scr = QGuiApplication.primaryScreen()
        if scr is not None and QGuiApplication.platformName() != 'offscreen':
            geo = scr.availableGeometry()
            self.move(geo.x() + (geo.width() - self.width()) // 2,
                      geo.y() + (geo.height() - self.height()) // 2)
        self.setStyleSheet(f'background: {PAGE_BG};')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(8)
        # 六钮：按内容定宽（Merge 实拍为内容定宽居中成组，非等宽拉满），
        # 高 36、图标 18px、圆角 4。
        gray_qss = (
            f'QPushButton {{ background: {GRAY_BTN}; color: {INK}; border: none;'
            ' border-radius: 4px; padding: 0px 14px; font-size: 14px;'
            ' min-height: 36px; }'
            'QPushButton:hover { background: #ECECEC; }')
        color_qss = lambda bg, hover, fg='white': (
            f'QPushButton {{ background: {bg}; color: {fg}; border: none;'
            ' border-radius: 4px; padding: 0px 14px; font-size: 14px;'
            ' min-height: 36px; }'
            f'QPushButton:hover {{ background: {hover}; }}')
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addStretch(1)
        self.top_buttons = {}
        crop_b = QPushButton('导出裁剪')
        crop_b.setIcon(icon('cut_dark'))
        crop_b.setStyleSheet(gray_qss)
        crop_b.clicked.connect(self.export_crops)
        top.addWidget(crop_b)
        self.top_buttons['导出裁剪'] = crop_b
        for text, fn, qss, icn in (
                ('导出为 JPG', self.export_jpg,
                 color_qss(BLUE, BLUE_HOVER), 'image_white'),
                ('导出单页 PDF', self.export_pdf_single,
                 color_qss(BLUE, BLUE_HOVER), 'doc_white'),
                ('导出合并 PDF', self.export_pdf_merged,
                 color_qss(GREEN, '#37DB2C', fg='#202020'), 'doc_white'),
                ('系统直接打印', self.print_pages, gray_qss, 'printer_dark')):
            b = QPushButton(text)
            b.setIcon(icon(icn))
            b.setStyleSheet(qss)
            b.clicked.connect(fn)
            top.addWidget(b)
            self.top_buttons[text] = b
        self.print_btn = self.top_buttons['系统直接打印']
        # 第六钮「开启自由排版」：恒橙，开启后文案变「锁定当前排版」
        self.free_btn = QPushButton('开启自由排版')
        self.free_btn.setIcon(icon('grid_white'))
        self.free_btn.setStyleSheet(color_qss(ORANGE, ORANGE_HOVER))
        self.free_btn.clicked.connect(self.toggle_free)
        top.addWidget(self.free_btn)
        self.top_buttons['开启自由排版'] = self.free_btn
        for b in self.top_buttons.values():
            b.setFixedHeight(36)
            b.setIconSize(QSize(18, 18))
        top.addStretch(1)
        lay.addLayout(top)
        body = QHBoxLayout()
        body.setSpacing(8)
        lay.addLayout(body, 1)
        self.view = PageView(self.core, self.layout,
                             page_px=self.page_px, page_mm=self.page_mm)
        self.view.set_ref_width(self.width())
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.view)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setStyleSheet(f'background: {CANVAS_BG};')
        self.view.attach_scroll(self.scroll)
        body.addWidget(self.scroll, 1)
        # 自由排版展开面板：默认隐藏，开启自由排版时在右侧滑出、
        # 画布随之变窄（页宽参照随之扣掉面板宽，见 _sync_ref_width）
        self.panel = self._build_free_panel()
        body.addWidget(self.panel)
        self.view.sel_changed.connect(self._sync_panel_from_sel)

    def toggle_free(self):
        """自由排版开关（原版 func27＋补取专节）：文案双态、恒橙；
        开＝右侧面板滑出、画布变窄；关＝面板收起、view 清选中/全选
        态，但被拖的卡留在新位置（布局副本不回滚，按 Merge 锁定
        语义，实拍 52）。"""
        on = not self.view.free_mode
        self.view.set_free_mode(on)
        self.free_btn.setText('锁定当前排版' if on else '开启自由排版')
        self.panel.setVisible(on)
        if not on:
            # view 已清 all_sel，勾选随之落下（信号回环由 view 侧的
            # sel_changed 统一处理，字段回填会清成未选态）
            self.all_sel_check.setChecked(False)
        self._sync_ref_width()
        if on:
            self._sync_panel_from_sel()

    def _sync_ref_width(self):
        """页宽参照＝对话框宽（面板展开时扣掉面板宽，画布变窄后页
        随之缩，照 Merge 实拍 49 的画布变窄）。"""
        if getattr(self, 'view', None) is None:
            return
        w = self.width()
        panel = getattr(self, 'panel', None)
        if panel is not None and panel.isVisible():
            w -= panel.width()
        self.view.set_ref_width(w)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 页宽占比锚定参照宽：缩放窗体时同步（0.567 恒定）
        self._sync_ref_width()

    # ----- 自由排版展开面板（二期，照 MERGE-FULL-SPEC 补取专节） -----
    # 面板自上而下：画布基础设置／页面自动排版／🏷️ 常见尺寸快捷
    # 设置（含全选图片）／物理尺寸微调／操作图像旋转 90°／📄 文档
    # 模式增强／💡 提示：三行。除原版自带的节标题与提示外不加任何
    # 说明小字（定案 6）。色板与圆角照一期口径：蓝 #296FF6、橙
    # #FF9800、灰钮 #F5F5F5、圆角 4。
    PANEL_W = 328           # 面板外宽（Merge 实拍约 330@1280 窗宽）
    PT_MM = 25.4 / 72.0     # 1pt 的毫米数（页单位 pt 换算原版公式用）
    LAYOUT_MODES = ('上下居中（推荐）', '纵向紧凑 (双列)',
                    '单列居中 (清单)', '网格均分居中')

    def _build_free_panel(self):
        gray36 = MainWindow._gray_qss(36)
        radio_qss = MainWindow._radio_qss()

        def color_qss(bg, hover):
            return (f'QPushButton {{ background: {bg}; color: white;'
                    ' border: none; border-radius: 4px; padding: 0px 14px;'
                    ' font-size: 14px; min-height: 36px; }'
                    f'QPushButton:hover {{ background: {hover}; }}')

        def sep():
            ln = QFrame()
            ln.setFixedHeight(1)
            ln.setStyleSheet(f'background: {LINE}; border: none;')
            return ln

        def header(text, icn=None):
            row = QHBoxLayout()
            row.setSpacing(6)
            if icn:
                pic = QLabel()
                pic.setPixmap(icon(icn).pixmap(QSize(15, 15)))
                row.addWidget(pic)
            lab = QLabel(text)
            lab.setStyleSheet(f'color: {INK}; font-size: 14px;')
            row.addWidget(lab)
            row.addStretch(1)
            return row

        area = QScrollArea()
        area.setFixedWidth(self.PANEL_W)
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        area.setStyleSheet(f'background: {PAGE_BG};')
        content = QWidget()
        content.setStyleSheet(f'background: {PAGE_BG};')
        v = QVBoxLayout(content)
        v.setContentsMargins(8, 8, 8, 10)
        v.setSpacing(8)

        # ---- 画布基础设置 ----
        v.addLayout(header('画布基础设置', 'canvas_dark'))
        orow = QHBoxLayout()
        orow.setSpacing(16)
        self.radio_portrait = QRadioButton('纵向 (Portrait)')
        self.radio_landscape = QRadioButton('横向 (Landscape)')
        self._orient_group = QButtonGroup(self)
        for r in (self.radio_portrait, self.radio_landscape):
            r.setStyleSheet(radio_qss)
            self._orient_group.addButton(r)
            orow.addWidget(r)
        (self.radio_portrait if self.orientation == 'portrait'
         else self.radio_landscape).setChecked(True)
        orow.addStretch(1)
        v.addLayout(orow)
        self.radio_portrait.toggled.connect(
            lambda on: on and self._set_orientation('portrait'))
        self.radio_landscape.toggled.connect(
            lambda on: on and self._set_orientation('landscape'))
        v.addWidget(sep())

        # ---- 页面自动排版 ----
        v.addLayout(header('页面自动排版', 'layout_dark'))
        self.layout_combo = QComboBox()
        self.layout_combo.addItems(list(self.LAYOUT_MODES))
        self.layout_combo.setCurrentText(self._layout_mode)
        self.layout_combo.setFixedHeight(32)
        self.layout_combo.setStyleSheet(
            f'QComboBox {{ background: {COMBO_BG}; color: {INK};'
            ' border: none; border-radius: 4px; padding-left: 9px;'
            ' font-size: 14px; min-height: 32px; }'
            'QComboBox::drop-down { border: none; width: 26px; }'
            f'QComboBox::down-arrow {{ image: url({ICONS_DIR}/down_arrow.svg);'
            ' width: 12px; height: 12px; }'
            f'QComboBox QAbstractItemView {{ background: white; color: {INK};'
            f' selection-background-color: {BLUE}; selection-color: white; }}')
        self.layout_combo.currentTextChanged.connect(self._on_layout_mode)
        v.addWidget(self.layout_combo)
        v.addWidget(sep())

        # ---- 🏷️ 常见尺寸快捷设置 ----
        v.addLayout(header('🏷️ 常见尺寸快捷设置（需要鼠标选中图片）'))
        for text, icn, size_mm in (
                ('身份证/银行卡 (85.6 x 54)', 'pencil_dark', (85.6, 54.0)),
                ('户口本单页 (143 x 105)', 'book_dark', (143.0, 105.0)),
                ('护照相片页 (125 x 88)', 'plane_dark', (125.0, 88.0)),
                ('驾驶证/行驶证 (88 x 60)', 'car_dark', (88.0, 60.0))):
            b = QPushButton(text)
            b.setIcon(icon(icn))
            b.setIconSize(QSize(16, 16))
            b.setFixedHeight(36)
            b.setStyleSheet(gray36)
            b.clicked.connect(
                lambda _=False, wh=size_mm: self._quick_size(*wh))
            v.addWidget(b)
        frow = QHBoxLayout()
        frow.setSpacing(8)
        fill_b = QPushButton('铺满 A4 全页（保持比例）')
        fill_b.setIcon(icon('expand_dark'))
        # 与「全选图片」同行：横向内边距收紧防标签被挤出面板
        fill_b.setStyleSheet(gray36 + ' padding-left: 6px; padding-right: 6px;')
        fill_b.setIconSize(QSize(16, 16))
        fill_b.setFixedHeight(36)
        fill_b.setStyleSheet(gray36)
        fill_b.clicked.connect(self._fill_page)
        frow.addWidget(fill_b, 1)
        self.all_sel_check = QCheckBox('全选图片')
        self.all_sel_check.setStyleSheet(
            f'QCheckBox {{ color: {INK}; font-size: 14px; spacing: 6px; }}'
            'QCheckBox::indicator { width: 18px; height: 18px; }'
            f'QCheckBox::indicator:unchecked {{ image: url({ICONS_DIR}/checkbox_unchecked.svg); }}'
            f'QCheckBox::indicator:checked {{ image: url({ICONS_DIR}/checkbox_checked.svg); }}')
        self.all_sel_check.toggled.connect(self.view.set_all_sel)
        frow.addWidget(self.all_sel_check)
        v.addLayout(frow)
        v.addWidget(sep())

        # ---- 物理尺寸微调 ----
        v.addLayout(header('物理尺寸微调', 'wrench_dark'))
        edit_qss = (f'QLineEdit {{ background: white; color: {INK};'
                    ' border: 1px solid #D9D9D9; border-radius: 4px;'
                    ' padding: 0px 6px; font-size: 14px; min-height: 32px; }')
        step_qss = (f'QPushButton {{ background: {GRAY_BTN}; color: {INK};'
                    ' border: none; border-radius: 4px; min-height: 32px; }'
                    'QPushButton:hover { background: #ECECEC; }')

        def step_btn(icn):
            b = QPushButton()
            b.setIcon(icon(icn))
            b.setIconSize(QSize(14, 14))
            b.setFixedSize(28, 32)
            b.setStyleSheet(step_qss)
            return b

        def dim_row(label, edit):
            row = QHBoxLayout()
            row.setSpacing(4)
            lab = QLabel(label)
            lab.setStyleSheet(f'color: {INK}; font-size: 14px;')
            row.addWidget(lab)
            minus = step_btn('minus_dark')
            plus = step_btn('plus_dark')
            row.addWidget(minus)
            row.addWidget(edit, 1)
            row.addWidget(plus)
            return row, minus, plus

        self.w_edit = QLineEdit()
        self.h_edit = QLineEdit()
        self.scale_edit = QLineEdit('100')
        for e in (self.w_edit, self.h_edit, self.scale_edit):
            e.setStyleSheet(edit_qss)
            e.setFixedHeight(32)
        wh = QHBoxLayout()
        wh.setSpacing(10)
        wpart = QHBoxLayout()
        wpart.setSpacing(4)
        wlab = QLabel('宽：')
        wlab.setStyleSheet(f'color: {INK}; font-size: 14px;')
        wpart.addWidget(wlab)
        self.w_minus = step_btn('minus_dark')
        self.w_plus = step_btn('plus_dark')
        wpart.addWidget(self.w_minus)
        wpart.addWidget(self.w_edit, 1)
        wpart.addWidget(self.w_plus)
        wh.addLayout(wpart, 1)
        hpart = QHBoxLayout()
        hpart.setSpacing(4)
        hlab = QLabel('高：')
        hlab.setStyleSheet(f'color: {INK}; font-size: 14px;')
        hpart.addWidget(hlab)
        self.h_minus = step_btn('minus_dark')
        self.h_plus = step_btn('plus_dark')
        hpart.addWidget(self.h_minus)
        hpart.addWidget(self.h_edit, 1)
        hpart.addWidget(self.h_plus)
        wh.addLayout(hpart, 1)
        v.addLayout(wh)
        srow, self.s_minus, self.s_plus = dim_row('缩放(%)：', self.scale_edit)
        v.addLayout(srow)
        self.w_minus.clicked.connect(lambda: self._step_field('w', -1.0))
        self.w_plus.clicked.connect(lambda: self._step_field('w', 1.0))
        self.h_minus.clicked.connect(lambda: self._step_field('h', -1.0))
        self.h_plus.clicked.connect(lambda: self._step_field('h', 1.0))
        self.s_minus.clicked.connect(lambda: self._step_field('s', -5.0))
        self.s_plus.clicked.connect(lambda: self._step_field('s', 5.0))
        self.w_edit.editingFinished.connect(lambda: self._fields_edited('size'))
        self.h_edit.editingFinished.connect(lambda: self._fields_edited('size'))
        self.scale_edit.editingFinished.connect(
            lambda: self._fields_edited('scale'))
        apply_b = QPushButton('应用当前尺寸')
        apply_b.setFixedHeight(36)
        apply_b.setStyleSheet(color_qss(BLUE, BLUE_HOVER))
        apply_b.clicked.connect(self._apply_size_fields)
        v.addWidget(apply_b)
        rotate_b = QPushButton('操作图像旋转 90°')
        rotate_b.setIcon(icon('rotate_white'))
        rotate_b.setIconSize(QSize(16, 16))
        rotate_b.setFixedHeight(36)
        rotate_b.setStyleSheet(color_qss(ORANGE, ORANGE_HOVER))
        rotate_b.clicked.connect(self._rotate_targets)
        v.addWidget(rotate_b)
        v.addWidget(sep())

        # ---- 📄 文档模式增强 ----
        v.addLayout(header('📄 文档模式增强'))
        erow = QHBoxLayout()
        erow.setSpacing(14)
        self.enh_none = QRadioButton('原图')
        self.enh_bw = QRadioButton('黑白增强')
        self.enh_color = QRadioButton('彩色增强')
        self._enh_group = QButtonGroup(self)
        for r, m in ((self.enh_none, 'none'), (self.enh_bw, 'docbw'),
                     (self.enh_color, 'doccolor')):
            r.setStyleSheet(radio_qss)
            self._enh_group.addButton(r)
            erow.addWidget(r)
            r.toggled.connect(
                lambda on, mode=m: on and self._set_enhance(mode))
        self.enh_none.setChecked(True)
        erow.addStretch(1)
        v.addLayout(erow)
        v.addWidget(sep())

        # ---- 💡 提示：三行（exe 原文逐字，冒号为半角） ----
        hint = QLabel('💡 提示:\n开启自由排版后\n可通过拖拽移动选中图')
        hint.setStyleSheet(f'color: {INK}; font-size: 14px;')
        v.addWidget(hint)
        v.addStretch(1)

        area.setWidget(content)
        area.setVisible(False)
        return area

    # ----- 面板：选中同步与操作目标 -----
    def _targets(self):
        """面板操作的作用对象：全选时全部卡，否则当前选中卡；
        未选时为空（点尺寸/旋转/增强均无反应，照专节「需要鼠标
        选中图片」）。"""
        if self.view.all_sel:
            return [(pi, ci) for pi, page in enumerate(self.layout)
                    for ci in range(len(page))]
        if self.view.sel is not None:
            return [self.view.sel]
        return []

    def _sync_panel_from_sel(self):
        """选中变化时回填面板（照实测 50：宽/高回填卡的毫米尺寸、
        缩放回 100、增强单选回到该卡的当前模式）。全选时以当前
        点选卡为准（无点选时取第一张），未选时清空。"""
        if self._syncing or getattr(self, 'w_edit', None) is None:
            return
        ref = self.view.sel
        if ref is None and self.view.all_sel:
            for pi, page in enumerate(self.layout):
                if page:
                    ref = (pi, 0)
                    break
        self._syncing = True
        try:
            if ref is None:
                self._sel_base = None
                self.w_edit.clear()
                self.h_edit.clear()
                self.scale_edit.setText('100')
                self.enh_none.setChecked(True)
            else:
                img = self.layout[ref[0]][ref[1]][0]
                s = self.core.mm_to_pixel
                self._sel_base = (img.width / s, img.height / s)
                self.w_edit.setText(f'{img.width / s:.1f}')
                self.h_edit.setText(f'{img.height / s:.1f}')
                self.scale_edit.setText('100')
                mode = self._state(ref[0], ref[1])['enh']
                {('none'): self.enh_none, 'docbw': self.enh_bw,
                 'doccolor': self.enh_color}[mode].setChecked(True)
        finally:
            self._syncing = False

    # ----- 面板：卡片操作状态与重算 -----
    def _state(self, pi, ci):
        st = self._card_state.get((pi, ci))
        if st is None:
            st = {'base': self.layout[pi][ci][0], 'enh': 'none',
                  'rot': 0, 'size': None}
            self._card_state[(pi, ci)] = st
        return st

    def _enhanced(self, img, mode):
        """借 core.apply_enhancement 出图（文档黑白/文档彩色）：临时
        切 core.enhance_mode、调用、原样恢复，不改 core 本体。"""
        prev = getattr(self.core, 'enhance_mode', 'color')
        try:
            self.core.enhance_mode = mode
            return self.core.apply_enhancement(img)
        finally:
            self.core.enhance_mode = prev

    def _render_card(self, pi, ci, center_on_page=False):
        """按状态重算一张卡的显示图并写回布局：base → 旋转 → 增强
        → 缩放到目标尺寸；位置默认保持原中心，center_on_page 时
        改为页内居中（铺满全页用）。"""
        st = self._state(pi, ci)
        img = st['base']
        for _ in range(st['rot'] % 4):
            img = img.transpose(Image.ROTATE_270)   # 顺时针 90°
        if st['enh'] != 'none':
            img = self._enhanced(img, st['enh'])
        if st['size'] is not None and tuple(st['size']) != img.size:
            img = img.resize((int(st['size'][0]), int(st['size'][1])),
                             Image.Resampling.LANCZOS)
        old_img, ox, oy = self.layout[pi][ci]
        if center_on_page:
            nx = (self.page_px[0] - img.width) / 2.0
            ny = (self.page_px[1] - img.height) / 2.0
        else:
            nx = ox + old_img.width / 2.0 - img.width / 2.0
            ny = oy + old_img.height / 2.0 - img.height / 2.0
        self.layout[pi][ci] = [img, nx, ny]
        self.view.update()

    def _set_targets_size_mm(self, w_mm, h_mm):
        s = self.core.mm_to_pixel
        for pi, ci in self._targets():
            st = self._state(pi, ci)
            st['size'] = (max(1, int(round(w_mm * s))),
                          max(1, int(round(h_mm * s))))
            self._render_card(pi, ci)

    def _quick_size(self, w_mm, h_mm):
        if not self._targets():
            return
        self._set_targets_size_mm(w_mm, h_mm)
        self._sync_panel_from_sel()

    def _fill_page(self):
        """铺满 A4 全页（保持比例）：按当前显示宽高比放大到恰好
        塞进整页，并移到页内居中。"""
        for pi, ci in self._targets():
            img = self.layout[pi][ci][0]
            f = min(self.page_px[0] / float(img.width),
                    self.page_px[1] / float(img.height))
            st = self._state(pi, ci)
            st['size'] = (max(1, int(round(img.width * f))),
                          max(1, int(round(img.height * f))))
            self._render_card(pi, ci, center_on_page=True)
        self._sync_panel_from_sel()

    # ----- 面板：尺寸字段（宽/高与缩放联动） -----
    @staticmethod
    def _parse(text):
        try:
            return float(str(text).strip())
        except (TypeError, ValueError):
            return None

    def _step_field(self, which, delta):
        edit = {'w': self.w_edit, 'h': self.h_edit,
                's': self.scale_edit}[which]
        val = self._parse(edit.text())
        if val is None:
            return
        edit.setText(f'{max(0.0, val + delta):.1f}' if which != 's'
                     else f'{max(0.0, val + delta):.0f}')
        self._fields_edited('scale' if which == 's' else 'size')

    def _fields_edited(self, driver):
        """宽/高与缩放(%)联动：改宽高 → 缩放按宽比回算；改缩放 →
        宽高按基准尺寸（最近一次回填时的卡尺寸）同比例回算。"""
        if self._syncing or self._sel_base is None:
            return
        bw, bh = self._sel_base
        self._syncing = True
        try:
            if driver == 'scale':
                pct = self._parse(self.scale_edit.text())
                if pct is not None and bw > 0:
                    self.w_edit.setText(f'{bw * pct / 100.0:.1f}')
                    self.h_edit.setText(f'{bh * pct / 100.0:.1f}')
            else:
                w = self._parse(self.w_edit.text())
                if w is not None and bw > 0:
                    self.scale_edit.setText(f'{w / bw * 100.0:.0f}')
        finally:
            self._syncing = False

    def _apply_size_fields(self):
        """「应用当前尺寸」：以宽/高字段（毫米）为准改目标卡尺寸，
        保持中心；完后按新尺寸回填（缩放回 100）。"""
        if not self._targets():
            return
        w = self._parse(self.w_edit.text())
        h = self._parse(self.h_edit.text())
        if w is None or h is None or w <= 0 or h <= 0:
            self._sync_panel_from_sel()
            return
        self._set_targets_size_mm(w, h)
        self._sync_panel_from_sel()

    def _rotate_targets(self):
        """「操作图像旋转 90°」：目标卡顺时针转 90°（绕中心，宽高
        互换）；状态里的目标尺寸同步互换，后续重算不叠加误差。"""
        for pi, ci in self._targets():
            st = self._state(pi, ci)
            st['rot'] = (st['rot'] + 1) % 4
            if st['size'] is not None:
                st['size'] = (st['size'][1], st['size'][0])
            self._render_card(pi, ci)
        self._sync_panel_from_sel()

    def _set_enhance(self, mode):
        """📄 文档模式增强：原图＝回 base 原样，黑白/彩色增强＝借
        core 的文档黑白/文档彩色增强（apply_enhancement）。"""
        if self._syncing:
            return
        for pi, ci in self._targets():
            st = self._state(pi, ci)
            if st['enh'] != mode:
                st['enh'] = mode
                self._render_card(pi, ci)

    # ----- 面板：画布方向与页面自动排版 -----
    def _set_orientation(self, orient):
        """纵/横切换：页宽高互换（含页毫米数），随后按当前自动排版
        模式在新画布上重排（原版 func17 以横版标志取 PW/PH）。"""
        if orient == self.orientation:
            return
        self.orientation = orient
        self.page_px = (self.page_px[1], self.page_px[0])
        self.page_mm = (self.page_mm[1], self.page_mm[0])
        self.view.set_page_px(self.page_px[0], self.page_px[1],
                              self.page_mm)
        self._apply_auto_layout()

    def _on_layout_mode(self, text):
        self._layout_mode = text
        self._apply_auto_layout()

    def _apply_auto_layout(self):
        """页面自动排版：exe 取证四模式公式（merge-layout-geometry.md
        §2.3 与 §三，页单位 pt），按当前页宽高（毫米）逐页重排卡片
        位置；只动位置不动尺寸。混尺寸时网格/紧凑的节距取页内首卡
        的 W/H（原版公式以单一 W/H 立式；卡组同尺寸时逐项一致）。"""
        s = self.core.mm_to_pixel
        PW, PH = float(self.page_mm[0]), float(self.page_mm[1])
        pt = self.PT_MM
        mode = self._layout_mode
        for page in self.layout:
            n = len(page)
            if not n:
                continue
            sizes = [(img.width / s, img.height / s)
                     for img, _x, _y in page]
            pos = []
            if mode == '单列居中 (清单)':
                gap = 20 * pt
                total = sum(h for _w, h in sizes) + (n - 1) * gap
                y = max(gap, (PH - total) / 2.0)
                for w, h in sizes:
                    pos.append(((PW - w) / 2.0, y))
                    y += h + gap
            elif mode == '纵向紧凑 (双列)':
                W0, H0 = sizes[0]
                gap = max(10 * pt, (PW - 2 * W0) / 3.0)
                for i, (_w, _h) in enumerate(sizes):
                    col, row = i % 2, i // 2
                    pos.append((gap + col * (W0 + gap),
                                60 * pt + row * (H0 + 15 * pt)))
            elif mode == '网格均分居中':
                cols = int(np.ceil(np.sqrt(n)))
                rows = int(np.ceil(n / cols))
                cw, ch = PW / cols, PH / rows
                for i, (w, h) in enumerate(sizes):
                    col, row = i % cols, i // cols
                    pos.append((col * cw + (cw - w) / 2.0,
                                row * ch + (ch - h) / 2.0))
            else:   # 上下居中（推荐）
                if n == 1:
                    w, h = sizes[0]
                    pos.append(((PW - w) / 2.0, (PH - h) / 2.0))
                elif n == 2:
                    for i, (w, h) in enumerate(sizes):
                        frac = 0.25 if i == 0 else 0.75
                        pos.append(((PW - w) / 2.0, frac * PH - h / 2.0))
                else:
                    cols = 3 if n > 4 else 2
                    W0, H0 = sizes[0]
                    gap = max(10 * pt, (PW - cols * W0) / (cols + 1))
                    for i, (_w, _h) in enumerate(sizes):
                        col, row = i % cols, i // cols
                        pos.append((gap + col * (W0 + gap),
                                    80 * pt + row * (H0 + 40 * pt)))
            for card, (x_mm, y_mm) in zip(page, pos):
                card[1] = x_mm * s
                card[2] = y_mm * s
        self.view.update()

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
        if self.base_name:
            return self.base_name
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
        """按当前（可能拖过的）布局渲染最终页。身份证页走 core 渲染
        （含水印）；页图模式把整页图贴回同尺寸白页（无水印，照定案：
        执照页与身份证页共用本窗，页内容即处理成品本身）。"""
        if self.images_mode:
            pages = []
            for page in self.layout:
                canvas = Image.new('RGB', self.page_px, (255, 255, 255))
                for img, x, y in page:
                    canvas.paste(img, (int(round(x)), int(round(y))))
                pages.append(canvas)
            return pages
        if self.orientation == 'landscape':
            # 横版页：core.render_pages 只认纵版 A4（core 本期不许
            # 改），故横版在 app 层同口径合成——白页贴卡后按 core
            # 当前水印参数调 core 的 _apply_watermark（与纵版同参）。
            pages = []
            for page in self.layout:
                canvas = Image.new('RGB', self.page_px, (255, 255, 255))
                for img, x, y in page:
                    canvas.paste(img, (int(round(x)), int(round(y))))
                pages.append(canvas)
            if self.core.wm_enabled and self.core.wm_text.strip():
                s = self.core.mm_to_pixel
                _wm_colors = {'黑色': (0, 0, 0), '蓝色': (30, 90, 200),
                              '红色': (220, 30, 30), '灰色': (110, 110, 110)}
                rgb = _wm_colors.get(self.core.wm_color, (0, 0, 0))
                pages = [_apply_watermark(
                    p, self.core.wm_text.strip(),
                    int(self.core.wm_size * s), self.core.wm_opacity,
                    self.core.wm_rows, rgb) for p in pages]
            return pages
        return self.core.render_pages(self.layout)

    def _ask_folder(self, title):
        d = QFileDialog.getExistingDirectory(self, title, self._default_dir())
        return d or None

    def export_crops(self):
        # 命名照原版 §5：%s_Crop_P{页}_{页内序号}.jpg，%s 为该卡源图基底；
        # 页图模式由调用方注入成品项（执照整页即裁剪成品）。
        items = []
        if self.crop_items_override is not None:
            for page_items in self.crop_items_override:
                items.extend(page_items)
        elif self._src_map:
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
        # S4: 写文件全程守卫，失败弹错误框，不让异常穿出槽函数
        try:
            for name, img in items:
                img.convert('RGB').save(os.path.join(d, name), quality=95)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败：{e}')
            return
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
        try:
            for i, p in enumerate(pages):
                p.save(os.path.join(d, f'{self._page_base(i)}_IDCard_{i + 1}.jpg'),
                       quality=95, dpi=(300, 300))
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败：{e}')
            return
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
        try:
            for i, p in enumerate(pages):
                data = self.core._pages_to_pdf_bytes([p])
                with open(os.path.join(
                        d, f'{self._page_base(i)}_IDCard_{i + 1}.pdf'), 'wb') as f:
                    f.write(data)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败：{e}')
            return
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
        try:
            data = self.core._pages_to_pdf_bytes(pages)
            with open(path, 'wb') as f:
                f.write(data)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'导出失败：{e}')
            return
        self._remember(os.path.dirname(path))
        self.win.set_status(f'合并 PDF 已导出：{path}')
        report_saved(self, os.path.dirname(path),
                     f'已导出合并 PDF：{os.path.basename(path)}')

    @staticmethod
    def _calibrated(page, pct):
        """物理尺寸微调（对照表 §6/§7）：整页绕中心按 pct% 缩放贴回
        同尺寸白页；只供打印路径调用，导出文件不走这里。"""
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
        pages = self._pages()
        if not pages:
            return
        # 横版页（执照）走横向、竖版页走纵向（照页宽高比逐页设）
        printer.setOrientation(
            QPrinter.Landscape if pages[0].width > pages[0].height
            else QPrinter.Portrait)
        dlg = QPrintDialog(printer, self)
        if dlg.exec_() != QDialog.Accepted:
            return
        scale = float(getattr(self.win, 'print_scale', 100) or 100)
        if scale != 100.0:
            pages = [self._calibrated(p, scale) for p in pages]
        painter = QPainter(printer)
        for i, page in enumerate(pages):
            if i:
                printer.newPage()
            printer.setOrientation(
                QPrinter.Landscape if page.width > page.height
                else QPrinter.Portrait)
            rect = printer.pageRect()
            painter.drawImage(rect, pil_to_qimage(page))
        painter.end()
        self.win.set_status('打印任务已发送')

def main():
    app = QApplication(sys.argv)
    # 界面字体照 Merge 实测＝仿宋（MERGE-UI-MEASURE.md §2/§9：正文墨高
    # 12px ≈ 14px 字号）。Qt 按族名链回退：目标机 Win7 系统自带仿宋，
    # 本机开发用 Noto Serif CJK 顶替渲染。字号用像素口径与实测对齐。
    _fams = QFontDatabase().families()
    for fam in ('FangSong', '仿宋', 'STFangsong', 'Noto Serif CJK SC',
                'SimSun', 'Microsoft YaHei'):
        if fam in _fams:
            _f = QFont(fam)
            _f.setPixelSize(14)
            app.setFont(_f)
            break
    app.setStyleSheet(GLOBAL_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
