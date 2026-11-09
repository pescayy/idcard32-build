import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import tempfile
import subprocess
import sys
import platform
import hashlib
import img2pdf

# === 图标设置代码（由打包工具自动添加） ===
import os
import sys

def set_window_icon(window):
    """设置窗口图标 - 修复任务栏图标和角标显示"""
    try:
        if getattr(sys, 'frozen', False):
            # 打包后执行
            base_path = sys._MEIPASS
            icon_path = os.path.join(base_path, 'favicon (1).ico')
        else:
            # 开发环境执行
            icon_path = 'favicon (1).ico'
        
        if os.path.exists(icon_path):
            window.iconbitmap(icon_path)
            # 设置任务栏图标
            try:
                import ctypes
                myappid = 'mycompany.myproduct.subproduct.version'  # 任意字符串
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except:
                pass
    except Exception as e:
        print(f"设置图标失败: {e}")

# Toplevel图标设置函数
def set_toplevel_icon(toplevel_window):
    """设置Toplevel窗口图标"""
    try:
        if getattr(sys, 'frozen', False):
            # 打包后执行
            base_path = sys._MEIPASS
            icon_path = os.path.join(base_path, 'favicon (1).ico')
        else:
            # 开发环境执行
            icon_path = 'favicon (1).ico'
        
        if os.path.exists(icon_path):
            toplevel_window.iconbitmap(icon_path)
    except Exception as e:
        print(f"设置Toplevel图标失败: {e}")

# 包装Toplevel类以自动设置图标
class ToplevelWithIcon(tk.Toplevel):
    """自动设置图标的Toplevel类"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        set_toplevel_icon(self)

# === 图标设置结束 ===



# === 修复openssl md5报错 ===
# 解决Python 3.9+中hashlib.md5的usedforsecurity参数问题
try:
    # 尝试使用新的方式调用md5
    hashlib.md5(usedforsecurity=False)
except TypeError:
    # 如果不支持usedforsecurity参数，创建兼容版本
    import warnings
    warnings.filterwarnings("ignore")
    
    def md5_compat(data=b'', *, usedforsecurity=True):
        return hashlib.md5(data)
    
    # 替换hashlib.md5为兼容版本
    hashlib.md5 = md5_compat
# === 修复结束 ===

# === 图标设置代码 ===
def set_window_icon(root):
    """设置窗口图标 - 修复任务栏图标和角标显示"""
    try:
        if getattr(sys, 'frozen', False):
            # 打包后执行
            base_path = sys._MEIPASS
            icon_path = os.path.join(base_path, 'favicon (1).ico')
        else:
            # 开发环境执行
            icon_path = 'favicon (1).ico'
        
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
            # 设置任务栏图标
            try:
                import ctypes
                myappid = 'mycompany.myproduct.subproduct.version'  # 任意字符串
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except:
                pass
    except Exception as e:
        print(f"设置图标失败: {e}")

# === 图标设置结束 ===

# 尝试导入tkinterdnd2实现拖拽功能
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    DRAG_DROP_SUPPORTED = True
except ImportError:
    DRAG_DROP_SUPPORTED = False
    print("提示: tkinterdnd2库未安装，拖拽功能不可用。")
    print("请运行: pip install tkinterdnd2 来启用拖拽功能")


def _order_quad_points(pts):
    """把任意顺序点击的 4 个角点排成 [左上, 右上, 右下, 左下]。"""
    pts = [tuple(p) for p in pts]
    tl = min(pts, key=lambda p: p[0] + p[1])
    br = max(pts, key=lambda p: p[0] + p[1])
    tr = max(pts, key=lambda p: p[0] - p[1])
    bl = min(pts, key=lambda p: p[0] - p[1])
    return [tl, tr, br, bl]


def _order_points_np(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


class CardDetectorAI:
    """SCRFD 证件关键点检测（算法摘自云星球 V7.3.1，经三方对测验证）。
    用 OpenCV 自带 DNN 加载 ONNX 模型，不依赖 onnxruntime。"""

    def __init__(self, onnxmodel, confThreshold=0.5, nmsThreshold=0.5):
        self.inpWidth, self.inpHeight = 640, 640
        self.confThreshold, self.nmsThreshold = confThreshold, nmsThreshold
        self.net = cv2.dnn.readNet(onnxmodel)
        self.keep_ratio, self.fmc = True, 3
        self._feat_stride_fpn, self._num_anchors = [8, 16, 32], 4

    def resize_image(self, srcimg):
        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw,
                                         cv2.BORDER_CONSTANT, value=0)
            else:
                newh, neww = int(self.inpHeight * hw_scale) + 1, self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0,
                                         cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight))
        return img, newh, neww, padh, padw

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]; y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]; y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1]); y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1]); y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]; py = points[:, 1] + distance[:, i + 1]
            if max_shape is not None:
                px = np.clip(px, 0, max_shape[1]); py = np.clip(py, 0, max_shape[0])
            preds.append(px); preds.append(py)
        return np.stack(preds, axis=-1)

    def detect(self, srcimg):
        img, newh, neww, padh, padw = self.resize_image(srcimg)
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, (self.inpWidth, self.inpHeight),
                                     (127.5, 127.5, 127.5), swapRB=True)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outs[idx * self.fmc][0]
            bbox_preds = outs[idx * self.fmc + 1][0] * stride
            kps_preds = outs[idx * self.fmc + 2][0] * stride
            height, width = blob.shape[2] // stride, blob.shape[3] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
            pos_inds = np.where(scores >= self.confThreshold)[0]
            bboxes = self.distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds]); bboxes_list.append(bboxes[pos_inds])
            kpss = self.distance2kps(anchor_centers, kps_preds).reshape((-1, 4, 2))
            kpss_list.append(kpss[pos_inds])
        if not scores_list:
            return None
        scores = np.vstack(scores_list).ravel(); bboxes = np.vstack(bboxes_list); kpss = np.vstack(kpss_list)
        if len(scores) == 0:
            return None
        bboxes[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]
        ratioh, ratiow = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        bboxes[:, 0] = (bboxes[:, 0] - padw) * ratiow; bboxes[:, 1] = (bboxes[:, 1] - padh) * ratioh
        bboxes[:, 2] = bboxes[:, 2] * ratiow; bboxes[:, 3] = bboxes[:, 3] * ratioh
        kpss[:, :, 0] = (kpss[:, :, 0] - padw) * ratiow; kpss[:, :, 1] = (kpss[:, :, 1] - padh) * ratioh
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold)
        if len(indices) == 0:
            return None
        indices = np.array(indices).flatten()
        best, best_score = indices[0], scores[indices[0]]
        for i in indices:
            if scores[i] > best_score:
                best_score, best = scores[i], i
        pts = kpss[best]
        pts_list = [(float(pts[i, 0]), float(pts[i, 1])) for i in range(4)]
        pts_ordered = _order_points_np(np.array(pts_list))
        return [(float(q[0]), float(q[1])) for q in pts_ordered]


def _enhance_idcard_bw(pil_img):
    """身份证黑白增强：BT.601 灰度 + S 曲线 LUT（陡度 k=4，鱼还原配方）"""
    arr = np.asarray(pil_img.convert('RGB')).astype(np.float32)
    gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    i = np.arange(256, dtype=np.float32)
    x = i / 255.0 - 0.5
    t = 4.0 * x
    f = (np.exp(t) - np.exp(-t)) / (np.exp(t) + np.exp(-t) + 1.0)
    lut = np.clip((f + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    out = lut[np.clip(gray, 0, 255).astype(np.uint8)]
    return Image.fromarray(np.stack([out] * 3, axis=-1), 'RGB')


def _enhance_idcard_color(pil_img):
    """身份证彩色增强：灰世界白平衡，只统计通道值在 (30,225] 的像素，增益 clamp 到 [1.15, 30]（鱼还原配方）"""
    arr = np.asarray(pil_img.convert('RGB')).astype(np.float32)
    means = []
    for c in range(3):
        ch = arr[..., c]
        sel = ch[(ch > 30) & (ch <= 225)]
        means.append(float(sel.mean()) if sel.size else float(ch.mean()))
    gray_mean = sum(means) / 3.0
    out = arr.copy()
    for c in range(3):
        if means[c] > 0:
            gain = min(max(gray_mean / means[c], 1.15), 30.0)
            out[..., c] = np.clip(arr[..., c] * gain, 0, 255)
    return Image.fromarray(out.astype(np.uint8), 'RGB')


def _doc_three_band(v):
    """文档模式三段处理：<=215 压黑，>=245 提纯白，中间线性拉伸（鱼还原的 215/245/255 阈值）"""
    return np.clip((v - 215.0) / 30.0, 0.0, 1.0) * 255.0


def _enhance_document_bw(pil_img):
    """文档黑白：÷4 降采样求亮度均值 → 对比拉伸 → Gamma 2.5 → 三段阈值（鱼还原配方）"""
    arr = np.asarray(pil_img.convert('L')).astype(np.float32)
    h, w = arr.shape
    small = np.asarray(pil_img.convert('L').resize((max(1, w // 4), max(1, h // 4)))).astype(np.float32)
    mean = float(small.mean()) / 255.0
    x = arr / 255.0
    denom = 1.0 - 0.8 * mean
    if abs(denom) > 1e-6:
        x = (x - 0.8 * mean) / denom
    x = np.clip(x, 0.0, 1.0) ** 2.5
    out = _doc_three_band(x * 255.0).astype(np.uint8)
    return Image.fromarray(np.stack([out] * 3, axis=-1), 'RGB')


def _enhance_document_color(pil_img):
    """文档彩色：亮度 ×1.2 → 对比拉伸 ×2.2 → Gamma 4 → 三段阈值（鱼还原配方）"""
    arr = np.asarray(pil_img.convert('RGB')).astype(np.float32) / 255.0
    x = np.clip(arr * 1.2, 0.0, 1.0)
    x = np.clip((x - 0.5) * 2.2 + 0.5, 0.0, 1.0)
    x = x ** 4.0
    out = _doc_three_band(x * 255.0).astype(np.uint8)
    return Image.fromarray(out, 'RGB')


def _load_cjk_font(px):
    for p in (r'C:\Windows\Fonts\msyh.ttc', r'C:\Windows\Fonts\simhei.ttf',
              r'C:\Windows\Fonts\simsun.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
              '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'):
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, px)
        except Exception:
            pass
    return ImageFont.load_default()


def _apply_watermark(page_img, text, font_px, opacity_pct, rows):
    """整页平铺斜向水印（对齐 IDCardMerge 的安全水印：文字/字号/透明度/行数）"""
    base = page_img.convert('RGBA')
    W, H = base.size
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_cjk_font(font_px)
    alpha = max(0, min(255, int(opacity_pct * 255 / 100)))
    color = (110, 110, 110, alpha)
    try:
        tw = draw.textlength(text, font=font)
    except Exception:
        tw = font_px * len(text)
    step_x = int(tw + font_px * 2)
    rows = max(1, rows)
    for r in range(rows):
        y = int((r + 0.5) * H / rows - font_px / 2)
        x = -int((r % 2) * step_x / 2)
        while x < W:
            draw.text((x, y), text, font=font, fill=color)
            x += step_x
    overlay = overlay.rotate(30, resample=Image.Resampling.BICUBIC, center=(W // 2, H // 2))
    return Image.alpha_composite(base, overlay).convert('RGB')


class IDCardProcessor:
    def __init__(self, root):
        self.root = root
        try:
            tk._default_root = root
        except Exception:
            pass
        self.root.title("证件图片处理器·开发版（AI批量） by xm")
        self.enhance_mode = "color"
        self.enhance_var = None
        
        # 窗口默认800x900，但允许手动调整大小
        self.root.geometry("1200x780")
        
        # 将窗口居中显示
        self.center_window(self.root)
        
        # 初始化变量
        self.front_image = None
        self.back_image = None
        self.card_image = None  # 银行卡图片
        self.front_points = []
        self.back_points = []
        # 批量证件组（对齐 IDCardMerge 批量格子）
        self.pairs = []
        self.current_open = None
        self._pair_photos = {}
        self.card_points = []   # 银行卡角点
        self.front_processed = None
        self.back_processed = None
        self.card_processed = None  # 银行卡处理结果
        
        # 标记状态
        self.is_marking = False
        self.current_marking_side = None
        
        # 十字线相关
        self.crosshair_lines = []
        
        # 身份证标准尺寸 (mm)
        self.id_width = 85.6
        self.id_height = 54.0
        
        # 银行卡标准尺寸 (mm) - 与身份证相同
        self.card_width = 85.6
        self.card_height = 54.0
        
        self.dpi = 300
        self.mm_to_pixel = self.dpi / 25.4
        
        # A4尺寸 (mm)
        self.a4_width = 210
        self.a4_height = 297
        
        # 固定外圆角半径 (3.18mm)
        self.corner_radius_mm = 3.18
        self.corner_radius = int(self.corner_radius_mm * self.mm_to_pixel)
        
        # 拖拽相关变量
        self.drag_over_side = None
        self.hover_timer = None
        
        # 文本编辑相关
        self.text_edit_mode = False
        self.text_objects = []
        self.dragging_text = None
        self.text_start_x = 0
        self.text_start_y = 0
        
        # 调整点模式相关
        self.adjust_mode = False
        self.adjust_points = []
        self.dragging_point_index = None
        self.point_circles = []  # 点的图形对象
        
        # 工作区域预览点击相关
        self.preview_click_areas = {}  # 存储预览区域的canvas对象ID
        self.preview_images = {}  # 存储预览图像的引用
        
        # 按钮引用
        self.finish_marking_btn = None
        self.finish_adjust_btn = None
        
        # 背景设置（默认白色背景，不再询问）
        self.background_color = "white"  # 默认白色背景
        self.use_transparent = False  # 默认不使用透明背景
        
        # 创建GUI
        self.setup_ui()
        
        # 初始状态
        self.update_status("就绪 - 请点击或拖拽图片到框内上传")
        
    def center_window(self, window):
        """将窗口居中显示"""
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        if width == 1 and height == 1:  # 如果窗口还未创建，使用默认尺寸
            width = 800
            height = 900
        
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        window.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_ui(self):
        # ===== UI 换脸（对齐 IDCardMerge 版式）：白底、顶栏、提示行、状态行 =====
        try:
            self.root.configure(bg='white')
        except Exception:
            pass
        try:
            self.root.option_add('*Font', ('Microsoft YaHei', 10))
        except Exception:
            pass
        # 主框架
        main_frame = tk.Frame(self.root, bg='#F5F6F8')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 顶栏：左侧蓝色「批量导入」，右侧「清空数据」
        top_bar = tk.Frame(main_frame, bg='#F5F6F8')
        top_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))
        tk.Button(top_bar, text="批量导入", command=self.batch_import,
                  bg='#2F7CF6', fg='white', activebackground='#2F7CF6',
                  activeforeground='white', relief=tk.FLAT,
                  font=("Arial", 11, "bold"), padx=18, pady=6,
                  cursor='hand2').pack(side=tk.LEFT)
        tk.Button(top_bar, text="清空数据", command=self._clear_pairs,
                  bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                  relief=tk.FLAT, font=("Arial", 10),
                  padx=14, pady=6, cursor='hand2').pack(side=tk.RIGHT)

        # 居中灰色提示行
        tk.Label(main_frame,
                 text='请分别添加证件正面和背面，支持批量导入。点已处理图片可微调角点，每组可一键互换正反面。',
                 bg='#F5F6F8', fg='#888888',
                 font=("Arial", 10)).pack(side=tk.TOP, pady=(0, 8))

        # 状态标签：窗口最底部居中（先于其它底部控件 pack，占住最底行）
        self.status_var = tk.StringVar(value="就绪")
        tk.Label(main_frame, textvariable=self.status_var,
                 bg='#F5F6F8', fg='#888888', font=("Arial", 10),
                 anchor=tk.CENTER).pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))

        # 银行卡功能按钮：保留对象引用（clear_all 等代码会引用它），不再摆出
        self.card_mode_btn = ttk.Button(main_frame, text="银行卡功能",
                                        command=self.toggle_card_mode, width=12)

        # 增强模式变量（选择控件移到输出设置第一行的下拉框）
        self.enhance_var = tk.StringVar(value="color")

        # 批量证件组面板（主区域）
        self.build_pairs_panel(main_frame)
        self.build_settings_bar(main_frame)

        # A4预览区域 - 主工作区：换脸后不再显示。
        # 画布对象保留创建，供 draw_a4_layout 等旧代码引用；不 pack、不绑
        # Configure、不注册画布级拖拽（格子上的拖拽保留）。
        self.a4_frame = ttk.LabelFrame(main_frame, text="工作区域", padding=10)
        self.a4_canvas = tk.Canvas(self.a4_frame, bg='white', highlightthickness=0)

        # 银行卡模式标志
        self.card_mode = False

        # 标记窗口相关
        self.mark_window = None
        self.mark_canvas = None
        self.mark_scrollbar_x = None
        self.mark_scrollbar_y = None

        # 文本编辑相关
        self.text_window = None
        self.text_entry = None

        # 初始布局标记（画布已隐藏，跳过首绘）
        self.is_in_edit_mode = True
    
    def toggle_card_mode(self):
        """切换银行卡模式（仅切换显示，不清除身份证图片）"""
        self.card_mode = not self.card_mode
        if self.card_mode:
            self.card_mode_btn.config(text="返回身份证模式")
            self.update_status("已切换到银行卡模式 - 可在同一A4排版中添加银行卡")
        else:
            self.card_mode_btn.config(text="银行卡功能")
            self.update_status("已切换到身份证模式")
        
        # 仅重新绘制布局，不清除身份证图片
        self.draw_a4_layout()
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
        _                         ____                    _ 
        | |     _    _           | __ )        __ _      (_)
  _    | |     | |  | |           |  _ \       / _` |       | |
 | |_ _| |    | |__| |          | |_) |      | (_| |      | |
  \___/      \__,_|          |____/      \__,_|     |_|


                                                                        
        欢迎使用 身份证图片处理器V3.0  -By:JuBai 公众号：云雪星球
        操作指南：
        
        1. 上传图片：
           - 点击人像面/国徽面/银行卡区域上传图片
           - 或直接将图片拖拽到对应区域
        
        2. 标记角点（两种方式）：
           - 手动标记：在标记窗口中依次点击4个角点（左上→右上→右下→左下）
           - 自动识别：点击"自动识别"按钮，系统将尝试自动检测证件边缘
        
        3. 调整角点：
           - 如果对自动识别的点不满意，可以进入调整模式
           - 在调整模式中，可以拖动四个角点微调位置
        
        4. 快速操作：
           - 标记完4个点后按回车键可以完成标记
           - 在工作区域单击预览画面可以再次进入调整模式
        
        5. 生成A4排版：
           - 处理完图片后点击"生成A4排版"
           - 默认使用白色背景，可在预览页面切换透明背景
        
        6. 保存文件：
           - 保存A4图片：支持PNG/JPG格式
           - 保存PDF：高质量PDF格式
           - 单独保存：可单独保存人像面/国徽面/银行卡
        
        7. 编辑文字：
           - 在A4预览页面可添加文字
           - 可拖动文字调整位置
        
        8. 其他功能：
           - 重置所有：清空所有数据
           - 银行卡功能：切换到银行卡模式，可在同一A4排版中添加银行卡
           - 打印：直接打印A4图片
        
        注意：身份证和银行卡外圆角均为透明背景
        """
        messagebox.showinfo("操作指南", help_text)
    
    def on_canvas_resize(self, event):
        """画布大小变化时的处理"""
        if hasattr(self, 'is_in_edit_mode') and self.is_in_edit_mode:
            self.draw_a4_layout()
    
    def setup_drag_drop(self):
        """设置拖拽功能"""
        try:
            # 注册拖拽目标
            self.a4_canvas.drop_target_register(DND_FILES)
            
            # 绑定拖拽事件
            self.a4_canvas.dnd_bind('<<DropEnter>>', self.on_drag_enter)
            self.a4_canvas.dnd_bind('<<DropLeave>>', self.on_drag_leave)
            self.a4_canvas.dnd_bind('<<Drop>>', self.on_drop)
            
            print("拖拽功能已启用")
        except Exception as e:
            print(f"拖拽功能设置失败: {e}")
    
    def on_drag_enter(self, event):
        """拖拽进入事件"""
        self.a4_canvas.config(cursor="hand2")
        self.update_status("拖拽文件到对应区域")
    
    def on_drag_leave(self, event):
        """拖拽离开事件"""
        self.a4_canvas.config(cursor="")
        self.update_status("就绪")
        
        # 清除悬停效果
        if self.drag_over_side == "front":
            self.a4_canvas.itemconfig(self.front_preview_area, outline="#BBDEFB", width=1)
        elif self.drag_over_side == "back":
            self.a4_canvas.itemconfig(self.back_preview_area, outline="#F8BBD0", width=1)
        elif self.drag_over_side == "card":
            self.a4_canvas.itemconfig(self.card_preview_area, outline="#C8E6C9", width=1)
        self.drag_over_side = None
        
        # 取消悬停定时器
        if self.hover_timer:
            self.root.after_cancel(self.hover_timer)
            self.hover_timer = None
    
    def on_drop(self, event):
        """处理拖拽释放事件"""
        try:
            # 恢复光标
            self.a4_canvas.config(cursor="")
            
            # 获取拖拽的文件路径
            file_path = event.data
            
            # 处理可能的格式（去除花括号）
            if isinstance(file_path, str):
                file_path = file_path.strip('{}')
            
            if file_path:
                # 检查文件类型
                valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
                file_ext = os.path.splitext(file_path.lower())[1]
                
                if file_ext in valid_extensions:
                    # 获取鼠标位置（相对于画布）
                    x = self.a4_canvas.winfo_pointerx() - self.a4_canvas.winfo_rootx()
                    y = self.a4_canvas.winfo_pointery() - self.a4_canvas.winfo_rooty()
                    
                    # 获取区域信息
                    info = self.canvas_info
                    
                    if not self.card_mode:
                        # 身份证模式
                        # 检查是否在人像面区域
                        if (info.get('margin_x', 0) <= x <= info.get('margin_x', 0) + info.get('id_width', 0) and
                            info.get('front_y', 0) <= y <= info.get('front_y', 0) + info.get('id_height', 0)):
                            self.process_dropped_file(file_path, "front")
                        
                        # 检查是否在国徽面区域
                        elif (info.get('margin_x', 0) <= x <= info.get('margin_x', 0) + info.get('id_width', 0) and
                              info.get('back_y', 0) <= y <= info.get('back_y', 0) + info.get('id_height', 0)):
                            self.process_dropped_file(file_path, "back")
                        else:
                            messagebox.showinfo("提示", "请将图片拖拽到人像面或国徽面区域")
                    else:
                        # 银行卡模式 - 检查是否在银行卡区域
                        if (info.get('card_margin_x', 0) <= x <= info.get('card_margin_x', 0) + info.get('card_width', 0) and
                            info.get('card_y', 0) <= y <= info.get('card_y', 0) + info.get('card_height', 0)):
                            self.process_dropped_file(file_path, "card")
                        else:
                            messagebox.showinfo("提示", "请将图片拖拽到银行卡区域")
                else:
                    messagebox.showerror("错误", "不支持的文件类型，请上传图片文件")
        
        except Exception as e:
            messagebox.showerror("错误", f"处理拖拽文件失败: {str(e)}")
            print(f"拖拽错误详情: {e}")
        
        # 清除悬停效果
        if self.drag_over_side == "front":
            self.a4_canvas.itemconfig(self.front_preview_area, outline="#BBDEFB", width=1)
        elif self.drag_over_side == "back":
            self.a4_canvas.itemconfig(self.back_preview_area, outline="#F8BBD0", width=1)
        elif self.drag_over_side == "card":
            self.a4_canvas.itemconfig(self.card_preview_area, outline="#C8E6C9", width=1)
        self.drag_over_side = None
        
        # 取消悬停定时器
        if self.hover_timer:
            self.root.after_cancel(self.hover_timer)
            self.hover_timer = None
    
    def process_dropped_file(self, file_path, side):
        """处理拖拽的文件"""
        try:
            # 加载图片
            image = Image.open(file_path)
            
            # 限制图片大小
            max_size = (1200, 1200)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 在A4预览区域显示缩略图
            self.show_preview_image(side, image)
            
            # 显示原始图片，等待用户标记
            self.show_image_for_marking(side, image, file_path)
            
            if side == "front":
                side_name = "身份证人像面"
            elif side == "back":
                side_name = "身份证国徽面"
            else:
                side_name = "银行卡"
                
            self.update_status(f"已通过拖拽加载{side_name}图片，请开始标记")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
    
    def draw_a4_layout(self):
        """绘制A4布局（人像面、国徽面和银行卡区域）"""
        self.a4_canvas.delete("all")
        self.is_in_edit_mode = True
        
        canvas_width = self.a4_canvas.winfo_width()
        canvas_height = self.a4_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 780
            canvas_height = 700
        
        # 计算可用高度
        available_height = canvas_height - 150
        
        if self.card_mode:
            # 银行卡模式：显示人像面、国徽面和银行卡三个区域
            # 每个区域高度
            area_height = available_height / 3
            
            # 计算宽度（保持证件比例）
            id_width = area_height * (self.id_width / self.id_height)
            card_width = area_height * (self.card_width / self.card_height)
            
            # 统一宽度（取较小的那个）
            common_width = min(id_width, card_width)
            
            # 如果宽度太大，则按宽度限制重新计算
            max_width = canvas_width * 0.7
            if common_width > max_width:
                common_width = max_width
                area_height = common_width * (self.id_height / self.id_width)
            
            # 计算位置（居中）
            margin_x = (canvas_width - common_width) / 2
            
            # 计算三个区域的位置
            front_y = (canvas_height - available_height) / 2
            back_y = front_y + area_height + 30
            card_y = back_y + area_height + 30
            
            # 人像面区域 - 实线边框
            self.front_rect = self.a4_canvas.create_rectangle(
                margin_x, front_y,
                margin_x + common_width, front_y + area_height,
                outline="#1E88E5", width=2
            )
            
            # 人像面标签
            self.front_label = self.a4_canvas.create_text(
                margin_x + common_width/2, max(front_y - 20, 20),
                text="身份证人像面", fill="#1E88E5", font=("Arial", 12, "bold")
            )
            
            # 人像面预览区域
            front_preview_y = front_y + 15
            front_preview_height = area_height - 30
            
            self.front_preview_area = self.a4_canvas.create_rectangle(
                margin_x + 15, front_preview_y,
                margin_x + common_width - 15, front_preview_y + front_preview_height,
                outline="#BBDEFB", width=1, fill="#F5F5F5"
            )
            
            self.front_preview_text = self.a4_canvas.create_text(
                margin_x + common_width/2, front_preview_y + front_preview_height/2,
                text="点击或拖拽上传人像面图片", fill="#9E9E9E", font=("Arial", 10)
            )
            
            # 为人像面预览区域添加点击事件
            self.a4_canvas.tag_bind(self.front_preview_area, "<Button-1>", 
                                   lambda e: self.load_image("front"))
            
            # 国徽面区域 - 实线边框
            self.back_rect = self.a4_canvas.create_rectangle(
                margin_x, back_y,
                margin_x + common_width, back_y + area_height,
                outline="#D81B60", width=2
            )
            
            # 国徽面标签
            self.back_label = self.a4_canvas.create_text(
                margin_x + common_width/2, max(back_y - 20, front_y + area_height + 10),
                text="身份证国徽面", fill="#D81B60", font=("Arial", 12, "bold")
            )
            
            # 国徽面预览区域
            back_preview_y = back_y + 15
            back_preview_height = area_height - 30
            
            self.back_preview_area = self.a4_canvas.create_rectangle(
                margin_x + 15, back_preview_y,
                margin_x + common_width - 15, back_preview_y + back_preview_height,
                outline="#F8BBD0", width=1, fill="#F5F5F5"
            )
            
            self.back_preview_text = self.a4_canvas.create_text(
                margin_x + common_width/2, back_preview_y + back_preview_height/2,
                text="点击或拖拽上传国徽面图片", fill="#9E9E9E", font=("Arial", 10)
            )
            
            # 为国徽面预览区域添加点击事件
            self.a4_canvas.tag_bind(self.back_preview_area, "<Button-1>", 
                                   lambda e: self.load_image("back"))
            
            # 银行卡区域 - 实线边框
            self.card_rect = self.a4_canvas.create_rectangle(
                margin_x, card_y,
                margin_x + common_width, card_y + area_height,
                outline="#4CAF50", width=2
            )
            
            # 银行卡标签
            self.card_label = self.a4_canvas.create_text(
                margin_x + common_width/2, max(card_y - 20, back_y + area_height + 10),
                text="银行卡", fill="#4CAF50", font=("Arial", 12, "bold")
            )
            
            # 银行卡预览区域
            card_preview_y = card_y + 15
            card_preview_height = area_height - 30
            
            self.card_preview_area = self.a4_canvas.create_rectangle(
                margin_x + 15, card_preview_y,
                margin_x + common_width - 15, card_preview_y + card_preview_height,
                outline="#C8E6C9", width=1, fill="#F5F5F5"
            )
            
            self.card_preview_text = self.a4_canvas.create_text(
                margin_x + common_width/2, card_preview_y + card_preview_height/2,
                text="点击或拖拽上传银行卡图片", fill="#9E9E9E", font=("Arial", 10)
            )
            
            # 为银行卡预览区域添加点击事件
            self.a4_canvas.tag_bind(self.card_preview_area, "<Button-1>", 
                                   lambda e: self.load_image("card"))
            
            # 保存区域信息
            self.canvas_info = {
                'width': canvas_width,
                'height': canvas_height,
                'id_width': common_width,
                'id_height': area_height,
                'card_width': common_width,
                'card_height': area_height,
                'margin_x': margin_x,
                'card_margin_x': margin_x,  # 银行卡区域使用相同的margin_x
                'front_y': front_y,
                'back_y': back_y,
                'card_y': card_y,
                'front_preview_y': front_preview_y,
                'front_preview_height': front_preview_height,
                'back_preview_y': back_preview_y,
                'back_preview_height': back_preview_height,
                'card_preview_y': card_preview_y,
                'card_preview_height': card_preview_height,
                'canvas_width': canvas_width,
                'canvas_height': canvas_height
            }
            
            # 如果已有处理过的图片，重新显示
            if hasattr(self, 'front_processed') and self.front_processed is not None:
                self.show_processed_result("front", self.front_processed)
            
            if hasattr(self, 'back_processed') and self.back_processed is not None:
                self.show_processed_result("back", self.back_processed)
                
            if hasattr(self, 'card_processed') and self.card_processed is not None:
                self.show_processed_result("card", self.card_processed)
                
        else:
            # 身份证模式：只显示人像面和国徽面两个区域
            # 计算每个身份证区域的高度（均分可用高度）
            id_area_height = available_height / 2
            
            # 计算身份证宽度（保持身份证比例）
            id_width = id_area_height * (self.id_width / self.id_height)
            
            # 如果宽度太大，则按宽度限制重新计算
            max_width = canvas_width * 0.7
            if id_width > max_width:
                id_width = max_width
                id_area_height = id_width * (self.id_height / self.id_width)
            
            # 计算位置（居中，上下均分）
            margin_x = (canvas_width - id_width) / 2
            
            # 计算人像面和国徽面的位置
            front_y = (canvas_height - available_height) / 2
            back_y = front_y + id_area_height + 30
            
            # 人像面区域 - 实线边框
            self.front_rect = self.a4_canvas.create_rectangle(
                margin_x, front_y,
                margin_x + id_width, front_y + id_area_height,
                outline="#1E88E5", width=2
            )
            
            # 人像面标签
            self.front_label = self.a4_canvas.create_text(
                margin_x + id_width/2, max(front_y - 20, 20),
                text="身份证人像面", fill="#1E88E5", font=("Arial", 12, "bold")
            )
            
            # 人像面预览区域
            front_preview_y = front_y + 15
            front_preview_height = id_area_height - 30
            
            self.front_preview_area = self.a4_canvas.create_rectangle(
                margin_x + 15, front_preview_y,
                margin_x + id_width - 15, front_preview_y + front_preview_height,
                outline="#BBDEFB", width=1, fill="#F5F5F5"
            )
            
            self.front_preview_text = self.a4_canvas.create_text(
                margin_x + id_width/2, front_preview_y + front_preview_height/2,
                text="点击或拖拽上传人像面图片", fill="#9E9E9E", font=("Arial", 10)
            )
            
            # 为人像面预览区域添加点击事件
            self.a4_canvas.tag_bind(self.front_preview_area, "<Button-1>", 
                                   lambda e: self.load_image("front"))
            
            # 国徽面区域 - 实线边框
            self.back_rect = self.a4_canvas.create_rectangle(
                margin_x, back_y,
                margin_x + id_width, back_y + id_area_height,
                outline="#D81B60", width=2
            )
            
            # 国徽面标签
            self.back_label = self.a4_canvas.create_text(
                margin_x + id_width/2, max(back_y - 20, front_y + id_area_height + 10),
                text="身份证国徽面", fill="#D81B60", font=("Arial", 12, "bold")
            )
            
            # 国徽面预览区域
            back_preview_y = back_y + 15
            back_preview_height = id_area_height - 30
            
            self.back_preview_area = self.a4_canvas.create_rectangle(
                margin_x + 15, back_preview_y,
                margin_x + id_width - 15, back_preview_y + back_preview_height,
                outline="#F8BBD0", width=1, fill="#F5F5F5"
            )
            
            self.back_preview_text = self.a4_canvas.create_text(
                margin_x + id_width/2, back_preview_y + back_preview_height/2,
                text="点击或拖拽上传国徽面图片", fill="#9E9E9E", font=("Arial", 10)
            )
            
            # 为国徽面预览区域添加点击事件
            self.a4_canvas.tag_bind(self.back_preview_area, "<Button-1>", 
                                   lambda e: self.load_image("back"))
            
            # 保存区域信息供后续使用
            self.canvas_info = {
                'width': canvas_width,
                'height': canvas_height,
                'id_width': id_width,
                'id_height': id_area_height,
                'margin_x': margin_x,
                'front_y': front_y,
                'back_y': back_y,
                'front_preview_y': front_preview_y,
                'front_preview_height': front_preview_height,
                'back_preview_y': back_preview_y,
                'back_preview_height': back_preview_height,
                'canvas_width': canvas_width,
                'canvas_height': canvas_height
            }
            
            # 如果已有处理过的图片，重新显示
            if hasattr(self, 'front_processed') and self.front_processed is not None:
                self.show_processed_result("front", self.front_processed)
            
            if hasattr(self, 'back_processed') and self.back_processed is not None:
                self.show_processed_result("back", self.back_processed)
    
    def _new_slot(self):
        return {'image': None, 'points': [], 'processed': None, 'path': None}

    def _stash_workspace(self):
        idx = getattr(self, 'current_open', None)
        if idx is None or idx >= len(self.pairs):
            return
        pair = self.pairs[idx]
        for side in ('front', 'back'):
            pair[side]['image'] = getattr(self, side + '_image', None)
            pts = getattr(self, side + '_points', []) or []
            pair[side]['points'] = list(pts)
            pair[side]['processed'] = getattr(self, side + '_processed', None)

    def _load_pair_to_workspace(self, idx):
        pair = self.pairs[idx]
        for side in ('front', 'back'):
            setattr(self, side + '_image', pair[side]['image'])
            setattr(self, side + '_points', list(pair[side]['points']))
            setattr(self, side + '_processed', pair[side]['processed'])
        self.current_open = idx
        try:
            self.draw_a4_layout()
        except Exception:
            pass
        for side in ('front', 'back'):
            if pair[side]['processed'] is not None:
                try:
                    self.show_processed_result(side, pair[side]['processed'])
                except Exception:
                    pass

    def open_pair(self, idx):
        if idx < 0 or idx >= len(self.pairs):
            return
        self._stash_workspace()
        self._load_pair_to_workspace(idx)
        self.refresh_pairs_panel()

    def add_pair(self):
        self._stash_workspace()
        self.pairs.append({'front': self._new_slot(), 'back': self._new_slot()})
        idx = len(self.pairs) - 1
        self._load_pair_to_workspace(idx)
        self.refresh_pairs_panel()
        return idx

    def delete_pair(self, idx):
        if 0 <= idx < len(self.pairs):
            del self.pairs[idx]
            if self.current_open is not None and self.current_open >= idx:
                self.current_open = None
            if self.pairs:
                self._load_pair_to_workspace(min(idx, len(self.pairs) - 1))
            else:
                self.current_open = None
            self.refresh_pairs_panel()

    def swap_pair_sides(self, idx):
        """正反面互换（青点名要的功能）：整组数据对调，含原图/角点/处理结果"""
        if 0 <= idx < len(self.pairs):
            pair = self.pairs[idx]
            pair['front'], pair['back'] = pair['back'], pair['front']
            if self.current_open == idx:
                self._load_pair_to_workspace(idx)
            self.refresh_pairs_panel()
            self.update_status('已互换正反面')

    def import_into_slot(self, idx, side, file_path=None):
        self.open_pair(idx)
        if not file_path:
            file_path = filedialog.askopenfilename(
                title='选择图片',
                filetypes=[('图像文件', '*.jpg *.jpeg *.png *.bmp *.tiff'), ('所有文件', '*.*')])
        if not file_path:
            return
        try:
            image = Image.open(file_path)
            image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            self.pairs[idx][side]['path'] = file_path
            self.show_image_for_marking(side, image, file_path)
        except Exception as e:
            messagebox.showerror('错误', f'加载图片失败: {str(e)}')

    def batch_import(self):
        """批量导入：按选择顺序两张一组（先正面后反面），逐张自动识别处理"""
        files = filedialog.askopenfilenames(
            title='批量选择证件照片（按正反面顺序）',
            filetypes=[('图像文件', '*.jpg *.jpeg *.png *.bmp *.tiff'), ('所有文件', '*.*')])
        if not files:
            return
        ok, manual = 0, 0
        bad = []
        # 启动时自带的那个空组先撤掉，免得导入完平白多出一个空组
        if len(self.pairs) == 1 and \
                self.pairs[0]['front']['image'] is None and \
                self.pairs[0]['back']['image'] is None:
            self.pairs = []
            self.current_open = None
        for start in range(0, len(files), 2):
            idx = self.add_pair()
            for off, side in enumerate(('front', 'back')):
                if start + off >= len(files):
                    break
                path = files[start + off]
                try:
                    image = Image.open(path)
                    image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                except Exception:
                    bad.append(path.replace('\\', '/').split('/')[-1])
                    continue
                self.pairs[idx][side]['path'] = path
                setattr(self, side + '_image', image)
                setattr(self, side + '_points', [])
                corners = None
                try:
                    corners = self.ai_detect_corners(image)
                    if not corners and not getattr(self, '_ai_ran', False):
                        corners = self.detect_card_corners(image)
                except Exception:
                    corners = None
                if corners:
                    setattr(self, side + '_points', corners)
                    self.finish_marking(side)
                    ok += 1
                else:
                    manual += 1
            self._stash_workspace()
        self.refresh_pairs_panel()
        _msg = f'批量导入完成：自动处理 {ok} 张，待手动标角 {manual} 张'
        if bad:
            _msg += f"，另有 {len(bad)} 张读不了：{'、'.join(bad[:3])}"
        self.update_status(_msg)

    def on_slot_click(self, idx, side):
        slot = self.pairs[idx][side]
        if slot['image'] is None:
            self.import_into_slot(idx, side)
        elif slot['processed'] is not None:
            self.open_pair(idx)
            self.reopen_for_adjustment(side)
        else:
            self.open_pair(idx)
            self.show_image_for_marking(side, slot['image'], slot['path'])

    def build_pairs_panel(self, parent):
        # 换脸：无边框白底，组列表占满主区域
        frame = tk.Frame(parent, bg='white')
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 6))
        wrap = tk.Frame(frame, bg='white')
        wrap.pack(fill=tk.BOTH, expand=True)
        self.pairs_canvas = tk.Canvas(wrap, bg='white', highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=self.pairs_canvas.yview)
        self.pairs_canvas.configure(yscrollcommand=sb.set)
        self.pairs_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.pairs_list_frame = tk.Frame(self.pairs_canvas, bg='white')
        self._pairs_win = self.pairs_canvas.create_window((0, 0), window=self.pairs_list_frame, anchor='nw')
        self.pairs_list_frame.bind('<Configure>', lambda e: self.pairs_canvas.configure(scrollregion=self.pairs_canvas.bbox('all')))
        self.pairs_canvas.bind('<Configure>', lambda e: self.pairs_canvas.itemconfigure(self._pairs_win, width=e.width))
        self.pairs_count_var = tk.StringVar(value='共 0 组')
        self.pairs.append({'front': self._new_slot(), 'back': self._new_slot()})
        self.current_open = 0
        self.refresh_pairs_panel()

    def _clear_pairs(self):
        self.pairs = []
        self.current_open = None
        self._pair_photos = {}
        for side in ('front', 'back'):
            setattr(self, side + '_image', None)
            setattr(self, side + '_points', [])
            setattr(self, side + '_processed', None)
        self.add_pair()
        try:
            self.draw_a4_layout()
        except Exception:
            pass

    def refresh_pairs_panel(self):
        if not hasattr(self, 'pairs_list_frame'):
            return
        for child in self.pairs_list_frame.winfo_children():
            child.destroy()
        self._pair_photos = {}
        if hasattr(self, 'pairs_count_var'):
            self.pairs_count_var.set(f'共 {len(self.pairs)} 组')
        for idx, pair in enumerate(self.pairs):
            row = tk.Frame(self.pairs_list_frame, bg='white')
            row.pack(fill=tk.X, pady=3)
            mark = '●' if self.current_open == idx else '○'
            tk.Label(row, text=f'{mark} 第{idx + 1}组', bg='white',
                     font=("Arial", 10)).pack(side=tk.LEFT, padx=(2, 8))
            for side, label in (('front', '正面'), ('back', '反面')):
                slot = pair[side]
                if slot['processed'] is not None:
                    state, src_img = '已处理', slot['processed']
                elif slot['image'] is not None:
                    state, src_img = '待标角', slot['image']
                else:
                    state, src_img = '未导入', None
                if src_img is not None:
                    thumb = src_img.copy()
                    thumb.thumbnail((200, 126), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(thumb, master=self.root)
                    self._pair_photos[(idx, side)] = photo
                    w = tk.Label(row, image=photo, bg='white')
                    w.pack(side=tk.LEFT, padx=2)
                else:
                    # 浅灰描边：外层 #E0E0E0 边框框 + 内层扁平大 Label
                    slot_outer = tk.Frame(row, bg='#E0E0E0')
                    w = tk.Label(slot_outer, text=f'导入证件 {label}\n{state}', width=30, height=3,
                                 bg='#F5F5F7', fg='#666666', relief=tk.FLAT,
                                 anchor=tk.CENTER, font=("Microsoft YaHei", 11))
                    w.pack(padx=1, pady=1)
                    slot_outer.pack(side=tk.LEFT, padx=2)
                w.bind('<Button-1>', lambda e, i=idx, s=side: self.on_slot_click(i, s))
                if slot['image'] is not None:
                    tk.Label(row, text=state, bg='white', fg='#666666').pack(side=tk.LEFT)
                if DRAG_DROP_SUPPORTED:
                    try:
                        w.drop_target_register(DND_FILES)
                        w.dnd_bind('<<Drop>>', lambda e, i=idx, s=side: self.import_into_slot(i, s, e.data.strip().strip('{}').split('} {')[0]))
                    except Exception:
                        pass
            tk.Button(row, text='正反互换', command=lambda i=idx: self.swap_pair_sides(i),
                      bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                      relief=tk.FLAT, padx=10, pady=4,
                      cursor='hand2').pack(side=tk.LEFT, padx=4)
            tk.Button(row, text='删', command=lambda i=idx: self.delete_pair(i),
                      bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                      relief=tk.FLAT, padx=8, pady=4,
                      cursor='hand2').pack(side=tk.LEFT)


    def build_settings_bar(self, parent):
        # 换脸：两行设置 + 一条橙色大按钮
        bar = tk.Frame(parent, bg='#EEF0F3')
        bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(6, 0))

        # 第一行：左 新增一行 + 批量导入（蓝）；右 增强下拉 + 组数
        row1 = tk.Frame(bar, bg='#EEF0F3')
        row1.pack(fill=tk.X, pady=(0, 4))
        tk.Button(row1, text='＋ 新增一行', command=self.add_pair,
                  bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                  relief=tk.FLAT, padx=12, pady=5,
                  cursor='hand2').pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(row1, text='批量导入（两张一组）', command=self.batch_import,
                  bg='#2F7CF6', fg='white', activebackground='#2F7CF6',
                  activeforeground='white', relief=tk.FLAT,
                  font=("Arial", 10, "bold"), padx=14, pady=5,
                  cursor='hand2').pack(side=tk.LEFT)
        enh_box = tk.Frame(row1, bg='#EEF0F3')
        enh_box.pack(side=tk.RIGHT)
        tk.Label(enh_box, text='增强：', bg='#EEF0F3').pack(side=tk.LEFT)
        self._enhance_text_var = tk.StringVar(value='彩色')
        _enh_map = {'原图': 'none', '彩色': 'color', '黑白': 'bw',
                    '文档黑白': 'docbw', '文档彩色': 'doccolor'}

        def _on_enh_combo(event=None):
            v = _enh_map.get(self._enhance_text_var.get())
            if v is not None:
                self.enhance_var.set(v)
                self._on_enhance_change()

        self._enhance_combo = ttk.Combobox(enh_box, textvariable=self._enhance_text_var,
                                           values=list(_enh_map.keys()),
                                           width=10, state='readonly')
        self._enhance_combo.pack(side=tk.LEFT)
        self._enhance_combo.bind('<<ComboboxSelected>>', _on_enh_combo)
        if hasattr(self, 'pairs_count_var'):
            tk.Label(enh_box, textvariable=self.pairs_count_var, bg='#EEF0F3',
                     fg='#888888').pack(side=tk.LEFT, padx=(8, 0))

        # 第二行：安全水印参数 + 排版（沿用原有变量与控件）
        row2 = tk.Frame(bar, bg='#EEF0F3')
        row2.pack(fill=tk.X, pady=(0, 2))
        self.wm_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text='安全水印', variable=self.wm_enabled_var).pack(side=tk.LEFT)
        self.wm_text_var = tk.StringVar(value='')
        ttk.Entry(row2, textvariable=self.wm_text_var, width=22).pack(side=tk.LEFT, padx=2)
        tk.Label(row2, text='字号', bg='#EEF0F3').pack(side=tk.LEFT)
        self.wm_size_var = tk.IntVar(value=7)
        ttk.Combobox(row2, values=list(range(4, 15)), textvariable=self.wm_size_var, width=4, state='readonly').pack(side=tk.LEFT)
        tk.Label(row2, text='透明度%', bg='#EEF0F3').pack(side=tk.LEFT)
        self.wm_opacity_var = tk.IntVar(value=18)
        ttk.Combobox(row2, values=[5, 10, 15, 18, 20, 25, 30, 40, 50, 60], textvariable=self.wm_opacity_var, width=4, state='readonly').pack(side=tk.LEFT)
        tk.Label(row2, text='行数', bg='#EEF0F3').pack(side=tk.LEFT)
        self.wm_rows_var = tk.IntVar(value=3)
        ttk.Combobox(row2, values=list(range(1, 7)), textvariable=self.wm_rows_var, width=4, state='readonly').pack(side=tk.LEFT)
        tk.Label(row2, text='排版', bg='#EEF0F3').pack(side=tk.LEFT, padx=(12, 0))
        self.template_var = tk.StringVar(value='标准（每组一页）')
        ttk.Combobox(row2, textvariable=self.template_var, values=['标准（每组一页）', '紧凑拼版'], width=14, state='readonly').pack(side=tk.LEFT)

        # 第三行：整条橙色大按钮
        tk.Button(bar, text='生成 A4 并预览', command=self.show_batch_preview,
                  bg='#FF9500', fg='white', activebackground='#FF9500',
                  activeforeground='white', relief=tk.FLAT,
                  font=("Arial", 13, "bold"), pady=10,
                  cursor='hand2').pack(fill=tk.X, pady=(6, 0))

    def _template_key(self):
        v = self.template_var.get() if hasattr(self, 'template_var') else ''
        return 'compact' if '紧凑' in v else 'standard'

    def collect_pair_images(self):
        self._stash_workspace()
        groups = []
        for pair in self.pairs:
            imgs = [pair[s]['processed'] for s in ('front', 'back') if pair[s]['processed'] is not None]
            if imgs:
                groups.append(imgs)
        return groups

    def compose_a4_pages(self):
        """把所有组的处理结果拼成 A4 页（300dpi），返回 PIL 页列表"""
        groups = self.collect_pair_images()
        if not groups:
            return []
        s = self.mm_to_pixel
        pw, ph = int(self.a4_width * s), int(self.a4_height * s)
        cw, ch = int(self.id_width * s), int(self.id_height * s)

        def prep(img):
            if img.size != (cw, ch):
                img = img.resize((cw, ch), Image.Resampling.LANCZOS)
            if img.mode == 'RGBA':
                bg = Image.new('RGB', img.size, (255, 255, 255))
                bg.paste(img, (0, 0), img)
                return bg
            return img.convert('RGB')

        pages = []
        if self._template_key() == 'compact':
            cards = [prep(im) for g in groups for im in g]
            margin = int(10 * s)
            gap_y = int(4 * s)
            cols = 2
            gap_x = pw - 2 * margin - cols * cw
            rows = max(1, int((ph - 2 * margin + gap_y) // (ch + gap_y)))
            per = cols * rows
            for start in range(0, len(cards), per):
                page = Image.new('RGB', (pw, ph), (255, 255, 255))
                for j, card in enumerate(cards[start:start + per]):
                    r, c = divmod(j, cols)
                    x = margin + c * (cw + gap_x)
                    y = margin + r * (ch + gap_y)
                    page.paste(card, (x, y))
                pages.append(page)
        else:
            spacing = int(20 * s)
            for g in groups:
                imgs = [prep(im) for im in g]
                page = Image.new('RGB', (pw, ph), (255, 255, 255))
                total_h = len(imgs) * ch + (len(imgs) - 1) * spacing
                y = (ph - total_h) // 2
                x = (pw - cw) // 2
                for im in imgs:
                    page.paste(im, (x, y))
                    y += ch + spacing
                pages.append(page)
        if getattr(self, 'wm_enabled_var', None) is not None and self.wm_enabled_var.get():
            text = self.wm_text_var.get().strip()
            if text:
                font_px = int(self.wm_size_var.get() * s)
                pages = [_apply_watermark(p, text, font_px, self.wm_opacity_var.get(), self.wm_rows_var.get()) for p in pages]
        self.batch_pages = pages
        return pages

    def show_batch_preview(self):
        pages = self.compose_a4_pages()
        if not pages:
            messagebox.showwarning('提示', '还没有已处理的证件，请先导入并处理至少一张')
            return
        win = ToplevelWithIcon(self.root)
        win.title('A4 拼版预览')
        win.geometry('860x1020')
        state = {'idx': 0, 'photo': None}
        canvas = tk.Canvas(win, bg='gray')
        canvas.pack(fill=tk.BOTH, expand=True)

        def render():
            img = pages[state['idx']].copy()
            img.thumbnail((820, 940), Image.Resampling.LANCZOS)
            state['photo'] = ImageTk.PhotoImage(img, master=self.root)
            canvas.delete('all')
            canvas.create_image(430, 500, image=state['photo'], anchor=tk.CENTER)
            info.config(text=f"第 {state['idx'] + 1} / {len(pages)} 页")

        bar = ttk.Frame(win)
        bar.pack(side=tk.BOTTOM, fill=tk.X, pady=6)
        info = ttk.Label(bar, text='')
        info.pack(side=tk.LEFT, padx=8)
        ttk.Button(bar, text='上一页', command=lambda: (state.update(idx=(state['idx'] - 1) % len(pages)), render())).pack(side=tk.LEFT)
        ttk.Button(bar, text='下一页', command=lambda: (state.update(idx=(state['idx'] + 1) % len(pages)), render())).pack(side=tk.LEFT)
        ttk.Button(bar, text='保存JPG', command=self._export_pages_jpg, width=10).pack(side=tk.RIGHT, padx=2)
        ttk.Button(bar, text='保存PDF', command=self._export_pages_pdf, width=10).pack(side=tk.RIGHT, padx=2)
        ttk.Button(bar, text='打印', command=self._print_pages, width=8).pack(side=tk.RIGHT, padx=2)
        render()

    def _export_pages_jpg(self):
        pages = getattr(self, 'batch_pages', None) or self.compose_a4_pages()
        if not pages:
            return
        folder = filedialog.askdirectory(title='选择保存文件夹')
        if not folder:
            return
        for i, p in enumerate(pages):
            p.save(os.path.join(folder, f'证件拼版_{i + 1:02d}.jpg'), quality=95, dpi=(self.dpi, self.dpi))
        messagebox.showinfo('完成', f'已保存 {len(pages)} 页 JPG 到所选文件夹')

    def _pages_to_pdf_bytes(self, pages):
        tmpdir = tempfile.mkdtemp()
        paths = []
        for i, p in enumerate(pages):
            fp = os.path.join(tmpdir, f'page{i + 1}.jpg')
            p.save(fp, quality=95, dpi=(self.dpi, self.dpi))
            paths.append(fp)
        return img2pdf.convert(*paths)

    def _export_pages_pdf(self):
        pages = getattr(self, 'batch_pages', None) or self.compose_a4_pages()
        if not pages:
            return
        folder = filedialog.askdirectory(title='选择保存文件夹')
        if not folder:
            return
        with open(os.path.join(folder, '证件拼版.pdf'), 'wb') as f:
            f.write(self._pages_to_pdf_bytes(pages))
        messagebox.showinfo('完成', '已保存 证件拼版.pdf 到所选文件夹')

    def _print_pages(self):
        pages = getattr(self, 'batch_pages', None) or self.compose_a4_pages()
        if not pages:
            return
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(self._pages_to_pdf_bytes(pages))
            temp_file = tmp.name
        try:
            if platform.system() == 'Windows':
                os.startfile(temp_file, 'print')
            else:
                subprocess.run(['lp', temp_file], check=True)
        except Exception as e:
            messagebox.showerror('打印失败',
                                 f'没能把打印任务发出去：{e}\n（拼版文件已存到 {temp_file}，可手动打开打印）')
            return
        messagebox.showinfo('打印', '打印任务已发送到打印机')


    def load_image(self, side):
        """加载图片"""
        file_path = filedialog.askopenfilename(
            title=f"选择{side}图片",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            image = Image.open(file_path)
            
            # 限制图片大小
            max_size = (1200, 1200)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # 在A4预览区域显示缩略图
            self.show_preview_image(side, image)
            
            # 显示原始图片，等待用户标记
            self.show_image_for_marking(side, image, file_path)
            
            if side == "front":
                side_name = "身份证人像面"
            elif side == "back":
                side_name = "身份证国徽面"
            else:
                side_name = "银行卡"
                
            self.update_status(f"已加载{side_name}图片，请开始标记")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
    
    def show_preview_image(self, side, image):
        """在A4预览区域显示缩略图"""
        if not hasattr(self, 'canvas_info'):
            return
            
        info = self.canvas_info
        
        if side == "front":
            # 移除预览文本
            if hasattr(self, 'front_preview_text'):
                self.a4_canvas.delete(self.front_preview_text)
            
            # 创建预览图片
            preview_img = image.copy()
            preview_width = info['id_width'] - 30
            preview_height = info['front_preview_height']
            preview_img.thumbnail((int(preview_width), int(preview_height)), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(preview_img, master=self.root)
            
            # 保存图片引用
            self.front_preview_img = photo
            
            # 在预览区域显示图片
            self.a4_canvas.create_image(
                info['margin_x'] + info['id_width']/2,
                info['front_preview_y'] + info['front_preview_height']/2,
                image=photo, anchor=tk.CENTER
            )
            
        elif side == "back":
            # 移除预览文本
            if hasattr(self, 'back_preview_text'):
                self.a4_canvas.delete(self.back_preview_text)
            
            # 创建预览图片
            preview_img = image.copy()
            preview_width = info['id_width'] - 30
            preview_height = info['back_preview_height']
            preview_img.thumbnail((int(preview_width), int(preview_height)), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(preview_img, master=self.root)
            
            # 保存图片引用
            self.back_preview_img = photo
            
            # 在预览区域显示图片
            self.a4_canvas.create_image(
                info['margin_x'] + info['id_width']/2,
                info['back_preview_y'] + info['back_preview_height']/2,
                image=photo, anchor=tk.CENTER
            )
            
        elif side == "card":
            # 移除预览文本
            if hasattr(self, 'card_preview_text'):
                self.a4_canvas.delete(self.card_preview_text)
            
            # 创建预览图片
            preview_img = image.copy()
            preview_width = info.get('card_width', info.get('id_width', 0)) - 30
            preview_height = info.get('card_preview_height', info.get('id_height', 0) - 30)
            preview_img.thumbnail((int(preview_width), int(preview_height)), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(preview_img, master=self.root)
            
            # 保存图片引用
            self.card_preview_img = photo
            
            # 在预览区域显示图片
            self.a4_canvas.create_image(
                info.get('card_margin_x', info.get('margin_x', 0)) + info.get('card_width', info.get('id_width', 0))/2,
                info['card_preview_y'] + info['card_preview_height']/2,
                image=photo, anchor=tk.CENTER
            )
    
    def show_image_for_marking(self, side, image, file_path, existing_points=None):
        """显示图片供用户标记 - 带缩放功能"""
        if self.mark_window is not None:
            self.mark_window.destroy()
            
        # 创建标记窗口
        self.mark_window = ToplevelWithIcon(self.root)
        
        if side == "front":
            side_name = "身份证人像面"
        elif side == "back":
            side_name = "身份证国徽面"
        else:
            side_name = "银行卡"
            
        self.mark_window.title(f"标记{side_name} - 请点击4个角点（顺序不限）")
        
        # 根据图片大小设置窗口尺寸
        img_width, img_height = image.size
        window_width = min(img_width + 50, 1000)
        window_height = min(img_height + 200, 800)
        self.mark_window.geometry(f"{window_width}x{window_height}")
        
        # 将标记窗口居中显示
        self.center_window(self.mark_window)
        
        # 记录当前标记的侧面
        self.current_marking_side = side
        
        # 保存图片信息
        if side == "front":
            self.front_image = image
            self.front_image_path = file_path
            if existing_points is None:
                self.front_points = []
            else:
                self.front_points = existing_points.copy()
        elif side == "back":
            self.back_image = image
            self.back_image_path = file_path
            if existing_points is None:
                self.back_points = []
            else:
                self.back_points = existing_points.copy()
        else:
            self.card_image = image
            self.card_image_path = file_path
            if existing_points is None:
                self.card_points = []
            else:
                self.card_points = existing_points.copy()
        
        # 自动开始标记
        self.is_marking = True
        self._drag_idx = None
        
        # 创建主框架
        main_frame = ttk.Frame(self.mark_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # 添加缩放控制
        scale_frame = ttk.Frame(control_frame)
        scale_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(scale_frame, text="缩放:").pack(side=tk.LEFT)
        
        # 创建缩放滑块
        self.scale_var = tk.DoubleVar(value=1.0)
        scale_slider = ttk.Scale(scale_frame, from_=0.1, to=3.0, 
                                variable=self.scale_var, length=150)
        scale_slider.pack(side=tk.LEFT, padx=5)
        scale_slider.bind("<Motion>", self.on_scale_change)
        scale_slider.bind("<ButtonRelease-1>", self.on_scale_change)
        
        # 显示当前缩放比例
        self.scale_label = ttk.Label(scale_frame, text="100%")
        self.scale_label.pack(side=tk.LEFT, padx=5)
        
        # 控制按钮区域
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        # 添加自动识别按钮
        ttk.Button(btn_frame, text="自动识别", 
                  command=lambda: self.auto_detect_corners(side)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="重新开始", 
                  command=lambda: self.reset_marking(side)).pack(side=tk.LEFT, padx=5)
        
        # 创建完成标记按钮（初始状态）
        self.finish_marking_btn = ttk.Button(btn_frame, text="完成标记", 
                                          command=lambda: self.finish_marking(side),
                                          state="normal")
        self.finish_marking_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="取消", 
                  command=self.mark_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # 创建画布框架
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建画布和滚动条
        self.mark_canvas = tk.Canvas(canvas_frame, bg='gray', cursor="cross")
        
        # 创建滚动条
        self.mark_scrollbar_y = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.mark_canvas.yview)
        self.mark_scrollbar_x = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.mark_canvas.xview)
        
        # 配置画布
        self.mark_canvas.configure(yscrollcommand=self.mark_scrollbar_y.set,
                                  xscrollcommand=self.mark_scrollbar_x.set)
        
        # 放置组件
        self.mark_canvas.grid(row=0, column=0, sticky="nsew")
        self.mark_scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.mark_scrollbar_x.grid(row=1, column=0, sticky="ew")
        
        # 配置网格权重
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # 绑定事件
        self.mark_canvas.bind("<Button-1>", self.mark_point)
        self.mark_canvas.bind("<B1-Motion>", self.drag_mark_point)
        self.mark_canvas.bind("<ButtonRelease-1>", self.end_drag_mark_point)
        self.mark_canvas.bind("<Motion>", self.draw_crosshair)
        self.mark_canvas.bind("<Leave>", self.clear_crosshair)
        
        # 绑定鼠标滚轮事件（缩放控制）
        self.mark_canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.mark_canvas.bind("<Button-4>", self.on_mousewheel)    # Linux
        self.mark_canvas.bind("<Button-5>", self.on_mousewheel)    # Linux
        
        # 绑定键盘回车事件
        self.mark_window.bind("<Return>", lambda e: self.finish_marking(side))
        
        # 居中显示窗口
        self.mark_window.transient(self.root)
        self.mark_window.grab_set()
        self.mark_window.focus_set()
        
        # 窗口关闭事件
        self.mark_window.protocol("WM_DELETE_WINDOW", self.mark_window.destroy)
        
        # 保存原始图片
        self.original_image = image.copy()
        self.current_scale = 1.0
        
        # 如果有传入的点，直接完成标记
        if existing_points is not None and len(existing_points) == 4:
            # 更新完成标记按钮状态
            self.finish_marking_btn.config(state="normal")
            # 显示图片和点
            self.display_image_on_canvas()
            self.update_status(f"{side_name}已有标记点，可以进行调整或点击完成标记")
        else:
            # 显示图片
            self.display_image_on_canvas()
            
            # 添加说明文本
            self.mark_canvas.create_text(10, 20, anchor="nw",
                                         text="请点击证件4个角点（顺序不限，点偏了可直接拖动）或点击'自动识别'",
                                         fill="white", font=("Arial", 12, "bold"))
            
            side_name = "身份证人像面" if side == "front" else "身份证国徽面" if side == "back" else "银行卡"
            self.update_status(f"已自动开始标记{side_name}，请点击4个角点（顺序不限）或使用自动识别")
            # 新增：标记窗口打开后先静默尝试真自动识别，成功直接完成该面处理、
            # 不弹窗确认；失败则什么都不做，停留在原手动标记流程。
            self.mark_window.after(150, lambda s=side: self.try_auto_finish(s))
    
    def detect_card_corners(self, image):
        """纯检测：返回按左上→右上→右下→左下排序的角点列表（原图坐标），失败返回 None。
        多路二值化取最大轮廓，凸包近似四边形，近似不成功则退最小外接矩形，
        最后用原有四边形校验把关，宁可失败也不给错框。"""
        try:
            img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            h0, w0 = img_cv.shape[:2]
            scale = 800.0 / w0 if w0 > 800 else 1.0
            if scale != 1.0:
                work = cv2.resize(img_cv, (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
                                  interpolation=cv2.INTER_AREA)
            else:
                work = img_cv
            hw, ww = work.shape[:2]
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binaries = [otsu, 255 - otsu, cv2.Canny(gray, 50, 150)]

            kernel = np.ones((5, 5), np.uint8)
            best = None
            for binary in binaries:
                closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < hw * ww * 0.12:
                        continue
                    hull = cv2.convexHull(contour)
                    quad = None
                    for eps_factor in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
                        approx = cv2.approxPolyDP(hull, eps_factor * cv2.arcLength(hull, True), True)
                        if len(approx) == 4:
                            quad = approx.reshape(4, 2)
                            break
                    if quad is None:
                        quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
                    corners = self.order_points(np.array(quad, dtype=np.float32))
                    # 整幅画面的边框不是卡：面积占满全图的一律跳过（实测假阳性）
                    if cv2.contourArea(corners.reshape(-1, 1, 2)) > hw * ww * 0.90:
                        continue
                    if not cv2.isContourConvex(corners.astype(np.int32).reshape(-1, 1, 2)):
                        continue
                    if not self._validate_quadrilateral(corners.copy(), hw, ww):
                        continue
                    if best is None or area > best[0]:
                        best = (area, corners)
            if best is None:
                return None
            corners = best[1] / scale
            return [[float(x), float(y)] for x, y in corners]
        except Exception:
            return None

    def ai_detect_corners(self, image):
        """AI 检测：懒加载 SCRFD 模型（exe 同目录或打包内 models 文件夹），
        模型缺失或检测失败返回 None，由传统检测兜底。"""
        try:
            detector = getattr(self, "_ai_detector", None)
            if detector is None:
                import os as _os
                candidates = []
                if getattr(sys, "frozen", False):
                    candidates.append(_os.path.join(_os.path.dirname(sys.executable),
                                      "models", "carddetection_scrfd34gkps.onnx"))
                    candidates.append(_os.path.join(getattr(sys, "_MEIPASS", ""),
                                      "models", "carddetection_scrfd34gkps.onnx"))
                candidates.append(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                  "models", "carddetection_scrfd34gkps.onnx"))
                model = next((c for c in candidates if c and _os.path.exists(c)), None)
                if not model:
                    self._ai_detector = False
                    return None
                self._ai_detector = CardDetectorAI(model)
                detector = self._ai_detector
            if not detector:
                return None
            img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            self._ai_ran = True
            corners = detector.detect(img_cv)
            if not corners:
                return None
            arr = np.array(corners, dtype=np.float32)
            h, w = img_cv.shape[:2]
            if not self._validate_quadrilateral(arr.copy(), h, w):
                return None
            return [[float(x), float(y)] for x, y in arr]
        except Exception:
            return None

    def try_auto_finish(self, side):
        """标记窗口打开后静默尝试自动识别：成功直接完成该面处理（不弹窗确认）；
        失败不打扰用户，停留在手动标记流程。"""
        try:
            if self.mark_window is None or self.current_marking_side != side:
                return
            if side == "front":
                image, points = self.front_image, self.front_points
            elif side == "back":
                image, points = self.back_image, self.back_points
            else:
                image, points = self.card_image, self.card_points
            if image is None or len(points) == 4:
                return
            corners = self.ai_detect_corners(image)
            if not corners and not getattr(self, '_ai_ran', False):
                corners = self.detect_card_corners(image)
            if not corners:
                self.update_status("自动识别未成功，请手动点击4个角点，或点上方'自动识别'再试")
                return
            if side == "front":
                self.front_points = corners
            elif side == "back":
                self.back_points = corners
            else:
                self.card_points = corners
            self.finish_marking(side)
        except Exception:
            pass

    def auto_detect_corners(self, side):
        """改进的自动识别证件角点 - 使用凸包检测与最小四边形判断"""
        try:
            # 获取当前图片
            if side == "front":
                image = self.front_image
            elif side == "back":
                image = self.back_image
            else:
                image = self.card_image

            if image is None:
                messagebox.showwarning("警告", "图片未加载")
                return False

            # 新增快速通道：先用新检测器试一次，成功就不必走旧流程
            corners_new = self.detect_card_corners(image)
            if corners_new:
                if side == "front":
                    self.front_points = corners_new
                elif side == "back":
                    self.back_points = corners_new
                else:
                    self.card_points = corners_new
                self.display_image_on_canvas()
                if self.finish_marking_btn:
                    self.finish_marking_btn.config(state="normal")
                self.update_status("自动识别成功！可直接点击'完成标记'，或拖动角点微调")
                return True

            # 转换图片为OpenCV格式
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            height, width = img_cv.shape[:2]

            # 多种预处理方法
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # 方法1：自适应阈值
            thresh1 = cv2.adaptiveThreshold(gray, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)

            # 方法2：边缘检测
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            edges = cv2.Canny(blurred, 50, 150)

            # 方法3：Sobel算子
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = np.uint8(sobel / sobel.max() * 255)

            # 凸包检测与最小四边形判断
            detected_corners = None
            for method, binary_img in [("边缘", edges), 
                                       ("阈值", thresh1), 
                                       ("Sobel", sobel)]:

                # 形态学操作增强边缘
                kernel = np.ones((5, 5), np.uint8)
                processed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
                
                # 查找轮廓
                contours, _ = cv2.findContours(processed, 
                                              cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    continue

                # 找到面积最大的几个轮廓
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

                for contour in contours:
                    # 计算轮廓面积
                    area = cv2.contourArea(contour)
                    
                    # 过滤掉太小的轮廓
                    if area < height * width * 0.05:
                        continue
                    
                    # 使用凸包检测
                    hull = cv2.convexHull(contour)
                    
                    # 如果凸包点数大于4，尝试使用最小四边形拟合
                    if len(hull) > 4:
                        # 计算最小四边形
                        epsilon = 0.02 * cv2.arcLength(hull, True)
                        approx = cv2.approxPolyDP(hull, epsilon, True)
                        
                        # 如果近似后不是四边形，继续调整epsilon
                        for eps_factor in [0.01, 0.02, 0.03, 0.04, 0.05]:
                            epsilon = eps_factor * cv2.arcLength(hull, True)
                            approx = cv2.approxPolyDP(hull, epsilon, True)
                            if len(approx) == 4:
                                break
                    else:
                        approx = hull
                    
                    # 检查是否为四边形
                    if len(approx) == 4:
                        corners = self.order_points(approx.reshape(4, 2))

                        # 验证四边形质量
                        if self._validate_quadrilateral(corners, height, width):
                            detected_corners = corners
                            self.update_status(f"使用{method}检测方法找到角点")
                            break
                        
                if detected_corners is not None:
                    break
                
            # 如果没找到，尝试使用Harris角点检测结合凸包
            if detected_corners is None:
                corners = cv2.cornerHarris(gray, 2, 3, 0.04)
                corners = cv2.dilate(corners, None)
                
                # 阈值化获取角点位置
                _, corners_thresh = cv2.threshold(corners, 0.01 * corners.max(), 255, 0)
                corners_thresh = np.uint8(corners_thresh)
                
                # 找到连通分量
                _, labels, stats, _ = cv2.connectedComponentsWithStats(corners_thresh)
                
                # 获取质心位置
                centroids = cv2.connectedComponentsWithStats(corners_thresh)[3]
                
                if len(centroids) > 4:
                    # 使用凸包找到最外层的点
                    points = np.array([centroids[i] for i in range(1, min(len(centroids), 20))], dtype=np.float32)
                    hull = cv2.convexHull(points)
                    
                    # 如果凸包点数大于4，使用最小四边形拟合
                    if len(hull) > 4:
                        epsilon = 0.1 * cv2.arcLength(hull, True)
                        approx = cv2.approxPolyDP(hull, epsilon, True)
                    else:
                        approx = hull
                    
                    if len(approx) == 4:
                        detected_corners = self.order_points(approx.reshape(4, 2))

            if detected_corners is not None:
                # 更新对应的点列表
                if side == "front":
                    self.front_points = detected_corners.tolist()
                elif side == "back":
                    self.back_points = detected_corners.tolist()
                else:
                    self.card_points = detected_corners.tolist()

                # 重新显示图片以更新标记点
                self.display_image_on_canvas()

                # 禁用完成标记按钮，启用调整按钮
                if self.finish_marking_btn:
                    self.finish_marking_btn.config(state="disabled")
                
                self.update_status("自动识别成功！")

                # 不再弹窗询问是否满意：识别成功即启用完成标记，
                # 用户可直接完成，或自行拖动角点微调。
                if self.finish_marking_btn:
                    self.finish_marking_btn.config(state="normal")
                return True
            else:
                self.update_status("自动识别失败，请手动标记")
                messagebox.showwarning("警告", 
                    "自动识别失败，未能找到有效的证件边缘。\n\n" +
                    "建议：\n" +
                    "1. 确保证件在图片中清晰可见\n" +
                    "2. 调整图片角度，避免倾斜\n" +
                    "3. 选择光线均匀的图片\n" +
                    "4. 手动标记角点")
                return False

        except Exception as e:
            self.update_status(f"自动识别出错: {str(e)}")
            messagebox.showerror("错误", f"自动识别失败: {str(e)}")
            return False

    def enter_adjust_mode(self, side):
        """进入调整模式，允许手动调整四个角点"""
        self.adjust_mode = True
        self.current_marking_side = side
        
        # 获取当前的点
        if side == "front":
            self.adjust_points = self.front_points.copy()
        elif side == "back":
            self.adjust_points = self.back_points.copy()
        else:
            self.adjust_points = self.card_points.copy()
        
        # 清空画布，准备调整模式
        self.mark_canvas.delete("all")
        
        # 重新显示图片
        self.display_image_on_canvas()
        
        # 绘制调整模式的点和连线
        self.draw_adjust_points()
        
        # 更改说明文本
        self.mark_canvas.delete("instruction")
        self.mark_canvas.create_text(10, 20, anchor="nw", tags="instruction",
                                     text="调整模式：拖动四个角点调整位置，然后点击'完成调整'或按回车键",
                                     fill="yellow", font=("Arial", 12, "bold"))
        
        # 修改按钮 - 先移除原有按钮
        for widget in self.mark_window.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Frame):
                        for btn in child.winfo_children():
                            if isinstance(btn, ttk.Button):
                                btn.destroy()
        
        # 重新添加调整模式的按钮
        control_frame = self.mark_window.winfo_children()[0].winfo_children()[0]
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        # 创建完成调整按钮
        self.finish_adjust_btn = ttk.Button(btn_frame, text="完成调整", 
                                          command=lambda: self.finish_adjust(side))
        self.finish_adjust_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="取消调整", 
                  command=lambda: self.cancel_adjust(side)).pack(side=tk.LEFT, padx=5)
        
        # 重新绑定键盘事件
        self.mark_window.unbind("<Return>")
        self.mark_window.bind("<Return>", lambda e: self.finish_adjust(side))
        
        # 绑定调整模式的事件
        self.mark_canvas.bind("<Button-1>", self.start_drag_point)
        self.mark_canvas.bind("<B1-Motion>", self.drag_point)
        self.mark_canvas.bind("<ButtonRelease-1>", self.stop_drag_point)
        
        self.update_status("已进入调整模式，请拖动角点调整位置")
    
    def draw_adjust_points(self):
        """绘制调整模式的点和连线"""
        # 清除之前的点和连线
        for item in self.point_circles:
            self.mark_canvas.delete(item)
        self.point_circles = []
        
        # 绘制四个角点（可拖动）
        colors = ["red", "green", "blue", "orange"]
        labels = ["左上", "右上", "右下", "左下"]
        
        for i, (px, py) in enumerate(self.adjust_points):
            # 缩放坐标
            scaled_x = px * self.current_scale
            scaled_y = py * self.current_scale
            
            # 绘制点（可拖动的圆）
            point_id = self.mark_canvas.create_oval(
                scaled_x-10, scaled_y-10, scaled_x+10, scaled_y+10,
                fill=colors[i], outline="white", width=2,
                tags=f"point_{i}"
            )
            self.point_circles.append(point_id)
            
            # 显示点标签
            self.mark_canvas.create_text(
                scaled_x, scaled_y-25,
                text=labels[i],
                fill="white", font=("Arial", 10, "bold"),
                tags=f"label_{i}"
            )
            self.point_circles.append(f"label_{i}")
        
        # 绘制四边形连线
        for i in range(4):
            x1 = self.adjust_points[i][0] * self.current_scale
            y1 = self.adjust_points[i][1] * self.current_scale
            x2 = self.adjust_points[(i+1) % 4][0] * self.current_scale
            y2 = self.adjust_points[(i+1) % 4][1] * self.current_scale
            
            line_id = self.mark_canvas.create_line(
                x1, y1, x2, y2,
                fill="cyan", width=2, dash=(5, 2)
            )
            self.point_circles.append(line_id)
    
    def start_drag_point(self, event):
        """开始拖动点"""
        # 获取点击位置（考虑滚动）
        canvas_x = self.mark_canvas.canvasx(event.x)
        canvas_y = self.mark_canvas.canvasy(event.y)
        
        # 检查是否点击了点
        for i in range(4):
            point_id = self.mark_canvas.find_withtag(f"point_{i}")[0]
            coords = self.mark_canvas.coords(point_id)
            
            # 检查点击是否在点内
            if (coords[0] <= canvas_x <= coords[2] and 
                coords[1] <= canvas_y <= coords[3]):
                self.dragging_point_index = i
                self.update_status(f"正在拖动{['左上', '右上', '右下', '左下'][i]}角点")
                break
    
    def drag_point(self, event):
        """拖动点"""
        if self.dragging_point_index is not None:
            # 获取画布坐标（考虑滚动）
            canvas_x = self.mark_canvas.canvasx(event.x)
            canvas_y = self.mark_canvas.canvasy(event.y)
            
            # 转换为原始图像坐标（考虑缩放）
            img_x = canvas_x / self.current_scale
            img_y = canvas_y / self.current_scale
            
            # 确保坐标在图像范围内
            img_width, img_height = self.original_image.size
            img_x = max(0, min(img_x, img_width-1))
            img_y = max(0, min(img_y, img_height-1))
            
            # 更新点位置
            self.adjust_points[self.dragging_point_index] = [img_x, img_y]
            
            # 重新绘制点和连线
            self.draw_adjust_points()
    
    def stop_drag_point(self, event):
        """停止拖动点"""
        self.dragging_point_index = None
        self.update_status("点位置已更新")
    
    def finish_adjust(self, side):
        """完成调整"""
        # 更新对应的点列表
        if side == "front":
            self.front_points = self.adjust_points.copy()
        elif side == "back":
            self.back_points = self.adjust_points.copy()
        else:
            self.card_points = self.adjust_points.copy()
        
        # 退出调整模式
        self.adjust_mode = False
        self.adjust_points = []
        self.dragging_point_index = None
        
        # 重新显示图片和点（正常模式）
        self.display_image_on_canvas()
        
        # 恢复原有点击事件
        self.mark_canvas.bind("<Button-1>", self.mark_point)
        self.mark_canvas.unbind("<B1-Motion>")
        self.mark_canvas.unbind("<ButtonRelease-1>")
        self.mark_canvas.bind("<B1-Motion>", self.drag_mark_point)
        self.mark_canvas.bind("<ButtonRelease-1>", self.end_drag_mark_point)
        
        # 更新按钮
        control_frame = self.mark_window.winfo_children()[0].winfo_children()[0]
        for widget in control_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                widget.destroy()
        
        # 重新添加正常模式的按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="自动识别", 
                  command=lambda: self.auto_detect_corners(side)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="重新开始", 
                  command=lambda: self.reset_marking(side)).pack(side=tk.LEFT, padx=5)
        
        # 重新创建完成标记按钮
        self.finish_marking_btn = ttk.Button(btn_frame, text="完成标记", 
                                          command=lambda: self.finish_marking(side))
        self.finish_marking_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="取消", 
                  command=self.mark_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # 重新绑定键盘回车事件
        self.mark_window.unbind("<Return>")
        self.mark_window.bind("<Return>", lambda e: self.finish_marking(side))
        
        self.update_status(f"{side}角点调整完成，可以继续处理或完成标记")
        
        # 询问是否完成标记
        response = messagebox.askyesno("完成调整", "角点调整完成，是否立即完成标记并开始处理？")
        if response:
            self.finish_marking(side)
    
    def cancel_adjust(self, side):
        """取消调整，恢复原始点"""
        self.adjust_mode = False
        self.adjust_points = []
        self.dragging_point_index = None
        
        # 恢复原有点击事件
        self.mark_canvas.bind("<Button-1>", self.mark_point)
        self.mark_canvas.unbind("<B1-Motion>")
        self.mark_canvas.unbind("<ButtonRelease-1>")
        self.mark_canvas.bind("<B1-Motion>", self.drag_mark_point)
        self.mark_canvas.bind("<ButtonRelease-1>", self.end_drag_mark_point)
        
        # 重新显示原始点
        self.display_image_on_canvas()
        
        # 更新按钮
        control_frame = self.mark_window.winfo_children()[0].winfo_children()[0]
        for widget in control_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                widget.destroy()
        
        # 重新添加正常模式的按钮
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        
        ttk.Button(btn_frame, text="自动识别", 
                  command=lambda: self.auto_detect_corners(side)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="重新开始", 
                  command=lambda: self.reset_marking(side)).pack(side=tk.LEFT, padx=5)
        
        # 重新创建完成标记按钮
        self.finish_marking_btn = ttk.Button(btn_frame, text="完成标记", 
                                          command=lambda: self.finish_marking(side))
        self.finish_marking_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="取消", 
                  command=self.mark_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # 重新绑定键盘回车事件
        self.mark_window.unbind("<Return>")
        self.mark_window.bind("<Return>", lambda e: self.finish_marking(side))
        
        self.update_status("已取消调整模式")

    def _validate_quadrilateral(self, corners, img_height, img_width):
        """验证四边形是否合理"""
        # 检查点是否在图像范围内
        for point in corners:
            if point[0] < 0 or point[0] > img_width or \
               point[1] < 0 or point[1] > img_height:
                return False

        # 计算长宽比
        width1 = np.linalg.norm(corners[1] - corners[0])
        width2 = np.linalg.norm(corners[2] - corners[3])
        height1 = np.linalg.norm(corners[3] - corners[0])
        height2 = np.linalg.norm(corners[2] - corners[1])

        avg_width = (width1 + width2) / 2
        avg_height = (height1 + height2) / 2

        if avg_height == 0:
            return False

        aspect_ratio = avg_width / avg_height
        expected_ratio = self.id_width / self.id_height

        # 允许的比例范围更宽（考虑透视变形）
        if not (0.7 * expected_ratio <= aspect_ratio <= 1.3 * expected_ratio):
            return False

        # 检查四边形面积是否合理（至少占图像的15%）
        area = cv2.contourArea(corners.reshape(-1, 1, 2))
        if area < img_height * img_width * 0.15:
            return False

        return True
    
    def order_points(self, pts):
        """按左上、右上、右下、左下顺序排序角点"""
        # 初始化坐标列表
        rect = np.zeros((4, 2), dtype="float32")
        
        # 计算质心
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上角
        rect[2] = pts[np.argmax(s)]  # 右下角
        
        # 计算差值
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上角
        rect[3] = pts[np.argmax(diff)]  # 左下角
        
        return rect
    
    def on_mousewheel(self, event):
        """鼠标滚轮缩放控制"""
        # 计算缩放增量
        scale_increment = 0.1
        
        # 不同系统的事件处理
        if event.num == 4:  # Linux向上滚
            self.scale_var.set(min(self.scale_var.get() + scale_increment, 3.0))
        elif event.num == 5:  # Linux向下滚
            self.scale_var.set(max(self.scale_var.get() - scale_increment, 0.1))
        elif hasattr(event, 'delta'):  # Windows
            if event.delta > 0:
                self.scale_var.set(min(self.scale_var.get() + scale_increment, 3.0))
            else:
                self.scale_var.set(max(self.scale_var.get() - scale_increment, 0.1))
        
        self.on_scale_change()
    
    def on_scale_change(self, event=None):
        """缩放改变事件"""
        scale = self.scale_var.get()
        self.current_scale = scale
        self.scale_label.config(text=f"{int(scale*100)}%")
        
        # 重新显示图片
        self.display_image_on_canvas()
        
        # 重新绘制十字线
        if hasattr(self, 'last_mouse_x') and hasattr(self, 'last_mouse_y'):
            self.draw_crosshair_manual(self.last_mouse_x, self.last_mouse_y)
    
    def draw_crosshair_manual(self, x, y):
        """手动绘制十字线（用于缩放后重新绘制）"""
        # 清除之前的十字线
        for line in self.crosshair_lines:
            self.mark_canvas.delete(line)
        self.crosshair_lines = []
        
        # 获取画布尺寸
        canvas_width = self.mark_canvas.winfo_width()
        canvas_height = self.mark_canvas.winfo_height()
        
        # 绘制水平虚线
        if 0 <= y <= canvas_height:
            self.crosshair_lines.append(
                self.mark_canvas.create_line(
                    0, y, canvas_width, y,
                    fill="cyan", dash=(5, 2), width=1
                )
            )
        
        # 绘制垂直虚线
        if 0 <= x <= canvas_width:
            self.crosshair_lines.append(
                self.mark_canvas.create_line(
                    x, 0, x, canvas_height,
                    fill="cyan", dash=(5, 2), width=1
                )
            )
        
        # 保存鼠标位置
        self.last_mouse_x = x
        self.last_mouse_y = y
    
    def display_image_on_canvas(self):
        """在标记画布上显示图片（带缩放）"""
        if self.mark_canvas and hasattr(self, 'original_image'):
            # 清除画布上的所有内容
            self.mark_canvas.delete("all")
            
            # 获取原始图片尺寸
            img_width, img_height = self.original_image.size
            
            # 计算缩放后的尺寸
            scaled_width = max(1, int(img_width * self.current_scale))
            scaled_height = max(1, int(img_height * self.current_scale))
            
            # 缩放图片
            if self.current_scale != 1.0:
                scaled_image = self.original_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            else:
                scaled_image = self.original_image.copy()
            
            # 创建PhotoImage
            self.mark_photo = ImageTk.PhotoImage(scaled_image, master=self.root)
            
            # 在画布上显示图像
            self.mark_canvas.create_image(0, 0, image=self.mark_photo, anchor=tk.NW)
            
            # 设置画布的可滚动区域
            self.mark_canvas.config(scrollregion=(0, 0, scaled_width, scaled_height))
            
            # 如果处于调整模式，绘制调整点
            if self.adjust_mode:
                self.draw_adjust_points()
                return
            
            # 重新绘制已标记的点（正常模式）
            if self.current_marking_side == "front":
                points = self.front_points
            elif self.current_marking_side == "back":
                points = self.back_points
            else:
                points = self.card_points
            
            for i, (px, py) in enumerate(points):
                # 缩放坐标
                scaled_x = px * self.current_scale
                scaled_y = py * self.current_scale
                
                # 在画布上标记点
                self.mark_canvas.create_oval(
                    scaled_x-5, scaled_y-5, scaled_x+5, scaled_y+5,
                    fill="red", outline="white", width=2
                )
                
                # 显示点编号
                self.mark_canvas.create_text(
                    scaled_x+15, scaled_y-15,
                    text=str(i+1),
                    fill="white", font=("Arial", 12, "bold")
                )
                
                # 连接点（如果已选择多个点）
                if i > 0:
                    prev_x = points[i-1][0] * self.current_scale
                    prev_y = points[i-1][1] * self.current_scale
                    
                    self.mark_canvas.create_line(
                        prev_x, prev_y, scaled_x, scaled_y,
                        fill="green", width=2
                    )
            
            # 更新完成标记按钮状态
            if len(points) == 4:
                if self.finish_marking_btn:
                    self.finish_marking_btn.config(state="normal")
                # 添加提示文本
                self.mark_canvas.delete("instruction")
                self.mark_canvas.create_text(10, 20, anchor="nw", tags="instruction",
                                             text="已标记4个点，可以点击'完成标记'或按回车键",
                                             fill="yellow", font=("Arial", 12, "bold"))
            else:
                if self.finish_marking_btn:
                    self.finish_marking_btn.config(state="disabled")
                # 重新添加说明文本
                self.mark_canvas.create_text(10, 20, anchor="nw",
                                             text="请点击证件4个角点（顺序不限，点偏了可直接拖动）或点击'自动识别'",
                                             fill="white", font=("Arial", 12, "bold"))
    
    def draw_crosshair(self, event):
        """绘制十字虚线"""
        if not self.mark_canvas:
            return
            
        # 保存鼠标位置
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
            
        # 清除之前的十字线
        for line in self.crosshair_lines:
            self.mark_canvas.delete(line)
        self.crosshair_lines = []
        
        # 获取画布坐标（考虑滚动）
        canvas_x = self.mark_canvas.canvasx(event.x)
        canvas_y = self.mark_canvas.canvasy(event.y)
        
        # 获取画布尺寸
        canvas_width = self.mark_canvas.winfo_width()
        canvas_height = self.mark_canvas.winfo_height()
        
        # 绘制水平虚线
        if 0 <= canvas_y <= canvas_height:
            self.crosshair_lines.append(
                self.mark_canvas.create_line(
                    0, canvas_y, canvas_width, canvas_y,
                    fill="cyan", dash=(5, 2), width=1
                )
            )
        
        # 绘制垂直虚线
        if 0 <= canvas_x <= canvas_width:
            self.crosshair_lines.append(
                self.mark_canvas.create_line(
                    canvas_x, 0, canvas_x, canvas_height,
                    fill="cyan", dash=(5, 2), width=1
                )
            )
    
    def clear_crosshair(self, event):
        """清除十字线"""
        if not self.mark_canvas:
            return
            
        for line in self.crosshair_lines:
            self.mark_canvas.delete(line)
        self.crosshair_lines = []
    
    def reset_marking(self, side):
        """重置标记"""
        self._drag_idx = None
        if side == "front":
            self.front_points = []
        elif side == "back":
            self.back_points = []
        else:
            self.card_points = []
        
        # 重新显示图片
        self.display_image_on_canvas()
        
        self.update_status("标记已重置，请重新开始标记")
    
    def mark_point(self, event):
        """标记点"""
        if not self.is_marking:
            # 如果未开始标记，自动开始
            self.is_marking = True
            if self.current_marking_side == "front":
                side_name = "身份证人像面"
            elif self.current_marking_side == "back":
                side_name = "身份证国徽面"
            else:
                side_name = "银行卡"
            self.update_status(f"已开始标记{side_name}，请点击4个角点（顺序不限）")
            
        side = self.current_marking_side
        
        if side == "front":
            points = self.front_points
        elif side == "back":
            points = self.back_points
        else:
            points = self.card_points
        
        # 获取画布坐标（考虑滚动）
        canvas_x = self.mark_canvas.canvasx(event.x)
        canvas_y = self.mark_canvas.canvasy(event.y)
        
        # 转换坐标到原始图像坐标（考虑缩放）
        img_x = canvas_x / self.current_scale
        img_y = canvas_y / self.current_scale
        # 自由标角：点中已有角点附近 → 抓起该点拖动，不新增点
        if points:
            _thr = 14 / self.current_scale if self.current_scale else 14
            for _i, (_px, _py) in enumerate(points):
                if ((_px - img_x) ** 2 + (_py - img_y) ** 2) ** 0.5 <= _thr:
                    self._drag_idx = _i
                    _iw, _ih = self.original_image.size
                    points[_i] = (int(max(0, min(img_x, _iw - 1))),
                                  int(max(0, min(img_y, _ih - 1))))
                    self.display_image_on_canvas()
                    self.update_status('正在拖动角点，松开鼠标放开')
                    return
        
        if len(points) >= 4:
            messagebox.showinfo("提示", "已选择4个点，可以点击'完成标记'或按回车键")
            return
            
        
        # 确保坐标在图像范围内
        img_width, img_height = self.original_image.size
        if 0 <= img_x < img_width and 0 <= img_y < img_height:
            points.append((int(img_x), int(img_y)))
            if len(points) == 4:
                points[:] = _order_quad_points(points)
            
            # 重新显示图片以更新标记点
            self.display_image_on_canvas()
            
            self.update_status(f"已选择 {len(points)}/4 个点")
            
            # 如果选择了4个点，询问是否完成
            if len(points) == 4:
                # 自动启用完成标记按钮
                if self.finish_marking_btn:
                    self.finish_marking_btn.config(state="normal")
                
                # 提示用户可以使用回车键
                response = messagebox.askyesno("完成标记", "已选择4个点！\n\n是否完成标记并开始处理？\n\n提示：也可以按回车键完成标记")
                if response:
                    self.finish_marking(side)
    

    def drag_mark_point(self, event):
        """标记模式下拖动已放下的角点（自由标角）"""
        if getattr(self, '_drag_idx', None) is None or getattr(self, 'adjust_mode', False):
            return
        if not self.mark_canvas:
            return
        side = self.current_marking_side
        if side == "front":
            points = self.front_points
        elif side == "back":
            points = self.back_points
        else:
            points = self.card_points
        if self._drag_idx >= len(points):
            return
        canvas_x = self.mark_canvas.canvasx(event.x)
        canvas_y = self.mark_canvas.canvasy(event.y)
        img_x = canvas_x / self.current_scale
        img_y = canvas_y / self.current_scale
        img_width, img_height = self.original_image.size
        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))
        new_point = (int(img_x), int(img_y))
        old_point = points[self._drag_idx]
        if abs(new_point[0] - old_point[0]) < 2 and abs(new_point[1] - old_point[1]) < 2:
            return
        points[self._drag_idx] = new_point
        self.display_image_on_canvas()
    
    def end_drag_mark_point(self, event):
        """结束拖动角点"""
        if getattr(self, '_drag_idx', None) is None:
            return
        self._drag_idx = None
        try:
            side = self.current_marking_side
            if side == "front":
                points = self.front_points
            elif side == "back":
                points = self.back_points
            else:
                points = self.card_points
            self.update_status(f"已选择 {len(points)}/4 个点")
        except Exception:
            pass

    def _on_enhance_change(self):
        if self.enhance_var is not None:
            self.enhance_mode = self.enhance_var.get()


    def apply_enhancement(self, img):
        """按当前增强模式处理已拉正的证件图；失败时原样返回"""
        mode = getattr(self, 'enhance_mode', 'color')
        try:
            if mode == 'bw':
                return _enhance_idcard_bw(img)
            if mode == 'docbw':
                return _enhance_document_bw(img)
            if mode == 'doccolor':
                return _enhance_document_color(img)
            if mode == 'color':
                return _enhance_idcard_color(img)
        except Exception:
            return img
        return img


    def finish_marking(self, side):
        """完成标记并处理图片"""
        if side == "front":
            points = self.front_points
            image = self.front_image
            target_var = "front_processed"
            side_name = "身份证人像面"
        elif side == "back":
            points = self.back_points
            image = self.back_image
            target_var = "back_processed"
            side_name = "身份证国徽面"
        else:
            points = self.card_points
            image = self.card_image
            target_var = "card_processed"
            side_name = "银行卡"
        
        if len(points) != 4:
            messagebox.showwarning("警告", "请先标记4个点")
            return

        # 自由标角：点击顺序不限，统一排成 左上→右上→右下→左下
        points[:] = _order_quad_points(points)

        # 角点合法性把关：四点重合或围出的面积太小，拉正必然失败，先给人话提示
        try:
            _pts = [(float(p[0]), float(p[1])) for p in points]
            _area = 0.0
            for _i in range(4):
                _x1, _y1 = _pts[_i]
                _x2, _y2 = _pts[(_i + 1) % 4]
                _area += _x1 * _y2 - _x2 * _y1
            if abs(_area) / 2.0 < 200:
                messagebox.showwarning("警告", "标记的4个角点太挤或有重合，请重新点击4个角")
                return
        except Exception:
            pass
        
        try:
            # 执行透视变换
            processed = self.perspective_transform(image, points)
            
            # 增强处理（对齐 IDCardMerge：透视校正后、圆角前）
            processed = self.apply_enhancement(processed)
            
            # 应用3.18mm外圆角效果，透明背景（不再是白色）
            processed_with_rounded = self.add_rounded_corners(processed)
            
            # 保存处理结果
            setattr(self, target_var, processed_with_rounded)
            
            # 在A4区域显示处理结果
            self.show_processed_result(side, processed_with_rounded)
            
            # 关闭标记窗口
            if self.mark_window:
                self.mark_window.destroy()
                self.mark_window = None
            
            self.update_status(f"{side_name}处理完成")
            
            # 批量模式：同步回当前组并刷新格子，不再弹 A4 询问
            if getattr(self, 'current_open', None) is not None:
                self._stash_workspace()
                self.refresh_pairs_panel()
                return
            
            # 根据不同情况询问用户
            if side in ["front", "back"]:
                # 处理身份证时
                front_done = hasattr(self, 'front_processed') and self.front_processed is not None
                back_done = hasattr(self, 'back_processed') and self.back_processed is not None
                
                if front_done and back_done:
                    # 身份证人像面和国徽面都处理完成
                    choice = messagebox.askyesnocancel("处理完成", 
                                                      "身份证人像面和国徽面都已处理完成！\n\n"
                                                      "点击'是'生成A4排版\n"
                                                      "点击'否'切换到银行卡模式添加银行卡\n"
                                                      "点击'取消'继续其他操作")
                    
                    if choice is True:
                        self.generate_a4_layout()
                    elif choice is False:
                        # 切换到银行卡模式
                        self.toggle_card_mode()
            
            elif side == "card":
                # 处理银行卡时，询问是否生成A4排版
                response = messagebox.askyesno("处理完成", "银行卡处理完成，是否立即生成A4排版？")
                if response:
                    self.generate_a4_layout()
            
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {str(e)}")
    
    def show_processed_result(self, side, processed_image):
        """在A4区域显示处理结果，并添加点击事件"""
        if not hasattr(self, 'canvas_info'):
            return
            
        info = self.canvas_info
        
        if side == "front":
            # 删除之前的预览区域
            if hasattr(self, 'front_preview_area'):
                self.a4_canvas.delete(self.front_preview_area)
            
            # 创建处理结果预览
            preview_img = processed_image.copy()
            preview_width = info['id_width'] - 30
            preview_height = info['front_preview_height']
            preview_img.thumbnail((int(preview_width), int(preview_height)), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(preview_img, master=self.root)
            
            # 保存图片引用
            self.preview_images['front'] = photo
            
            # 在预览区域显示处理结果
            image_id = self.a4_canvas.create_image(
                info['margin_x'] + info['id_width']/2,
                info['front_preview_y'] + info['front_preview_height']/2,
                image=photo, anchor=tk.CENTER
            )
            
            # 创建透明矩形覆盖在图片上，用于点击事件
            click_area = self.a4_canvas.create_rectangle(
                info['margin_x'] + 15, info['front_preview_y'],
                info['margin_x'] + info['id_width'] - 15, 
                info['front_preview_y'] + info['front_preview_height'],
                outline="", fill="", width=0  # 透明矩形
            )
            
            # 保存点击区域ID
            self.preview_click_areas['front'] = click_area
            
            # 绑定点击事件
            self.a4_canvas.tag_bind(click_area, "<Button-1>", 
                                   lambda e: self.reopen_for_adjustment("front"))
            
            # 更新人像面区域边框颜色（表示已处理）
            self.a4_canvas.itemconfig(self.front_rect, outline="#0D47A1")
            
        elif side == "back":
            # 删除之前的预览区域
            if hasattr(self, 'back_preview_area'):
                self.a4_canvas.delete(self.back_preview_area)
            
            # 创建处理结果预览
            preview_img = processed_image.copy()
            preview_width = info['id_width'] - 30
            preview_height = info['back_preview_height']
            preview_img.thumbnail((int(preview_width), int(preview_height)), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(preview_img, master=self.root)
            
            # 保存图片引用
            self.preview_images['back'] = photo
            
            # 在预览区域显示处理结果
            image_id = self.a4_canvas.create_image(
                info['margin_x'] + info['id_width']/2,
                info['back_preview_y'] + info['back_preview_height']/2,
                image=photo, anchor=tk.CENTER
            )
            
            # 创建透明矩形覆盖在图片上，用于点击事件
            click_area = self.a4_canvas.create_rectangle(
                info['margin_x'] + 15, info['back_preview_y'],
                info['margin_x'] + info['id_width'] - 15, 
                info['back_preview_y'] + info['back_preview_height'],
                outline="", fill="", width=0  # 透明矩形
            )
            
            # 保存点击区域ID
            self.preview_click_areas['back'] = click_area
            
            # 绑定点击事件
            self.a4_canvas.tag_bind(click_area, "<Button-1>", 
                                   lambda e: self.reopen_for_adjustment("back"))
            
            # 更新国徽面区域边框颜色（表示已处理）
            self.a4_canvas.itemconfig(self.back_rect, outline="#880E4F")
            
        elif side == "card":
            # 删除之前的预览区域
            if hasattr(self, 'card_preview_area'):
                self.a4_canvas.delete(self.card_preview_area)
            
            # 创建处理结果预览
            preview_img = processed_image.copy()
            preview_width = info.get('card_width', info.get('id_width', 0)) - 30
            preview_height = info.get('card_preview_height', info.get('id_height', 0) - 30)
            preview_img.thumbnail((int(preview_width), int(preview_height)), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(preview_img, master=self.root)
            
            # 保存图片引用
            self.preview_images['card'] = photo
            
            # 在预览区域显示处理结果
            image_id = self.a4_canvas.create_image(
                info.get('card_margin_x', info.get('margin_x', 0)) + info.get('card_width', info.get('id_width', 0))/2,
                info['card_preview_y'] + info['card_preview_height']/2,
                image=photo, anchor=tk.CENTER
            )
            
            # 创建透明矩形覆盖在图片上，用于点击事件
            click_area = self.a4_canvas.create_rectangle(
                info.get('card_margin_x', info.get('margin_x', 0)) + 15, info['card_preview_y'],
                info.get('card_margin_x', info.get('margin_x', 0)) + info.get('card_width', info.get('id_width', 0)) - 15, 
                info['card_preview_y'] + info['card_preview_height'],
                outline="", fill="", width=0  # 透明矩形
            )
            
            # 保存点击区域ID
            self.preview_click_areas['card'] = click_area
            
            # 绑定点击事件
            self.a4_canvas.tag_bind(click_area, "<Button-1>", 
                                   lambda e: self.reopen_for_adjustment("card"))
            
            # 更新银行卡区域边框颜色（表示已处理）
            self.a4_canvas.itemconfig(self.card_rect, outline="#2E7D32")
    
    def reopen_for_adjustment(self, side):
        """重新打开标记窗口进行调整"""
        # 根据side获取原始图片和点
        if side == "front":
            if not hasattr(self, 'front_image') or self.front_image is None:
                messagebox.showwarning("警告", "没有找到原始图片，请重新上传")
                return
            image = self.front_image
            points = self.front_points
            file_path = self.front_image_path
            side_name = "身份证人像面"
        elif side == "back":
            if not hasattr(self, 'back_image') or self.back_image is None:
                messagebox.showwarning("警告", "没有找到原始图片，请重新上传")
                return
            image = self.back_image
            points = self.back_points
            file_path = self.back_image_path
            side_name = "身份证国徽面"
        else:  # card
            if not hasattr(self, 'card_image') or self.card_image is None:
                messagebox.showwarning("警告", "没有找到原始图片，请重新上传")
                return
            image = self.card_image
            points = self.card_points
            file_path = self.card_image_path
            side_name = "银行卡"
        
        # 重新显示标记窗口，并传入已有的点
        self.show_image_for_marking(side, image, file_path, existing_points=points)
        
        # 如果已有4个点，直接进入调整模式
        if len(points) == 4:
            self.mark_window.after(100, lambda: self.enter_adjust_mode(side))
        
        self.update_status(f"重新打开{side_name}进行标记点调整")
    
    def add_rounded_corners(self, image):
        """添加3.18mm外圆角效果到图片，背景为透明（不再是白色）"""
        if self.corner_radius <= 0:
            return image.copy()
        
        # 获取原始图片尺寸
        width, height = image.size
        radius = min(self.corner_radius, width//2, height//2)
        
        # 如果原图是RGB模式，转换为RGBA模式以支持透明度
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # 创建一个透明背景的圆角图像
        rounded_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        
        # 创建一个圆角蒙版
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # 绘制圆角矩形作为蒙版
        draw.rounded_rectangle([0, 0, width-1, height-1], radius=radius, fill=255)
        
        # 应用蒙版
        rounded_image.paste(image, (0, 0), mask)
        
        return rounded_image
    
    def perspective_transform(self, image, points):
        """透视变换"""
        # 转换为OpenCV格式
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 定义源点（用户选择的点）
        src_points = np.array(points, dtype=np.float32)
        
        # 计算目标点（证件标准尺寸）
        if self.current_marking_side == "card":
            width_mm = self.card_width
            height_mm = self.card_height
        else:
            width_mm = self.id_width
            height_mm = self.id_height
            
        width_px = int(width_mm * self.mm_to_pixel)
        height_px = int(height_mm * self.mm_to_pixel)
        
        dst_points = np.array([
            [0, 0],
            [width_px, 0],
            [width_px, height_px],
            [0, height_px]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 执行透视变换
        result = cv2.warpPerspective(img_cv, matrix, (width_px, height_px))
        
        # 转换回PIL格式
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    
    def generate_a4_layout(self):
        """生成A4排版 - 不再询问背景，使用当前选择的背景"""
        # 检查是否有处理过的图片
        front_available = hasattr(self, 'front_processed') and self.front_processed is not None
        back_available = hasattr(self, 'back_processed') and self.back_processed is not None
        card_available = hasattr(self, 'card_processed') and self.card_processed is not None
        
        if not front_available and not back_available and not card_available:
            messagebox.showwarning("警告", "请先处理至少一张图片")
            return
            
        try:
            # 使用当前背景设置
            use_transparent = self.use_transparent
            
            # 计算A4尺寸（像素）
            a4_width_px = int(self.a4_width * self.mm_to_pixel)
            a4_height_px = int(self.a4_height * self.mm_to_pixel)
            
            # 创建A4画布
            if use_transparent:
                a4_image = Image.new('RGBA', (a4_width_px, a4_height_px), (255, 255, 255, 0))
            else:
                a4_image = Image.new('RGB', (a4_width_px, a4_height_px), 'white')
            
            # 证件尺寸（像素）
            id_width_px = int(self.id_width * self.mm_to_pixel)
            id_height_px = int(self.id_height * self.mm_to_pixel)
            
            # 增加A4排版证件之间的距离 - 从10mm增加到20mm
            spacing = int(20 * self.mm_to_pixel)  # 增加到20mm间距
            
            # 确定图片数量和类型
            images = []
            if front_available:
                images.append(("front", self.front_processed, "身份证人像面"))
            if back_available:
                images.append(("back", self.back_processed, "身份证国徽面"))
            if card_available:
                images.append(("card", self.card_processed, "银行卡"))
            
            # 计算总高度和起始位置
            total_height = len(images) * id_height_px + (len(images) - 1) * spacing
            start_y = (a4_height_px - total_height) // 2
            
            # 居中放置图片
            for i, (side, img, name) in enumerate(images):
                # 确保图片尺寸正确
                if img.size != (id_width_px, id_height_px):
                    img = img.resize((id_width_px, id_height_px), Image.Resampling.LANCZOS)
                
                # 计算位置（水平居中）
                x_pos = (a4_width_px - id_width_px) // 2
                y_pos = start_y + i * (id_height_px + spacing)
                
                # 如果是白色背景，需要特殊处理透明图片
                if not use_transparent:
                    # 对于白色背景，确保图片是RGB模式
                    if img.mode == 'RGBA':
                        # 创建白色背景的RGB图像
                        rgb_image = Image.new('RGB', img.size, (255, 255, 255))
                        # 将RGBA图像粘贴到白色背景上，透明区域会自动填充为白色
                        rgb_image.paste(img, (0, 0), img if img.mode == 'RGBA' else None)
                        img = rgb_image
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 直接粘贴到白色背景上
                    a4_image.paste(img, (x_pos, y_pos))
                else:
                    # 对于透明背景，确保图片是RGBA模式
                    if img.mode != 'RGBA':
                        img = img.convert('RGBA')
                    
                    # 使用alpha通道粘贴，保持透明背景
                    a4_image.paste(img, (x_pos, y_pos), img)
            
            # 保存A4排版图片
            self.a4_layout_image = a4_image
            self.a4_use_transparent = use_transparent
            self.a4_images_info = images  # 保存图片信息供PDF生成使用
            
            # 显示A4排版在画布上
            self.show_a4_layout_on_canvas(a4_image)
            
            self.update_status("A4排版已生成，可以保存或打印")
            
        except Exception as e:
            messagebox.showerror("错误", f"生成A4排版失败: {str(e)}")
    
    def show_a4_layout_on_canvas(self, a4_image):
        """在A4画布上显示排版结果"""
        # 清空画布
        self.a4_canvas.delete("all")
        self.is_in_edit_mode = False
        
        canvas_width = self.a4_canvas.winfo_width()
        canvas_height = self.a4_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 780
            canvas_height = 700
        
        # 缩放图片以适应画布
        display_img = a4_image.copy()
        display_img.thumbnail((canvas_width-20, canvas_height-20), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_img, master=self.root)
        
        # 保存图片引用
        self.a4_preview_img = photo
        
        # 在画布上显示A4排版
        self.a4_canvas.create_image(
            canvas_width/2, canvas_height/2,
            image=photo, anchor=tk.CENTER
        )
        
        # 绘制A4边框
        border_width = 1
        self.a4_canvas.create_rectangle(
            border_width, border_width,
            canvas_width - border_width, canvas_height - border_width,
            outline="#CCCCCC", width=border_width
        )
        
        # 添加标题
        self.a4_canvas.create_text(
            canvas_width/2, 20,
            text="A4排版预览 - 已就绪，可以保存或打印",
            fill="green", font=("Arial", 14, "bold")
        )
        
        # 添加返回按钮
        back_btn = tk.Button(self.a4_frame, text="返回编辑",
                            command=self.draw_a4_layout,
                            bg="#4CAF50", fg="white",
                            font=("Arial", 10), padx=15, pady=5)
        self.a4_canvas.create_window(
            canvas_width - 80, canvas_height - 30,
            window=back_btn
        )
        
        # 添加保存和打印按钮
        save_btn = tk.Button(self.a4_frame, text="保存图片",
                           command=self.save_a4_image,
                           bg="#2196F3", fg="white",
                           font=("Arial", 10), padx=15, pady=5)
        self.a4_canvas.create_window(
            canvas_width/2 - 150, canvas_height - 30,
            window=save_btn
        )
        
        # 添加保存PDF按钮
        save_pdf_btn = tk.Button(self.a4_frame, text="保存PDF",
                               command=self.save_a4_pdf,
                               bg="#9C27B0", fg="white",
                               font=("Arial", 10), padx=15, pady=5)
        self.a4_canvas.create_window(
            canvas_width/2, canvas_height - 30,
            window=save_pdf_btn
        )
        
        print_btn = tk.Button(self.a4_frame, text="打印图片",
                            command=self.print_a4_image,
                            bg="#FF9800", fg="white",
                            font=("Arial", 10), padx=15, pady=5)
        self.a4_canvas.create_window(
            canvas_width/2 + 150, canvas_height - 30,
            window=print_btn
        )
        
        # 背景选择区域
        bg_frame = tk.Frame(self.a4_frame, bg="white")
        
        tk.Label(bg_frame, text="背景:", bg="white").pack(side=tk.LEFT, padx=5)
        
        # 背景选择单选按钮
        self.bg_var = tk.StringVar(value="white")  # 默认白色背景
        white_rb = tk.Radiobutton(bg_frame, text="白色", variable=self.bg_var, 
                                 value="white", bg="white", command=self.change_background)
        white_rb.pack(side=tk.LEFT)
        
        trans_rb = tk.Radiobutton(bg_frame, text="透明", variable=self.bg_var, 
                                 value="transparent", bg="white", command=self.change_background)
        trans_rb.pack(side=tk.LEFT)
        
        self.a4_canvas.create_window(
            canvas_width/2, canvas_height - 70,
            window=bg_frame
        )
        
        # 重新生成按钮
        regen_btn = tk.Button(self.a4_frame, text="重新生成",
                            command=self.regenerate_a4_layout,
                            bg="#607D8B", fg="white",
                            font=("Arial", 10), padx=15, pady=5)
        self.a4_canvas.create_window(
            80, canvas_height - 30,
            window=regen_btn
        )
        
        # 编辑文字按钮
        text_btn = tk.Button(self.a4_frame, text="编辑文字",
                           command=self.toggle_text_edit,
                           bg="#795548", fg="white",
                           font=("Arial", 10), padx=15, pady=5)
        self.a4_canvas.create_window(
            80, canvas_height - 70,
            window=text_btn
        )
        
        # 单独保存按钮
        save_single_btn = tk.Button(self.a4_frame, text="单独保存",
                                  command=self.show_single_save_menu,
                                  bg="#FF5722", fg="white",
                                  font=("Arial", 10), padx=15, pady=5)
        self.a4_canvas.create_window(
            canvas_width - 80, canvas_height - 70,
            window=save_single_btn
        )
    
    def change_background(self):
        """切换背景"""
        if self.bg_var.get() == "transparent":
            self.use_transparent = True
            self.background_color = "transparent"
            self.update_status("已切换到透明背景")
        else:
            self.use_transparent = False
            self.background_color = "white"
            self.update_status("已切换到白色背景")
        
        # 如果已有A4排版，重新生成
        if hasattr(self, 'a4_layout_image'):
            self.regenerate_a4_layout()
    
    def show_single_save_menu(self):
        """显示单独保存菜单"""
        menu = tk.Menu(self.root, tearoff=0)
        
        if hasattr(self, 'front_processed') and self.front_processed is not None:
            menu.add_command(label="保存人像面", command=lambda: self.save_single_side("front"))
        
        if hasattr(self, 'back_processed') and self.back_processed is not None:
            menu.add_command(label="保存国徽面", command=lambda: self.save_single_side("back"))
        
        if hasattr(self, 'card_processed') and self.card_processed is not None:
            menu.add_command(label="保存银行卡", command=lambda: self.save_single_side("card"))
        
        if menu.index("end") is not None:
            menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
    
    def regenerate_a4_layout(self):
        """重新生成A4排版（使用当前背景设置）"""
        self.generate_a4_layout()
    
    def save_a4_image(self):
        """保存A4图片（根据当前背景设置保存）"""
        if not hasattr(self, 'a4_layout_image'):
            messagebox.showwarning("警告", "请先生成A4排版")
            return
        
        # 根据当前背景设置确定默认扩展名
        default_ext = ".png" if self.use_transparent else ".jpg"
        filetypes = [("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("所有文件", "*.*")]
        
        file_path = filedialog.asksaveasfilename(
            title="保存A4图片",
            defaultextension=default_ext,
            filetypes=filetypes
        )
        
        if file_path:
            try:
                # 根据文件扩展名保存
                ext = os.path.splitext(file_path.lower())[1]
                if ext == '.png':
                    # PNG格式，如果原图不是RGBA模式，转换为RGBA
                    if self.a4_layout_image.mode != 'RGBA':
                        save_image = self.a4_layout_image.convert('RGBA')
                    else:
                        save_image = self.a4_layout_image
                    save_image.save(file_path, dpi=(self.dpi, self.dpi))
                else:
                    # JPG格式，确保是RGB模式
                    if self.a4_layout_image.mode != 'RGB':
                        save_image = self.a4_layout_image.convert('RGB')
                    else:
                        save_image = self.a4_layout_image
                    save_image.save(file_path, dpi=(self.dpi, self.dpi), quality=95)
                    
                self.update_status(f"A4图片已保存: {file_path}")
                messagebox.showinfo("成功", f"A4图片已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def save_a4_pdf(self):
        """保存A4图片为PDF - 使用img2pdf库"""
        if not hasattr(self, 'a4_layout_image'):
            messagebox.showwarning("警告", "请先生成A4排版")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存A4 PDF",
            defaultextension=".pdf",
            filetypes=[("PDF文件", "*.pdf"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                # 创建临时文件保存图片
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    temp_file = tmp.name
                    # 保存为JPG格式（PDF不支持透明背景）
                    if self.a4_layout_image.mode != 'RGB':
                        save_image = self.a4_layout_image.convert('RGB')
                    else:
                        save_image = self.a4_layout_image
                    save_image.save(temp_file, dpi=(self.dpi, self.dpi), quality=100)
                
                # 使用img2pdf将图片转换为PDF
                with open(file_path, "wb") as pdf_file:
                    pdf_file.write(img2pdf.convert(temp_file))
                
                # 删除临时文件
                os.unlink(temp_file)
                
                self.update_status(f"A4 PDF已保存: {file_path}")
                messagebox.showinfo("成功", f"A4 PDF已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存PDF失败: {str(e)}\n\n请确保已安装img2pdf库: pip install img2pdf")
    
    def save_single_side(self, side):
        """单独保存人像面、国徽面或银行卡的图片"""
        if side == "front":
            image = self.front_processed
            side_name = "身份证人像面"
        elif side == "back":
            image = self.back_processed
            side_name = "身份证国徽面"
        else:
            image = self.card_processed
            side_name = "银行卡"
        
        if image is None:
            messagebox.showwarning("警告", f"请先处理{side_name}图片")
            return
        
        file_path = filedialog.asksaveasfilename(
            title=f"保存{side_name}",
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("JPEG文件", "*.jpg"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                # 根据文件扩展名保存
                ext = os.path.splitext(file_path.lower())[1]
                if ext == '.png':
                    # PNG格式，确保是RGBA模式以保持透明背景
                    if image.mode != 'RGBA':
                        save_image = image.convert('RGBA')
                    else:
                        save_image = image
                    save_image.save(file_path, dpi=(self.dpi, self.dpi))
                else:
                    # JPG格式，确保是RGB模式（透明背景会转换为白色）
                    if image.mode != 'RGB':
                        # 创建一个白色背景的RGB图像
                        rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                        # 将RGBA图像粘贴到白色背景上
                        if image.mode == 'RGBA':
                            rgb_image.paste(image, (0, 0), image)
                        else:
                            rgb_image.paste(image, (0, 0))
                        save_image = rgb_image
                    else:
                        save_image = image
                    save_image.save(file_path, dpi=(self.dpi, self.dpi), quality=95)
                    
                self.update_status(f"{side_name}图片已保存: {file_path}")
                messagebox.showinfo("成功", f"{side_name}图片已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def toggle_text_edit(self):
        """切换文本编辑模式"""
        if not hasattr(self, 'a4_layout_image'):
            messagebox.showwarning("警告", "请先生成A4排版")
            return
        
        self.text_edit_mode = not self.text_edit_mode
        
        if self.text_edit_mode:
            self.show_text_edit_window()
        else:
            if self.text_window:
                self.text_window.destroy()
                self.text_window = None
    
    def show_text_edit_window(self):
        """显示文本编辑窗口（可拖动版本）"""
        if self.text_window is not None:
            self.text_window.destroy()
        
        # 创建文本编辑窗口
        self.text_window = ToplevelWithIcon(self.root)
        self.text_window.title("添加文字 - 可在A4预览区域拖动文字")
        
        # 设置窗口尺寸
        self.text_window.geometry("500x400")
        self.center_window(self.text_window)
        
        # 主框架
        main_frame = ttk.Frame(self.text_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 文字输入区域
        ttk.Label(main_frame, text="输入要添加的文字:").pack(anchor=tk.W, pady=(0, 5))
        
        self.text_entry = tk.Text(main_frame, height=5, width=50, font=("Arial", 12))
        self.text_entry.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 字体设置区域
        font_frame = ttk.Frame(main_frame)
        font_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 字体大小选择
        ttk.Label(font_frame, text="字体大小:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.font_size = tk.IntVar(value=24)
        font_size_spin = ttk.Spinbox(font_frame, from_=10, to=50, textvariable=self.font_size, width=10)
        font_size_spin.pack(side=tk.LEFT, padx=(0, 20))
        
        # 字体颜色选择
        ttk.Label(font_frame, text="文字颜色:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.text_color = tk.StringVar(value="black")
        colors = [("黑色", "black"), ("红色", "red"), ("蓝色", "blue"), ("绿色", "green")]
        
        for text, value in colors:
            rb = ttk.Radiobutton(font_frame, text=text, variable=self.text_color, value=value)
            rb.pack(side=tk.LEFT, padx=5)
        
        # 位置设置区域
        pos_frame = ttk.Frame(main_frame)
        pos_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(pos_frame, text="初始位置:").pack(side=tk.LEFT, padx=(0, 10))
        
        # X坐标
        ttk.Label(pos_frame, text="X:").pack(side=tk.LEFT)
        self.pos_x = tk.IntVar(value=100)
        pos_x_spin = ttk.Spinbox(pos_frame, from_=0, to=1000, textvariable=self.pos_x, width=8)
        pos_x_spin.pack(side=tk.LEFT, padx=(0, 10))
        
        # Y坐标
        ttk.Label(pos_frame, text="Y:").pack(side=tk.LEFT)
        self.pos_y = tk.IntVar(value=100)
        pos_y_spin = ttk.Spinbox(pos_frame, from_=0, to=1000, textvariable=self.pos_y, width=8)
        pos_y_spin.pack(side=tk.LEFT)
        
        # 按钮区域
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(btn_frame, text="添加到图片并拖动", command=self.add_draggable_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.text_window.destroy).pack(side=tk.LEFT, padx=5)
        
        # 提示信息
        ttk.Label(main_frame, text="提示: 添加文字后可以在A4预览区域拖动文字调整位置", 
                 font=("Arial", 10, "italic"), foreground="gray").pack(pady=(10, 0))
        
        # 窗口关闭时退出文本编辑模式
        self.text_window.protocol("WM_DELETE_WINDOW", lambda: self.toggle_text_edit())
    
    def add_draggable_text(self):
        """添加可拖动的文字"""
        text = self.text_entry.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("警告", "请输入要添加的文字")
            return
        
        try:
            # 关闭文本编辑窗口
            if self.text_window:
                self.text_window.destroy()
                self.text_window = None
                self.text_edit_mode = False
            
            # 设置拖动模式
            self.dragging_text = {
                'text': text,
                'color': self.text_color.get(),
                'size': self.font_size.get(),
                'x': self.pos_x.get(),
                'y': self.pos_y.get(),
                'id': None  # 画布对象ID
            }
            
            # 在A4画布上添加可拖动文字
            self.add_text_to_canvas()
            
            self.update_status("文字已添加到图片，可以拖动调整位置")
            
        except Exception as e:
            messagebox.showerror("错误", f"添加文字失败: {str(e)}")
    
    def add_text_to_canvas(self):
        """在A4画布上添加可拖动文字"""
        if not hasattr(self, 'dragging_text') or self.dragging_text is None:
            return
        
        # 清空之前的文字对象
        for text_obj in self.text_objects:
            self.a4_canvas.delete(text_obj['id'])
        self.text_objects = []
        
        # 添加新文字对象
        text_id = self.a4_canvas.create_text(
            self.dragging_text['x'],
            self.dragging_text['y'],
            text=self.dragging_text['text'],
            fill=self.dragging_text['color'],
            font=("Arial", self.dragging_text['size'], "bold"),
            anchor=tk.NW
        )
        
        # 保存文字对象信息
        self.text_objects.append({
            'id': text_id,
            'text': self.dragging_text['text'],
            'color': self.dragging_text['color'],
            'size': self.dragging_text['size'],
            'x': self.dragging_text['x'],
            'y': self.dragging_text['y']
        })
        
        # 绑定拖动事件
        self.a4_canvas.tag_bind(text_id, "<Button-1>", self.start_text_drag)
        self.a4_canvas.tag_bind(text_id, "<B1-Motion>", self.drag_text)
        self.a4_canvas.tag_bind(text_id, "<ButtonRelease-1>", self.stop_text_drag)
        
        # 更新文字位置信息
        self.dragging_text['id'] = text_id
    
    def start_text_drag(self, event):
        """开始拖动文字"""
        self.text_start_x = event.x
        self.text_start_y = event.y
        
        # 高亮显示当前拖动的文字
        if self.dragging_text and self.dragging_text['id']:
            self.a4_canvas.itemconfig(self.dragging_text['id'], fill="red")
    
    def drag_text(self, event):
        """拖动文字"""
        if self.dragging_text and self.dragging_text['id']:
            # 计算移动距离
            dx = event.x - self.text_start_x
            dy = event.y - self.text_start_y
            
            # 移动文字
            self.a4_canvas.move(self.dragging_text['id'], dx, dy)
            
            # 更新起始位置
            self.text_start_x = event.x
            self.text_start_y = event.y
            
            # 更新文字位置信息
            if self.text_objects:
                for text_obj in self.text_objects:
                    if text_obj['id'] == self.dragging_text['id']:
                        coords = self.a4_canvas.coords(text_obj['id'])
                        text_obj['x'] = coords[0]
                        text_obj['y'] = coords[1]
    
    def stop_text_drag(self, event):
        """停止拖动文字"""
        if self.dragging_text and self.dragging_text['id']:
            # 恢复文字颜色
            self.a4_canvas.itemconfig(self.dragging_text['id'], fill=self.dragging_text['color'])
    
    def apply_text_to_image(self):
        """将文字应用到图片"""
        if not self.text_objects:
            messagebox.showinfo("提示", "没有要添加的文字")
            return
        
        try:
            # 创建临时副本
            text_image = self.a4_layout_image.copy()
            
            # 创建绘图对象
            draw = ImageDraw.Draw(text_image)
            
            # 获取A4图片的尺寸
            img_width, img_height = text_image.size
            
            # 获取画布上文字的缩放比例
            canvas_width = self.a4_canvas.winfo_width()
            canvas_height = self.a4_canvas.winfo_height()
            
            # 计算缩放比例
            scale_x = img_width / canvas_width
            scale_y = img_height / canvas_height
            
            # 设置字体
            try:
                # 尝试使用系统字体
                font_paths = ["simhei.ttf", "arial.ttf", "C:/Windows/Fonts/msyh.ttc"]
                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, self.text_objects[0]['size'])
                        break
                    except:
                        continue
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # 添加所有文字
            for text_obj in self.text_objects:
                # 缩放坐标
                x = text_obj['x'] * scale_x
                y = text_obj['y'] * scale_y
                
                # 添加文字
                draw.text((x, y), text_obj['text'], font=font, fill=text_obj['color'])
            
            # 更新A4图片
            self.a4_layout_image = text_image
            
            # 更新预览
            self.show_a4_layout_on_canvas(text_image)
            
            self.update_status("文字已应用到图片")
            
        except Exception as e:
            messagebox.showerror("错误", f"应用文字失败: {str(e)}")
    
    def print_a4_image(self):
        """打印A4图片"""
        if not hasattr(self, 'a4_layout_image'):
            messagebox.showwarning("警告", "请先生成A4排版")
            return
        
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_file = tmp.name
                # 保存为PNG格式
                self.a4_layout_image.save(temp_file, dpi=(self.dpi, self.dpi))
            
            # 根据操作系统调用打印命令
            system = platform.system()
            if system == 'Windows':
                # Windows系统
                os.startfile(temp_file, "print")
                messagebox.showinfo("打印", "打印任务已发送到打印机")
            elif system == 'Darwin':
                # macOS系统
                subprocess.run(["lpr", temp_file])
                messagebox.showinfo("打印", "打印任务已发送到打印机")
            else:
                # Linux系统
                subprocess.run(["lp", temp_file])
                messagebox.showinfo("打印", "打印任务已发送到打印机")
            
            # 稍后删除临时文件
            self.root.after(5000, lambda: os.unlink(temp_file) if os.path.exists(temp_file) else None)
            
            self.update_status("打印任务已发送")
            
        except Exception as e:
            messagebox.showerror("错误", f"打印失败: {str(e)}")
    
    def clear_all(self):
        """清除所有数据"""
        self.front_image = None
        self.back_image = None
        self.pairs = []
        self.current_open = None
        self._pair_photos = {}
        if hasattr(self, 'pairs_list_frame'):
            self.refresh_pairs_panel()
        self.card_image = None
        self.front_points = []
        self.back_points = []
        self.card_points = []
        self.front_processed = None
        self.back_processed = None
        self.card_processed = None
        self.is_marking = False
        self.adjust_mode = False
        self.adjust_points = []
        self.dragging_point_index = None
        self.point_circles = []
        self.preview_click_areas = {}
        self.preview_images = {}
        
        # 关闭标记窗口
        if self.mark_window:
            self.mark_window.destroy()
            self.mark_window = None
        
        # 关闭文本编辑窗口
        if self.text_window:
            self.text_window.destroy()
            self.text_window = None
        
        # 重置文本编辑模式
        self.text_edit_mode = False
        self.text_objects = []
        self.dragging_text = None
        
        # 重置银行卡模式
        self.card_mode = False
        self.card_mode_btn.config(text="银行卡功能")
        
        # 重置背景设置
        self.background_color = "white"
        self.use_transparent = False
        
        # 重新绘制A4布局
        self.draw_a4_layout()
        
        self.update_status("已清除所有数据，请重新上传图片")
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        self.root.update_idletasks()

def main():
    # 如果支持拖拽功能，使用TkinterDnD.Tk()
    if DRAG_DROP_SUPPORTED:
        try:
            root = TkinterDnD.Tk()
            # 设置窗口图标 - 修复任务栏图标
            set_window_icon(root)
        except:
            root = tk.Tk()
            # 设置窗口图标 - 修复任务栏图标
            set_window_icon(root)
            print("TkinterDnD初始化失败，使用标准tkinter")
    else:
        root = tk.Tk()

    # 钉死默认根为实际使用的 root：拖拽根创建中途失败回落时可能留下双根，
    # PhotoImage 默认挂靠第一个根会与界面所在根不是同一个，导致 image 不存在报错
    try:
        tk._default_root = root
    except Exception:
        pass

    app = IDCardProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()