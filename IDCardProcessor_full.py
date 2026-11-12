import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import tempfile
import subprocess
import sys
import time
import traceback
import platform
import shutil
import hashlib
import img2pdf

# refineCorners 开关（对齐原版 IDCardMerge v2.2：Detect 出角后紧接着在
# 原图上精修四角）。置 False 则原样使用模型角点。
REFINE_CORNERS = True

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
    优先用微软 ONNX Runtime 加载模型（与原版同路线），不可用时回落 OpenCV DNN。"""

    def __init__(self, onnxmodel, confThreshold=0.5, nmsThreshold=0.5):
        self.inpWidth, self.inpHeight = 640, 640
        self.confThreshold, self.nmsThreshold = confThreshold, nmsThreshold
        self._model_path = onnxmodel
        self._sess = None
        self._cv_net = None
        try:
            import onnxruntime as _ort
            # Reserve one core for the UI: with ORT's default
            # all-cores setting, detection saturated Qing's 4-core
            # i5-7400 and even the mouse cursor stuttered during the
            # recognition wait (2026-11-12). Detection behavior is
            # unchanged; only the thread budget is capped.
            _opts = _ort.SessionOptions()
            _opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
            _opts.inter_op_num_threads = 1
            self._sess = _ort.InferenceSession(onnxmodel, sess_options=_opts,
                                              providers=['CPUExecutionProvider'])
            self._iname = self._sess.get_inputs()[0].name
        except Exception:
            self._sess = None
            self._cv_net = cv2.dnn.readNet(onnxmodel)
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
                # Aligned with the original IDCardMerge letterbox: plain
                # truncation (cvttsd2si), no +1.
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
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
        if self._sess is not None:
            raw = self._sess.run(None, {self._iname: blob})
            groups = {1: [], 4: [], 8: []}
            for a in raw:
                groups[a.shape[2]].append((a.shape[1], a))
            for k in groups:
                groups[k] = [a for _, a in sorted(groups[k], key=lambda t: -t[0])]
            outs = []
            for i in range(3):
                outs += [groups[1][i], groups[4][i], groups[8][i]]
        else:
            self._cv_net.setInput(blob)
            outs = self._cv_net.forward(self._cv_net.getUnconnectedOutLayersNames())
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
            # Original keeps a candidate only when score > 0.5 (strict).
            pos_inds = np.where(scores > self.confThreshold)[0]
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
        raw = [(float(pts[i, 0]), float(pts[i, 1])) for i in range(4)]
        # Corner order, aligned with the original IDCardMerge v2.2: its
        # Detect returns the 4 keypoints rotated by one, [kp1, kp2, kp3,
        # kp0]. The model's RAW keypoint order is [BL, TL, TR, BR] --
        # measured 2026-11-12 on real front/back photos (raw kp0 always
        # lands on the card's bottom-left corner), so the rotation yields
        # exactly the canonical [TL, TR, BR, BL] this pipeline consumes.
        # (An older comment here claimed the raw order was already
        # TL-first; it is not -- downstream geometric re-sorts were
        # masking the one-position rotation.)
        pts_list = [raw[1], raw[2], raw[3], raw[0]]
        # Degenerate-case guard only: if the rotated quad self-crosses,
        # fall back to a geometric sort rather than emit a bow-tie.
        _arr = np.array(pts_list, dtype=np.float64)
        def _cr(o, p, q):
            return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
        def _cross(a, b, c, d):
            d1 = _cr(c, d, a); d2 = _cr(c, d, b); d3 = _cr(a, b, c); d4 = _cr(a, b, d)
            return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
        if _cross(_arr[0], _arr[1], _arr[2], _arr[3]) or _cross(_arr[1], _arr[2], _arr[3], _arr[0]):
            pts_ordered = _order_points_np(np.array(pts_list))
            return [(float(q[0]), float(q[1])) for q in pts_ordered]
        return pts_list


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


def _apply_watermark(page_img, text, font_px, opacity_pct, rows, color_rgb=(110, 110, 110)):
    """整页平铺斜向水印（对齐 IDCardMerge 的安全水印：文字/字号/透明度/行数）"""
    base = page_img.convert('RGBA')
    W, H = base.size
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_cjk_font(font_px)
    alpha = max(0, min(255, int(opacity_pct * 255 / 100)))
    color = (color_rgb[0], color_rgb[1], color_rgb[2], alpha)
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


# ===== OCR 文字识别（全能版新增：RapidOCR 的 ONNX 模型 + 微软 ORT 引擎运行） =====
import re as _re
import threading as _threading

_OCR_ENGINE = None
_OCR_LOCK = _threading.Lock()
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECKS = '10X98765432'


def _get_ocr_engine():
    global _OCR_ENGINE
    if _OCR_ENGINE is None:
        with _OCR_LOCK:
            if _OCR_ENGINE is None:
                # 打包后 rapidocr 的 load_image 会按 __file__ 去磁盘上找
                # load_image.pyc 实体文件，onefile 里它只在压缩包里、磁盘上
                # 没有，import 直接失败（ocr_error.log 实锤）；先在解压目录
                # 补一个同名空文件占位，让它的路径解析能过关。
                try:
                    _base = getattr(sys, '_MEIPASS', None)
                    if _base:
                        _p = os.path.join(_base, 'rapidocr_onnxruntime',
                                          'utils', 'load_image.pyc')
                        if not os.path.exists(_p):
                            os.makedirs(os.path.dirname(_p), exist_ok=True)
                            open(_p, 'wb').close()
                except Exception:
                    pass
                from rapidocr_onnxruntime import RapidOCR
                _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _id_checksum_ok(num):
    if len(num) != 18:
        return False
    try:
        total = sum(int(num[i]) * _ID_WEIGHTS[i] for i in range(17))
    except ValueError:
        return False
    return _ID_CHECKS[total % 11] == num[17].upper()


def _ocr_lines(pil_img):
    """对一张已拉正的卡面图做 OCR，返回按阅读顺序排好的文字行列表"""
    import numpy as _np
    engine = _get_ocr_engine()
    bg = Image.new('RGB', pil_img.size, (255, 255, 255))
    if pil_img.mode in ('RGBA', 'LA'):
        bg.paste(pil_img, mask=pil_img.split()[-1])
    else:
        bg = pil_img.convert('RGB')
    result, _elapse = engine(_np.asarray(bg))
    items = []
    if result:
        for box, text, score in result:
            if score < 0.5:
                continue
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            items.append((text.strip(), sum(ys) / len(ys), min(xs)))
    items.sort(key=lambda t: (round(t[1] / 20.0), t[2]))
    return [t for t, _cy, _x in items]


def _parse_id_front(texts):
    info = {}
    flat = ''.join(texts).replace(' ', '')
    m = _re.search(r'[0-9]{17}[0-9Xx]', flat)
    if m:
        num = m.group(0).upper()
        info['公民身份号码'] = num
        info['号码校验'] = '通过' if _id_checksum_ok(num) else '存疑'
        info['出生'] = '%s-%s-%s' % (num[6:10], num[10:12], num[12:14])
    for i, t in enumerate(texts):
        if '姓名' in t and '姓名' not in info:
            v = t.split('姓名', 1)[1].strip()
            if not v and i + 1 < len(texts):
                v = texts[i + 1]
            if v:
                info['姓名'] = v
        mm = _re.search(r'性别\s*([男女])', t)
        if mm:
            info['性别'] = mm.group(1)
        mm = _re.search(r'民族\s*(\S+)', t)
        if mm:
            info['民族'] = mm.group(1)
        if '出生' in t:
            mm = _re.search(r'(\d{4})\D{0,2}(\d{1,2})\D{0,2}(\d{1,2})',
                            t + (texts[i + 1] if i + 1 < len(texts) else ''))
            if mm:
                info['出生'] = '%s-%02d-%02d' % (mm.group(1), int(mm.group(2)), int(mm.group(3)))
    zi = next((i for i, t in enumerate(texts) if '住址' in t), None)
    hi = next((i for i, t in enumerate(texts) if '公民身份号码' in t), None)
    if zi is not None:
        first = texts[zi].split('住址', 1)[1].strip()
        parts = [first] if first else []
        end = hi if hi is not None else len(texts)
        for j in range(zi + 1, end):
            if texts[j] not in ('姓名', '性别', '民族', '出生'):
                parts.append(texts[j])
        if parts:
            info['住址'] = ''.join(parts)
    return info


def _parse_id_back(texts):
    info = {}
    for t in texts:
        if '签发机关' in t:
            v = t.split('签发机关', 1)[1].strip()
            if v:
                info['签发机关'] = v
        if '有效期限' in t:
            mm = _re.search(r'(\d{4})\.(\d{2})\.(\d{2})\s*-\s*(\d{4})\.(\d{2})\.(\d{2})', t)
            if mm:
                info['有效期限'] = '%s.%s.%s-%s.%s.%s' % mm.groups()
            else:
                v = t.split('有效期限', 1)[1].strip()
                if v:
                    info['有效期限'] = v
    return info


def _log_ocr_error(exc):
    """OCR 异常留痕：写到程序旁边的 ocr_error.log，避免窗口版无声失败。"""
    try:
        base = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base, 'ocr_error.log'), 'a', encoding='utf-8') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + repr(exc) + '\n')
            traceback.print_exc(file=f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# refineCorners 复刻（据 IDCardMerge v2.2 refineCorners 反汇编定案重写，
# 见 idcard-audit/original-detect-spec.md §9，全部 PROVEN；与
# idcard-qt/refine.py 同源，此处内嵌以便单文件分发）。
# 原版算法：对每条边取 n+1 个基点，n = clamp(int(L/20), 15, 60)（边长
# L < 10 不采样、直接用原边线）；每基点沿法线在 [-15,+15] 步长 1.0 横扫，
# 响应 = R/G/B 三通道各自完整 Sobel 梯度幅值 sqrt(Gx^2+Gy^2) 的最大值，
# 取首个达峰的偏移，峰值 > 30.0 才把浮点峰位收为边点；边点 >= 10 时跑
# RANSAC（1000 轮，每条边用新鲜的时间种子随机源）：候选线软评分
# sum(exp(-d^2/32))（仅 |d| < 4.0 的点计入），冠军仅在严格更优时替换
# （初始分 -1.0），冠军分 >= 0.15*m 才采纳、否则用原边线。四边成线后
# 相邻线求交得新角，|det| <= 1e-9 跳过，新角离原角 <= 25.0 才保留。
# 老机器保险丝：整体墙钟预算 _REFINE_TIME_BUDGET_S（本机尺度 0.6s），
# 超时即以"已完成的边用拟合线、未完成的边用原边线"收尾，绝不无界运行。
_REFINE_SCAN_LO, _REFINE_SCAN_HI, _REFINE_SCAN_STEP = -15.0, 15.0, 1.0
_REFINE_GRAD_MIN = 30.0
_REFINE_MIN_EDGE_LEN = 10.0
_REFINE_SAMPLE_DIV = 20.0
_REFINE_SAMPLE_MIN, _REFINE_SAMPLE_MAX = 15, 60
_REFINE_RANSAC_ITER = 1000
_REFINE_RANSAC_MIN_POINTS = 10
_REFINE_INLIER_DIST = 4.0
_REFINE_EXP_DENOM = 32.0
_REFINE_ACCEPT_RATIO = 0.15
_REFINE_MAX_CORNER_SHIFT = 25.0
_REFINE_DET_EPS = 1e-9
_REFINE_TIME_BUDGET_S = 0.6


def _refine_line_from_points(p, q):
    """过两点的归一化直线 (a, b, c)，a*x + b*y + c = 0。"""
    a = float(p[1] - q[1])
    b = float(q[0] - p[0])
    n = np.hypot(a, b)
    if n == 0:
        return None
    a, b = a / n, b / n
    c = -(a * p[0] + b * p[1])
    return np.array([a, b, c], dtype=np.float64)


def _refine_corners_np(img_rgb, corners, rng=None, stats=None):
    """img_rgb: HxWx3 uint8（RGB 序）；corners: (4,2) float，原图坐标。
    返回精修后的 (4,2) float64；任何一步不达标都退回原角（与原版一致）。
    stats 非 None 时回填 {'budget_hit': bool, 'time_s': float}。"""
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2):
        return corners.copy()
    t_start = time.monotonic()
    deadline = t_start + _REFINE_TIME_BUDGET_S
    h, w = img_rgb.shape[:2]
    chans = [img_rgb[:, :, c].astype(np.int32) for c in range(3)]
    offs = np.arange(_REFINE_SCAN_LO,
                     _REFINE_SCAN_HI + 0.5 * _REFINE_SCAN_STEP,
                     _REFINE_SCAN_STEP)

    def sobel_resp(xs, ys):
        """三通道完整 Sobel 幅值的最大值（原版响应函数）。"""
        best = None
        for ch in chans:
            gx = ((ch[ys - 1, xs + 1] + 2 * ch[ys, xs + 1] + ch[ys + 1, xs + 1])
                  - (ch[ys - 1, xs - 1] + 2 * ch[ys, xs - 1] + ch[ys + 1, xs - 1]))
            gy = ((ch[ys + 1, xs - 1] + 2 * ch[ys + 1, xs] + ch[ys + 1, xs + 1])
                  - (ch[ys - 1, xs - 1] + 2 * ch[ys - 1, xs] + ch[ys - 1, xs + 1]))
            mag = np.sqrt(gx.astype(np.float64) ** 2
                          + gy.astype(np.float64) ** 2)
            best = mag if best is None else np.maximum(best, mag)
        return best

    lines = []
    budget_hit = False
    for i in range(4):
        a_pt = corners[i]
        b_pt = corners[(i + 1) % 4]
        fallback = _refine_line_from_points(a_pt, b_pt)
        d = b_pt - a_pt
        length = float(np.hypot(d[0], d[1]))
        if length < _REFINE_MIN_EDGE_LEN or fallback is None or budget_hit:
            lines.append(fallback)
            continue
        if time.monotonic() > deadline:
            budget_hit = True
            lines.append(fallback)
            continue
        # 基点数 n+1，n = clamp(int(L/20), 15, 60)（原版 PROVEN）
        n = int(max(_REFINE_SAMPLE_MIN,
                    min(_REFINE_SAMPLE_MAX, length / _REFINE_SAMPLE_DIV)))
        u = d / length
        nrm = np.array([-u[1], u[0]])       # 法线 (-dy, dx)/L，正负无所谓
        t = np.arange(n + 1, dtype=np.float64) / n
        base = a_pt[None, :] + t[:, None] * d[None, :]              # (n+1, 2)
        pts = base[:, None, :] + offs[None, :, None] * nrm[None, None, :]
        xs = pts[:, :, 0].astype(np.int64)  # 向零截断（Go int() 语义）
        ys = pts[:, :, 1].astype(np.int64)
        valid = (xs >= 1) & (xs <= w - 2) & (ys >= 1) & (ys <= h - 2)
        resp = np.zeros(xs.shape, dtype=np.float64)
        if valid.any():
            resp[valid] = sobel_resp(xs[valid], ys[valid])
        best_idx = resp.argmax(axis=1)      # 首个达峰（与原版严格大于更新一致）
        best_val = resp[np.arange(resp.shape[0]), best_idx]
        keep = best_val > _REFINE_GRAD_MIN
        edge_pts = pts[keep, best_idx[keep]]

        line = fallback
        m = len(edge_pts)
        if m >= _REFINE_RANSAC_MIN_POINTS:
            # 原版每条边新建时间种子 RNG；调用方给 rng 时用它（测试可复现）
            edge_rng = rng if rng is not None else np.random.default_rng()
            best_line, best_score = None, -1.0
            for it in range(_REFINE_RANSAC_ITER):
                if it % 16 == 0 and time.monotonic() > deadline:
                    budget_hit = True
                    break
                i1 = int(edge_rng.integers(0, m))
                i2 = int(edge_rng.integers(0, m))
                if i1 == i2:
                    continue
                cand = _refine_line_from_points(edge_pts[i1], edge_pts[i2])
                if cand is None:
                    continue
                dist = np.abs(edge_pts @ cand[:2] + cand[2])
                inl = dist < _REFINE_INLIER_DIST
                score = (float(np.exp(-(dist[inl] ** 2)
                                      / _REFINE_EXP_DENOM).sum())
                         if inl.any() else 0.0)
                if score > best_score:
                    best_score, best_line = score, cand
            if best_line is not None and best_score >= _REFINE_ACCEPT_RATIO * m:
                line = best_line
        lines.append(line)

    out = corners.copy()
    for i in range(4):
        l_prev, l_cur = lines[i - 1], lines[i]
        if l_prev is None or l_cur is None:
            continue
        det = l_prev[0] * l_cur[1] - l_cur[0] * l_prev[1]
        if abs(det) <= _REFINE_DET_EPS:
            continue
        x = (l_prev[1] * l_cur[2] - l_cur[1] * l_prev[2]) / det
        y = (l_cur[0] * l_prev[2] - l_prev[0] * l_cur[2]) / det
        new_pt = np.array([x, y])
        if np.hypot(new_pt[0] - corners[i][0],
                    new_pt[1] - corners[i][1]) <= _REFINE_MAX_CORNER_SHIFT:
            out[i] = new_pt
    if stats is not None:
        stats['budget_hit'] = budget_hit
        stats['time_s'] = time.monotonic() - t_start
    return out


class IDCardProcessor:
    def __init__(self, root):
        self.root = root
        try:
            tk._default_root = root
        except Exception:
            pass
        self.root.title("证件图片处理器·全能版（AI批量） by xm · 1110修正版")
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
        tk.Button(top_bar, text="看图选片", command=self.open_thumbnail_picker,
                  bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                  relief=tk.FLAT, font=("Arial", 11),
                  padx=14, pady=6, cursor='hand2').pack(side=tk.LEFT, padx=(8, 0))
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
            # Audit 1.1: drops on the A4 canvas go through the NEW
            # multi-file handler like everywhere else; the legacy
            # single-file on_drop must never be bound again.
            self.a4_canvas.dnd_bind('<<Drop>>', self.on_files_dropped)
            
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
            
            self._remember_dir(file_path)
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
        return {'image': None, 'points': [], 'processed': None, 'path': None, 'ocr': {}, '_ocr_for': None}

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
        for side in ('front', 'back'):
            slot = pair[side]
            proc = slot.get('processed')
            # Identity by OBJECT, not id(): a freed image's id() can be
            # recycled by a different image, which used to suppress its
            # OCR forever (audit 2.7).
            if proc is not None and slot.get('_ocr_for') is not proc:
                slot['_ocr_for'] = proc
                self._schedule_ocr(idx, side, proc)

    def _load_pair_to_workspace(self, idx):
        pair = self.pairs[idx]
        for side in ('front', 'back'):
            setattr(self, side + '_image', pair[side]['image'])
            setattr(self, side + '_points', list(pair[side]['points']))
            setattr(self, side + '_points_detected', False)
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
        # UI entry point: while a detection is in flight, mutating the
        # pair list would reroute its result onto another card (audit
        # 2.3). Internal callers (the batch chain itself) use
        # _append_pair, which skips this guard.
        if getattr(self, '_import_busy', False):
            self.update_status('正在识别上一张，请稍候…')
            return self.current_open
        return self._append_pair()

    def _append_pair(self):
        self._stash_workspace()
        self.pairs.append({'front': self._new_slot(), 'back': self._new_slot()})
        idx = len(self.pairs) - 1
        self._load_pair_to_workspace(idx)
        self.refresh_pairs_panel()
        return idx

    def delete_pair(self, idx):
        if getattr(self, '_import_busy', False):
            self.update_status('正在识别上一张，请稍候…')
            return
        if not (0 <= idx < len(self.pairs)):
            return
        # Audit 2.2: stash FIRST (the old code overwrote unsaved
        # workspace work), and only touch the workspace when the open
        # pair itself is affected: deleting a LATER pair must not yank
        # the workspace to another group; deleting an EARLIER pair only
        # renumbers the open one.
        self._stash_workspace()
        cur = self.current_open
        del self.pairs[idx]
        if not self.pairs:
            self.current_open = None
            for side in ('front', 'back'):
                setattr(self, side + '_image', None)
                setattr(self, side + '_points', [])
                setattr(self, side + '_points_detected', False)
                setattr(self, side + '_processed', None)
            try:
                self.draw_a4_layout()
            except Exception:
                pass
        elif cur is None:
            pass
        elif idx < cur:
            self.current_open = cur - 1
        elif idx == cur:
            self._load_pair_to_workspace(min(idx, len(self.pairs) - 1))
        self.refresh_pairs_panel()

    def clear_slot(self, idx, side):
        """只清一面（行尾「删组」是整组）：清掉该槽位的原图/角点/处理结果/读字，可单独重传这一面"""
        if getattr(self, '_import_busy', False):
            self.update_status('正在识别上一张，请稍候…')
            return
        if not (0 <= idx < len(self.pairs)):
            return
        self.pairs[idx][side] = self._new_slot()
        if getattr(self, 'current_open', None) == idx:
            setattr(self, side + '_image', None)
            setattr(self, side + '_points', [])
            setattr(self, side + '_points_detected', False)
            setattr(self, side + '_processed', None)
            try:
                self.draw_a4_layout()
            except Exception:
                pass
        self.update_status(f"第{idx + 1}组{'正面' if side == 'front' else '反面'}已删除，可以重新导入这一面")
        self.refresh_pairs_panel()

    def swap_pair_sides(self, idx):
        """正反面互换（青点名要的功能）：整组数据对调，含原图/角点/处理结果"""
        if getattr(self, '_import_busy', False):
            self.update_status('正在识别上一张，请稍候…')
            return
        if 0 <= idx < len(self.pairs):
            pair = self.pairs[idx]
            pair['front'], pair['back'] = pair['back'], pair['front']
            if self.current_open == idx:
                self._load_pair_to_workspace(idx)
            self.refresh_pairs_panel()
            self.update_status('已互换正反面')

    # ---- 2026-11-12: detection runs on ONE background thread and the
    # result comes back through a polled queue. The old flow ran
    # detection on the UI thread; a single inference can take seconds
    # on a slow PC, freezing the whole window -- Qing's "software is
    # laggy, buttons don't respond".
    def _detect_async(self, image, done):
        """Queue a background detection; done(corners) runs on the UI thread.
        The worker thread never touches Tk: it puts the result into a
        thread-safe queue that the UI thread polls. (Calling root.after
        from the worker thread can block forever -- verified the hard
        way on 2026-11-12: the batch sat at job 1/4 with the UI alive
        and the worker parked.)"""
        import queue
        if not hasattr(self, '_detect_jobs'):
            self._detect_jobs = []
            self._detect_busy = False
            self._detect_results = queue.Queue()
            self._poll_detect_results()
        self._detect_jobs.append((image, done))
        self._pump_detect_jobs()

    def _poll_detect_results(self):
        try:
            while True:
                _done, _corners = self._detect_results.get_nowait()
                self._detect_done(_done, _corners)
        except Exception:
            pass
        try:
            self.root.after(120, self._poll_detect_results)
        except Exception:
            pass

    def _pump_detect_jobs(self):
        if getattr(self, '_detect_busy', False):
            return
        jobs = getattr(self, '_detect_jobs', None)
        if not jobs:
            return
        image, done = jobs.pop(0)
        self._detect_busy = True

        def work():
            corners = None
            try:
                # Aligned detection only: the original has NO classical
                # contour fallback stage; failure surfaces as None and
                # routes to manual marking.
                corners = self.ai_detect_any(image)
            except Exception:
                corners = None
            try:
                self._detect_results.put((done, corners))
            except Exception:
                pass

        import threading
        threading.Thread(target=work, daemon=True).start()

    def _detect_done(self, done, corners):
        self._detect_busy = False
        try:
            done(corners)
        except Exception:
            pass
        self._pump_detect_jobs()

    def _auto_process_slot(self, idx, side, image, on_done):
        """放入即自动识别（对齐 IDCardMerge 交互）：识别在后台线程跑、
        界面不冻结；识别成功直接处理完，完成后在主线程调
        on_done(True)；没认出来 on_done(False)，交给标角窗口。"""
        setattr(self, side + '_image', image)
        setattr(self, side + '_points', [])
        setattr(self, side + '_points_detected', False)
        try:
            self.update_status('正在识别，请稍候…')
        except Exception:
            pass

        def _done(corners):
            # Audit 2.1/3.1: on_done MUST run exactly once even if
            # finish_marking raises outside its own guard, or the batch
            # chain stalls and _import_busy leaks (every later import
            # refused until restart). And 'ok' must reflect the actual
            # processing result -- finish_marking reports failure only
            # via a dialog and returns None either way, so check that a
            # processed image really landed.
            ok = False
            try:
                if corners:
                    setattr(self, side + '_points', corners)
                    setattr(self, side + '_points_detected', True)
                    self.finish_marking(side)
                    ok = getattr(self, side + '_processed', None) is not None
            except Exception:
                ok = False
            try:
                on_done(ok)
            except Exception:
                pass

        self._detect_async(image, _done)

    def _settings_file(self):
        try:
            if getattr(sys, 'frozen', False):
                base = os.path.dirname(sys.executable)
            else:
                base = os.path.dirname(os.path.abspath(__file__))
        except Exception:
            base = os.getcwd()
        return os.path.join(base, 'idcard_settings.json')

    def _get_last_dir(self):
        d = getattr(self, '_last_dir_cache', None)
        if d is None:
            d = ''
            try:
                import json as _json
                with open(self._settings_file(), 'r', encoding='utf-8') as f:
                    v = _json.load(f).get('last_dir', '')
                if v and os.path.isdir(v):
                    d = v
            except Exception:
                pass
            self._last_dir_cache = d
        return d

    def _remember_dir(self, path):
        try:
            if not path:
                return
            d = path if os.path.isdir(path) else os.path.dirname(path)
            if not d or not os.path.isdir(d):
                return
            self._last_dir_cache = d
            _data = self._read_settings()
            _data['last_dir'] = d
            self._write_settings(_data)
        except Exception:
            pass

    def _read_settings(self):
        try:
            import json as _json
            with open(self._settings_file(), 'r', encoding='utf-8') as f:
                v = _json.load(f)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}

    def _write_settings(self, data):
        try:
            import json as _json
            with open(self._settings_file(), 'w', encoding='utf-8') as f:
                _json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass

    def _save_settings(self):
        """Persist the UI settings (watermark / template / enhance)
        merged with last_dir. Audit 3.5: these used to live only in Tk
        variables recreated with defaults at every launch -- a safety
        watermark she had configured silently turned itself OFF on
        restart."""
        _data = self._read_settings()
        if getattr(self, '_last_dir_cache', None):
            _data['last_dir'] = self._last_dir_cache
        try:
            if getattr(self, 'template_var', None) is not None:
                _data['template'] = self.template_var.get()
            if getattr(self, 'enhance_var', None) is not None:
                _data['enhance'] = self.enhance_var.get()
            if getattr(self, 'wm_enabled_var', None) is not None:
                _data['wm'] = {
                    'enabled': bool(self.wm_enabled_var.get()),
                    'text': self.wm_text_var.get(),
                    'size': int(self.wm_size_var.get()),
                    'color': self.wm_color_var.get(),
                    'opacity': int(self.wm_opacity_var.get()),
                    'rows': int(self.wm_rows_var.get()),
                }
        except Exception:
            pass
        self._write_settings(_data)

    def _restore_settings(self):
        _data = self._read_settings()
        try:
            _tpl = _data.get('template')
            if _tpl in ('标准（每组一页）', '紧凑拼版'):
                self.template_var.set(_tpl)
            _inv = {'none': '原图', 'color': '彩色', 'bw': '黑白',
                    'docbw': '文档黑白', 'doccolor': '文档彩色'}
            _enh = _data.get('enhance')
            if _enh in _inv:
                self._enhance_text_var.set(_inv[_enh])
                self.enhance_var.set(_enh)
                self._on_enhance_change()
            _wm = _data.get('wm')
            if isinstance(_wm, dict) and _wm:
                self.wm_enabled_var.set(bool(_wm.get('enabled', False)))
                if isinstance(_wm.get('text'), str):
                    self.wm_text_var.set(_wm['text'])
                if _wm.get('size') in range(4, 15):
                    self.wm_size_var.set(_wm['size'])
                if _wm.get('color') in ('黑色', '蓝色', '红色', '灰色'):
                    self.wm_color_var.set(_wm['color'])
                if _wm.get('opacity') in (5, 10, 15, 18, 20, 25, 30, 40, 50, 60):
                    self.wm_opacity_var.set(_wm['opacity'])
                if _wm.get('rows') in range(1, 7):
                    self.wm_rows_var.set(_wm['rows'])
        except Exception:
            pass
        # From now on, any change to these variables is saved at once.
        for _name in ('wm_enabled_var', 'wm_text_var', 'wm_size_var',
                      'wm_color_var', 'wm_opacity_var', 'wm_rows_var',
                      'template_var', '_enhance_text_var'):
            _var = getattr(self, _name, None)
            if _var is not None:
                try:
                    _var.trace_add('write', lambda *a: self._save_settings())
                except Exception:
                    pass

    def import_into_slot(self, idx, side, file_path=None):
        if getattr(self, '_import_busy', False):
            self.update_status('正在识别上一张，请稍候…')
            return
        self.open_pair(idx)
        if not file_path:
            file_path = filedialog.askopenfilename(
                title='选择图片', initialdir=self._get_last_dir() or None,
                filetypes=[('图像文件', '*.jpg *.jpeg *.png *.bmp *.tiff'), ('所有文件', '*.*')])
        if not file_path:
            return
        self._remember_dir(file_path)
        try:
            image = Image.open(file_path)
            image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        except Exception as e:
            messagebox.showerror('错误', f'加载图片失败: {str(e)}')
            return
        self.pairs[idx][side]['path'] = file_path
        self._import_busy = True

        def _fin(ok):
            self._import_busy = False
            if ok:
                self.refresh_pairs_panel()
                self.update_status(f"第{idx + 1}组{'正面' if side == 'front' else '反面'}已自动识别处理完成")
            else:
                self._skip_auto_finish = True
                self.show_image_for_marking(side, image, file_path)
                self.update_status('自动识别没成功，请在窗口里手动标4个角点')

        self._auto_process_slot(idx, side, image, _fin)

    def open_thumbnail_picker(self):
        """看图选片：文件夹大缩略图阅览窗（PS 式）。点照片选中（可多选），
        一键按文件名顺序导入成组——不用对着文件名猜，也不用逐张拖。"""
        import os as _os
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        win = ToplevelWithIcon(self.root)
        win.title('看图选片 - 点照片选中，可多选')
        win.geometry('920x680')
        state = {'dir': self._get_last_dir() or _os.path.expanduser('~'),
                 'files': [], 'names': [], 'sel': set(), 'photos': [], 'frames': []}
        top = tk.Frame(win, bg='#F5F6F8')
        top.pack(side=tk.TOP, fill=tk.X)
        dir_lbl = tk.Label(top, text='', bg='#F5F6F8', fg='#333333', font=('Arial', 10))
        dir_lbl.pack(side=tk.LEFT, padx=8, pady=6)

        def choose_dir():
            d = filedialog.askdirectory(title='选择照片文件夹', initialdir=state['dir'])
            if d:
                state['dir'] = d
                load_dir()

        tk.Button(top, text='换文件夹', command=choose_dir,
                  bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                  relief=tk.FLAT, padx=12, pady=5, cursor='hand2').pack(side=tk.RIGHT, padx=8, pady=4)
        wrap = tk.Frame(win, bg='white')
        wrap.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(wrap, bg='white', highlightthickness=0)
        sb = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        inner = tk.Frame(canvas, bg='white')
        _win_id = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(_win_id, width=e.width))
        canvas.bind('<MouseWheel>', lambda e: canvas.yview_scroll(int(-e.delta / 120), 'units'))
        bar = tk.Frame(win, bg='#EEF0F3')
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        cnt = tk.Label(bar, text='', bg='#EEF0F3', font=('Arial', 10))
        cnt.pack(side=tk.LEFT, padx=8, pady=6)

        def refresh_cnt():
            cnt.config(text=f"已选 {len(state['sel'])} 张（按文件名顺序导入，两张一组）")

        def paint_sel():
            for i, fr in enumerate(state['frames']):
                fr.config(bg='#2F7CF6' if i in state['sel'] else '#DDDDDD')

        def toggle(i):
            if i in state['sel']:
                state['sel'].discard(i)
            else:
                state['sel'].add(i)
            paint_sel()
            refresh_cnt()

        def do_import():
            files = [state['files'][i] for i in sorted(state['sel'])]
            if not files:
                messagebox.showinfo('提示', '先点几张照片把它们选中', parent=win)
                return
            win.destroy()
            self.import_files_as_pairs(files)

        tk.Button(bar, text='导入选中', command=do_import,
                  bg='#2F7CF6', fg='white', activebackground='#2F7CF6',
                  activeforeground='white', relief=tk.FLAT,
                  font=('Arial', 10, 'bold'), padx=16, pady=5,
                  cursor='hand2').pack(side=tk.RIGHT, padx=8, pady=4)
        tk.Button(bar, text='清空选择',
                  command=lambda: (state['sel'].clear(), paint_sel(), refresh_cnt()),
                  bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                  relief=tk.FLAT, padx=12, pady=5, cursor='hand2').pack(side=tk.RIGHT, pady=4)

        def load_dir():
            for w_ in inner.winfo_children():
                w_.destroy()
            state['sel'].clear()
            state['photos'] = []
            state['frames'] = []
            dir_lbl.config(text=state['dir'])
            try:
                names = sorted(n for n in _os.listdir(state['dir'])
                               if n.lower().endswith(exts))
            except Exception:
                names = []
            state['names'] = names[:300]
            state['files'] = [_os.path.join(state['dir'], n) for n in state['names']]
            cols = 5
            for i, path in enumerate(state['files']):
                cell = tk.Frame(inner, bg='#DDDDDD')
                cell.grid(row=i // cols, column=i % cols, padx=4, pady=4)
                state['frames'].append(cell)
                holder = tk.Frame(cell, bg='white')
                holder.pack(padx=2, pady=2)
                try:
                    _im = Image.open(path)
                    _im.thumbnail((150, 105), Image.Resampling.LANCZOS)
                    _ph = ImageTk.PhotoImage(_im, master=self.root)
                    state['photos'].append(_ph)
                    lab = tk.Label(holder, image=_ph, bg='white')
                except Exception:
                    lab = tk.Label(holder, text='读不了', bg='white', width=20, height=6)
                lab.pack()
                _nm = state['names'][i]
                _short = _nm if len(_nm) <= 22 else _nm[:19] + '…'
                t = tk.Label(holder, text=_short, bg='white', fg='#555555', font=('Arial', 8))
                t.pack()
                for _w in (cell, holder, lab, t):
                    _w.bind('<Button-1>', lambda e, i=i: toggle(i))
            refresh_cnt()
            canvas.yview_moveto(0)

        load_dir()

    def batch_import(self):
        """批量导入：按选择顺序两张一组（先正面后反面），逐张自动识别处理"""
        files = filedialog.askopenfilenames(
            title='批量选择证件照片（按正反面顺序）', initialdir=self._get_last_dir() or None,
            filetypes=[('图像文件', '*.jpg *.jpeg *.png *.bmp *.tiff'), ('所有文件', '*.*')])
        if not files:
            return
        self.import_files_as_pairs(list(files))

    def on_files_dropped(self, event, idx=None, side=None):
        """Handle files dropped on a slot, the pairs panel, or anywhere on
        the main window. splitlist keeps EVERY dropped file (the old slot
        handler silently kept only the first one). A single file on a
        slot keeps the single-slot behavior; anything else fills pairs
        in order, two per pair, without popping marking windows."""
        try:
            paths = list(self.root.tk.splitlist(event.data))
        except Exception:
            paths = []
        exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        files = [p for p in paths if p.lower().endswith(exts)]
        if not files:
            return
        if idx is not None and len(files) == 1:
            self.import_into_slot(idx, side, files[0])
        elif idx is not None:
            self.import_files_as_pairs(files, start_idx=idx, start_side=side)
        else:
            self.import_files_as_pairs(files)

    def import_files_as_pairs(self, files, start_idx=None, start_side='front'):
        """Fill pairs with image files in order (front then back per
        pair), auto-processing each IN THE BACKGROUND so the window
        stays responsive while recognition runs; failures stay as
        待标角 and are counted in the status line. With start_idx,
        filling starts at that pair/side and continues into
        following pairs."""
        files = [f for f in files if f]
        if not files:
            return
        if getattr(self, '_import_busy', False):
            self.update_status('正在识别上一批，请稍候再导入')
            return
        self._remember_dir(files[0])
        if start_idx is None:
            # 启动时自带的那个空组先撤掉，免得导入完平白多出一个空组
            if len(self.pairs) == 1 and \
                    self.pairs[0]['front']['image'] is None and \
                    self.pairs[0]['back']['image'] is None:
                self.pairs = []
                self.current_open = None
            chunks = [files[i:i + 2] for i in range(0, len(files), 2)]
            pair_seq = [(None, ('front', 'back')) for _ in chunks]
        else:
            first_order = ('front', 'back') if start_side == 'front' else ('back', 'front')
            chunks = [files[:len(first_order)]]
            rest = files[len(first_order):]
            while rest:
                chunks.append(rest[:2])
                rest = rest[2:]
            pair_seq = [(start_idx, first_order)] + \
                [(start_idx + 1 + j, ('front', 'back')) for j in range(len(chunks) - 1)]
        steps = []
        for ci, ((pidx, order), chunk) in enumerate(zip(pair_seq, chunks)):
            for side, path in zip(order, chunk):
                steps.append((ci, pidx, side, path))
        state = {'i': 0, 'ok': 0, 'manual': 0, 'bad': [], 'idx_of': {}}
        import gc as _gc
        self._import_busy = True

        def finish_all():
            # Busy flag released FIRST: whatever the reporting below
            # does, the app must never stay wedged (audit 2.1).
            self._import_busy = False
            try:
                self.refresh_pairs_panel()
                _msg = f"批量导入完成：自动处理 {state['ok']} 张，待手动标角 {state['manual']} 张"
                if state['bad']:
                    _msg += f"，另有 {len(state['bad'])} 张读不了：{'、'.join(state['bad'][:3])}"
                self.update_status(_msg)
            except Exception:
                pass

        def step():
            if state['i'] >= len(steps):
                finish_all()
                return
            ci, pidx, side, path = steps[state['i']]
            self.update_status(f"正在识别（{state['i'] + 1}/{len(steps)}），请稍候…")
            if ci in state['idx_of']:
                idx = state['idx_of'][ci]
                self.open_pair(idx)
            elif pidx is None or pidx >= len(self.pairs):
                idx = self._append_pair()
                state['idx_of'][ci] = idx
            else:
                idx = pidx
                state['idx_of'][ci] = idx
                self.open_pair(idx)
            try:
                image = Image.open(path)
                image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            except Exception:
                state['bad'].append(path.replace('\\', '/').split('/')[-1])
                state['i'] += 1
                step()
                return
            self.pairs[idx][side]['path'] = path
            setattr(self, side + '_image', image)
            setattr(self, side + '_points', [])
            setattr(self, side + '_points_detected', False)

            def _done(corners):
                # Audit 2.1/3.1: the chain advances NO MATTER WHAT --
                # an exception here used to strand the whole batch on
                # "正在识别（i/N）" with _import_busy stuck on. A card
                # whose finish_marking failed counts as 待手动标角,
                # not as a success.
                ok = False
                try:
                    if corners:
                        setattr(self, side + '_points', corners)
                        setattr(self, side + '_points_detected', True)
                        self.finish_marking(side)
                        ok = getattr(self, side + '_processed', None) is not None
                except Exception:
                    ok = False
                if ok:
                    state['ok'] += 1
                else:
                    state['manual'] += 1
                try:
                    self._stash_workspace()
                except Exception:
                    pass
                _gc.collect()
                state['i'] += 1
                step()

            self._detect_async(image, _done)

        step()

    def on_slot_click(self, idx, side):
        slot = self.pairs[idx][side]
        if slot['image'] is None:
            self.import_into_slot(idx, side)
        elif slot['processed'] is not None:
            if getattr(self, '_import_busy', False):
                self.update_status('正在识别上一张，请稍候…')
                return
            self.open_pair(idx)
            self.reopen_for_adjustment(side)
        else:
            if getattr(self, '_import_busy', False):
                self.update_status('正在识别上一张，请稍候…')
                return
            self.open_pair(idx)
            self._import_busy = True

            def _fin(ok):
                self._import_busy = False
                if ok:
                    self.refresh_pairs_panel()
                    self.update_status(f"第{idx + 1}组{'正面' if side == 'front' else '反面'}已自动识别处理完成")
                else:
                    self._skip_auto_finish = True
                    self.show_image_for_marking(side, slot['image'], slot['path'])

            self._auto_process_slot(idx, side, slot['image'], _fin)

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
        if DRAG_DROP_SUPPORTED:
            # 2026-11-12: accept drops anywhere on the main window, not
            # only on the small slot labels (which showed a no-drop
            # cursor over most of the window).
            try:
                for _dw in (self.pairs_canvas, self.pairs_list_frame, self.root):
                    _dw.drop_target_register(DND_FILES)
                    _dw.dnd_bind('<<Drop>>', lambda e: self.on_files_dropped(e))
            except Exception:
                pass
        self.pairs.append({'front': self._new_slot(), 'back': self._new_slot()})
        self.current_open = 0
        self.refresh_pairs_panel()

    def _clear_pairs(self):
        if getattr(self, '_import_busy', False):
            self.update_status('正在识别上一张，请稍候…')
            return
        self.pairs = []
        self.current_open = None
        self._pair_photos = {}
        for side in ('front', 'back'):
            setattr(self, side + '_image', None)
            setattr(self, side + '_points', [])
            setattr(self, side + '_points_detected', False)
            setattr(self, side + '_processed', None)
        self._append_pair()
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
                    tk.Button(row, text='×', command=lambda i=idx, s=side: self.clear_slot(i, s),
                              bg='white', fg='#C0392B', activebackground='#FDECEA',
                              relief=tk.FLAT, padx=2, pady=0,
                              font=("Arial", 12, "bold"), cursor='hand2').pack(side=tk.LEFT)
                if DRAG_DROP_SUPPORTED:
                    try:
                        w.drop_target_register(DND_FILES)
                        w.dnd_bind('<<Drop>>', lambda e, i=idx, s=side: self.on_files_dropped(e, i, s))
                    except Exception:
                        pass
            tk.Button(row, text='正反互换', command=lambda i=idx: self.swap_pair_sides(i),
                      bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                      relief=tk.FLAT, padx=10, pady=4,
                      cursor='hand2').pack(side=tk.LEFT, padx=4)
            tk.Button(row, text='删组', command=lambda i=idx: self.delete_pair(i),
                      bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                      relief=tk.FLAT, padx=8, pady=4,
                      cursor='hand2').pack(side=tk.LEFT)
    def _schedule_ocr(self, idx, side, pil_img):
        # Audit 2.6/2.8: results are addressed by IMAGE IDENTITY, never
        # by (idx, side) position -- OCR takes seconds on her PC, and a
        # delete/swap/clear in between used to land one customer's name
        # and ID number on another customer's row. Delivery also goes
        # through a polled queue: the worker thread never writes pair
        # state and never calls root.after (the codebase's own note in
        # _detect_async records that after() from a worker can block
        # forever).
        if not (0 <= idx < len(self.pairs)):
            return
        pair_ref = self.pairs[idx]
        img_ref = pil_img
        if not hasattr(self, '_ocr_results'):
            import queue as _queue
            self._ocr_results = _queue.Queue()
            self._poll_ocr_results()

        def worker():
            err_text = ''
            try:
                texts = _ocr_lines(img_ref)
                info = _parse_id_front(texts) if side == 'front' else _parse_id_back(texts)
            except Exception as e:
                info = {}
                err_text = repr(e)
                _log_ocr_error(e)
            try:
                self._ocr_results.put((pair_ref, side, img_ref, info, err_text))
            except Exception:
                pass
        _threading.Thread(target=worker, daemon=True).start()

    def _poll_ocr_results(self):
        try:
            while True:
                pair_ref, side, img_ref, info, err_text = self._ocr_results.get_nowait()
                try:
                    idx = next((i for i, p in enumerate(self.pairs) if p is pair_ref), -1)
                    # Deliver ONLY if that slot still holds the very
                    # image that was sent to OCR; otherwise the result
                    # belongs to a card that was deleted/swapped/cleared
                    # meanwhile and dies here.
                    if pair_ref[side].get('processed') is img_ref:
                        if idx >= 0:
                            pair_ref[side]['ocr'] = info
                        self._on_ocr_done(idx, side, info, err_text)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.root.after(150, self._poll_ocr_results)
        except Exception:
            pass

    def _on_ocr_done(self, idx, side, info, err_text=''):
        if not info and err_text:
            self.update_status('读字没跑起来，原因记在程序旁边的 ocr_error.log')
            return
        if getattr(self, 'current_open', None) == idx and info:
            if side == 'front' and info.get('姓名'):
                self.update_status('已识别：%s %s' % (info.get('姓名', ''), info.get('公民身份号码', '')))
            elif side == 'back' and info.get('签发机关'):
                self.update_status('已识别签发机关：%s' % info.get('签发机关', ''))
        try:
            self.refresh_pairs_panel()
        except Exception:
            pass

    def _pair_info_rows(self):
        rows = []
        for i, pair in enumerate(self.pairs):
            merged = {}
            for side in ('front', 'back'):
                merged.update(pair[side].get('ocr') or {})
            if merged:
                merged['组号'] = i + 1
                rows.append(merged)
        return rows

    _INFO_COLS = ['组号', '姓名', '性别', '民族', '出生', '住址', '公民身份号码', '号码校验', '签发机关', '有效期限']

    def export_info_excel(self):
        rows = self._pair_info_rows()
        if not rows:
            messagebox.showinfo('提示', '还没有识别到身份证信息，请先处理身份证照片')
            return
        path = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel 工作簿', '*.xlsx')],
                                            initialfile='证件信息.xlsx')
        if not path:
            return
        try:
            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws.title = '证件信息'
            ws.append(self._INFO_COLS)
            for r in rows:
                ws.append([r.get(c, '') for c in self._INFO_COLS])
            wb.save(path)
            self.update_status('已导出 Excel：%s' % path)
        except Exception as e:
            messagebox.showerror('错误', '导出失败：%s' % e)

    def export_info_txt(self):
        rows = self._pair_info_rows()
        if not rows:
            messagebox.showinfo('提示', '还没有识别到身份证信息，请先处理身份证照片')
            return
        path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('文本文件', '*.txt')],
                                            initialfile='证件信息.txt')
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8-sig') as f:
                for r in rows:
                    f.write('【第%s组】\n' % r.get('组号', ''))
                    for c in self._INFO_COLS[1:]:
                        if r.get(c):
                            f.write('%s：%s\n' % (c, r[c]))
                    f.write('\n')
            self.update_status('已导出 TXT：%s' % path)
        except Exception as e:
            messagebox.showerror('错误', '导出失败：%s' % e)

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
        tk.Label(row2, text='颜色', bg='#EEF0F3').pack(side=tk.LEFT)
        self.wm_color_var = tk.StringVar(value='黑色')
        ttk.Combobox(row2, values=['黑色', '蓝色', '红色', '灰色'], textvariable=self.wm_color_var, width=4, state='readonly').pack(side=tk.LEFT)
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

        # 全能版专属：读字结果导出（橙色大按钮下方一行小按钮）
        export_row = tk.Frame(bar, bg='#EEF0F3')
        export_row.pack(fill=tk.X, pady=(6, 0))
        tk.Button(export_row, text='导出信息Excel', command=self.export_info_excel,
                  bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                  relief=tk.FLAT, padx=12, pady=4,
                  cursor='hand2').pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(export_row, text='导出信息TXT', command=self.export_info_txt,
                  bg='#F5F5F7', fg='#333333', activebackground='#E8E8EA',
                  relief=tk.FLAT, padx=12, pady=4,
                  cursor='hand2').pack(side=tk.LEFT)

        # 恢复上次的水印/排版/增强设置，并挂上自动保存（审计 3.5）
        self._restore_settings()

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

    def compute_layout(self):
        """计算排版位置清单：[[(卡片图, x, y), ...], ...]，坐标为 A4 页 300dpi 像素"""
        groups = self.collect_pair_images()
        if not groups:
            self.layout_pages = []
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
                placements = []
                for j, card in enumerate(cards[start:start + per]):
                    r, c = divmod(j, cols)
                    x = margin + c * (cw + gap_x)
                    y = margin + r * (ch + gap_y)
                    placements.append((card, x, y))
                pages.append(placements)
        else:
            spacing = int(20 * s)
            for g in groups:
                imgs = [prep(im) for im in g]
                total_h = len(imgs) * ch + (len(imgs) - 1) * spacing
                y = (ph - total_h) // 2
                x = (pw - cw) // 2
                placements = []
                for im in imgs:
                    placements.append((im, x, y))
                    y += ch + spacing
                pages.append(placements)
        self.layout_pages = pages
        return pages

    def render_pages(self, layout=None):
        """按位置清单渲染 A4 页图（含水印），写入 self.batch_pages"""
        if layout is None:
            layout = getattr(self, 'layout_pages', None)
        if not layout:
            return []
        s = self.mm_to_pixel
        pw, ph = int(self.a4_width * s), int(self.a4_height * s)
        pages = []
        for placements in layout:
            page = Image.new('RGB', (pw, ph), (255, 255, 255))
            for img, x, y in placements:
                page.paste(img, (int(x), int(y)))
            pages.append(page)
        if getattr(self, 'wm_enabled_var', None) is not None and self.wm_enabled_var.get():
            text = self.wm_text_var.get().strip()
            if text:
                font_px = int(self.wm_size_var.get() * s)
                _wm_colors = {'黑色': (0, 0, 0), '蓝色': (30, 90, 200), '红色': (220, 30, 30), '灰色': (110, 110, 110)}
                _wm_rgb = _wm_colors.get(self.wm_color_var.get() if getattr(self, 'wm_color_var', None) is not None else '黑色', (0, 0, 0))
                pages = [_apply_watermark(p, text, font_px, self.wm_opacity_var.get(), self.wm_rows_var.get(), _wm_rgb) for p in pages]
        self.batch_pages = pages
        return pages

    def compose_a4_pages(self):
        """把所有组的处理结果拼成 A4 页（300dpi），返回 PIL 页列表"""
        layout = self.compute_layout()
        if not layout:
            return []
        return self.render_pages(layout)

    def show_batch_preview(self):
        layout = self.compute_layout()
        if not layout:
            messagebox.showwarning('提示', '还没有已处理的证件，请先导入并处理至少一张')
            return
        self.render_pages(layout)
        s = self.mm_to_pixel
        pw, ph = int(self.a4_width * s), int(self.a4_height * s)
        cw, ch = int(self.id_width * s), int(self.id_height * s)
        # Open fit-to-page by default (Qing's live feedback 2026-11-12:
        # original size opened far too big). '原始大小' (1.0) stays on the
        # toolbar; state['scale'] is set from fit_scale() before the
        # first render() below.
        state = {'idx': 0, 'photos': [], 'drag': None, 'scale': 1.0}
        win = ToplevelWithIcon(self.root)
        win.title('可交互排版预览（支持拖放）')
        # Clamp to the screen and center it (Qing's small Acer is
        # 768-class tall: a fixed 900x960 window opens taller than
        # the screen and the bottom bar becomes unreachable).
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        ww, wh = min(900, sw - 40), min(960, sh - 80)
        win.geometry(f'{ww}x{wh}+{max(0, (sw - ww) // 2)}+{max(0, (sh - wh) // 2)}')

        # 顶部工具条（对齐 IDCardMerge 交互预览：按钮在上、颜色分工，个个是真功能）
        toolbar = tk.Frame(win, bg='white')
        toolbar.pack(side=tk.TOP, fill=tk.X)
        _btn_font = ('Arial', 10)
        def _tbtn(text, cmd, bg=None, fg=None):
            kw = dict(text=text, command=cmd, font=_btn_font, relief=tk.FLAT,
                      padx=12, pady=5, cursor='hand2')
            if bg:
                kw.update(bg=bg, fg=fg or 'white', activebackground=bg, activeforeground=fg or 'white')
            b = tk.Button(toolbar, **kw)
            b.pack(side=tk.LEFT, padx=3, pady=5)
            return b
        _tbtn('导出裁剪', self._export_crops)
        _tbtn('导出为 JPG', lambda: self._export_pages_jpg(ask_mode=True), bg='#2E7CF6')
        _tbtn('导出单页 PDF', self._export_pages_pdf_single, bg='#2E7CF6')
        _tbtn('导出合并 PDF', self._export_pages_pdf, bg='#43C93B')
        _tbtn('系统直接打印', self._print_pages)
        _tbtn('原始大小', lambda: set_scale(1.0))
        _tbtn('整页显示', lambda: set_scale(fit_scale()))

        header = tk.Label(win, text='', bg='#808080', fg='#F0F0F0', font=('Arial', 10))
        header.pack(side=tk.TOP, fill=tk.X)
        wrap = tk.Frame(win, bg='#808080')
        wrap.pack(fill=tk.BOTH, expand=True)
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        canvas = tk.Canvas(wrap, bg='#808080', highlightthickness=0)
        ysb = ttk.Scrollbar(wrap, orient=tk.VERTICAL, command=canvas.yview)
        xsb = ttk.Scrollbar(wrap, orient=tk.HORIZONTAL, command=canvas.xview)
        canvas.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        ysb.grid(row=0, column=1, sticky='ns')
        xsb.grid(row=1, column=0, sticky='ew')

        def metrics():
            sc = state['scale']
            dw, dh = int(pw * sc), int(ph * sc)
            # Center the page in the viewport (the original centers its
            # page; the old fixed 24px anchor left a gray void whenever
            # the window was larger than the scaled page -- Qing flagged
            # it live 2026-11-12). Falls back to 24 whenever the page
            # fills or exceeds the viewport, and before the canvas has
            # a real size. on_motion/on_release read the same metrics(),
            # so drag clamping/release math stay consistent.
            try:
                vw, vh = canvas.winfo_width(), canvas.winfo_height()
            except Exception:
                vw, vh = 0, 0
            ox = max(24, (vw - dw) // 2)
            oy = max(24, (vh - dh) // 2)
            return (dw, dh, max(10, int(cw * sc)), max(10, int(ch * sc)), ox, oy)

        def fit_scale():
            try:
                win.update_idletasks()
                vw = max(200, canvas.winfo_width() - 8)
                vh = max(200, canvas.winfo_height() - 8)
            except Exception:
                vw, vh = 860, 800
            return min(vw / pw, vh / ph)

        def set_scale(sc):
            state['scale'] = sc
            render()

        def render():
            dw, dh, dcw, dch, ox, oy = metrics()
            sc = state['scale']
            canvas.delete('all')
            state['photos'] = []
            try:
                vw, vh = canvas.winfo_width(), canvas.winfo_height()
            except Exception:
                vw, vh = 0, 0
            canvas.configure(scrollregion=(0, 0, max(vw, ox + dw + 24), max(vh, oy + dh + 24)))
            canvas.create_rectangle(ox, oy, ox + dw, oy + dh, fill='white', outline='black')
            for i, (img, x, y) in enumerate(layout[state['idx']]):
                dx, dy = ox + int(x * sc), oy + int(y * sc)
                tag = f'card{i}'
                canvas.create_rectangle(dx, dy, dx + dcw, dy + dch, outline='#999999', tags=(tag,))
                thumb = img.resize((dcw, dch), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(thumb, master=self.root)
                state['photos'].append(photo)
                canvas.create_image(dx, dy, image=photo, anchor=tk.NW, tags=(tag,))
            header.config(text=f"--- 第 {state['idx'] + 1} 页 A4 画布 ---")
            info.config(text=f"第 {state['idx'] + 1} / {len(layout)} 页 · 显示 {int(sc * 100)}% · 拖动卡片可调整位置，保存/打印按新位置")

        def on_press(event):
            items = canvas.find_withtag('current')
            if not items:
                return
            for t in canvas.gettags(items[0]):
                if t.startswith('card'):
                    state['drag'] = (int(t[4:]), canvas.canvasx(event.x), canvas.canvasy(event.y))
                    return

        def on_motion(event):
            if not state['drag']:
                return
            dw, dh, dcw, dch, ox, oy = metrics()
            i, lx, ly = state['drag']
            dx = canvas.canvasx(event.x) - lx
            dy = canvas.canvasy(event.y) - ly
            tag = f'card{i}'
            box = canvas.bbox(tag)
            if box:
                if box[0] + dx < ox:
                    dx = ox - box[0]
                if box[2] + dx > ox + dw:
                    dx = ox + dw - box[2]
                if box[1] + dy < oy:
                    dy = oy - box[1]
                if box[3] + dy > oy + dh:
                    dy = oy + dh - box[3]
            if dx or dy:
                canvas.move(tag, dx, dy)
                state['drag'] = (i, lx + dx, ly + dy)

        def on_release(event):
            if not state['drag']:
                return
            i = state['drag'][0]
            state['drag'] = None
            box = canvas.bbox(f'card{i}')
            if box:
                _dw, _dh, _dcw, _dch, ox, oy = metrics()
                img, _x, _y = layout[state['idx']][i]
                layout[state['idx']][i] = (img, (box[0] - ox) / state['scale'], (box[1] - oy) / state['scale'])

        canvas.bind('<ButtonPress-1>', on_press)
        canvas.bind('<B1-Motion>', on_motion)
        canvas.bind('<ButtonRelease-1>', on_release)
        canvas.bind('<MouseWheel>', lambda e: canvas.yview_scroll(int(-e.delta / 120), 'units'))
        canvas.bind('<Shift-MouseWheel>', lambda e: canvas.xview_scroll(int(-e.delta / 120), 'units'))
        # Re-center on resize/maximize. render() never resizes the
        # canvas itself (setting scrollregion is not a size change),
        # so this cannot loop; the guard covers layout storms anyway.
        _cfg_guard = {'busy': False}
        def _on_canvas_configure(event):
            if _cfg_guard['busy']:
                return
            _cfg_guard['busy'] = True
            try:
                render()
            finally:
                _cfg_guard['busy'] = False
        canvas.bind('<Configure>', _on_canvas_configure)
        bar = tk.Frame(win, bg='#F0F0F0')
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        info = tk.Label(bar, text='', bg='#F0F0F0', fg='#555555', font=('Arial', 9))
        info.pack(side=tk.LEFT, padx=8, pady=4)
        ttk.Button(bar, text='下一页', command=lambda: (state.update(idx=(state['idx'] + 1) % len(layout)), render())).pack(side=tk.RIGHT, padx=4, pady=3)
        ttk.Button(bar, text='上一页', command=lambda: (state.update(idx=(state['idx'] - 1) % len(layout)), render())).pack(side=tk.RIGHT, padx=4, pady=3)
        state['scale'] = fit_scale()
        render()

    def _export_crops(self):
        """导出裁剪：把各组已拉正的证件图逐张存到所选文件夹（真功能）"""
        items = []
        for gi, pair in enumerate(self.pairs):
            for side, sname in (('front', '正面'), ('back', '反面')):
                img = pair[side].get('processed') if isinstance(pair.get(side), dict) else None
                if img is not None:
                    items.append((gi + 1, sname, img))
        if not items:
            messagebox.showwarning('提示', '还没有已处理的证件可导出')
            return
        folder = filedialog.askdirectory(title='选择保存文件夹', initialdir=self._get_last_dir() or None)
        if not folder:
            return
        self._remember_dir(folder)
        saved, failed = 0, []
        for gi, sname, img in items:
            try:
                img.convert('RGB').save(os.path.join(folder, f'第{gi}组-{sname}.jpg'), quality=95)
                saved += 1
            except Exception as e:
                failed.append(f'第{gi}组-{sname}：{e}')
        if failed:
            messagebox.showerror('导出不完整',
                                 f'已导出 {saved} 张，以下 {len(failed)} 张保存失败：\n' + '\n'.join(failed[:5]))
        elif saved:
            messagebox.showinfo('完成', f'已导出 {saved} 张裁剪图到所选文件夹')

    def _choose_jpg_mode(self, parent):
        """JPG 导出模式弹窗（对齐 IDCardMerge 子菜单）：完整 A4 / 紧凑拼图，
        默认预选当前排版，返回 'full' / 'compact' / None"""
        result = {'mode': None}
        dlg = ToplevelWithIcon(parent)
        dlg.title('选择 JPG 导出模式')
        dlg.geometry('380x210')
        dlg.transient(parent)
        dlg.grab_set()
        tk.Label(dlg, text='请选择需要导出的画布类型：', font=('Arial', 10)).pack(anchor='w', padx=18, pady=(16, 8))
        var = tk.StringVar(value='compact' if self._template_key() == 'compact' else 'full')
        tk.Radiobutton(dlg, text='完整 A4 画布（带白边，适合直接打印）', variable=var, value='full', font=('Arial', 10)).pack(anchor='w', padx=26, pady=3)
        tk.Radiobutton(dlg, text='紧凑证件拼图（无多余白边，适合插入 Word/Excel）', variable=var, value='compact', font=('Arial', 10)).pack(anchor='w', padx=26, pady=3)
        brow = tk.Frame(dlg)
        brow.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=12)
        tk.Button(brow, text='下一步', bg='#2E7CF6', fg='white', activebackground='#2E7CF6',
                  activeforeground='white', relief=tk.FLAT, padx=16, pady=4,
                  command=lambda: (result.update(mode=var.get()), dlg.destroy())).pack(side=tk.RIGHT)
        tk.Button(brow, text='取消', relief=tk.FLAT, padx=16, pady=4,
                  command=dlg.destroy).pack(side=tk.RIGHT, padx=8)
        dlg.wait_window()
        return result['mode']

    def _export_pages_jpg(self, ask_mode=False):
        mode = None
        if ask_mode:
            mode = self._choose_jpg_mode(self.root)
            if mode is None:
                return
        pages = None
        if mode is not None and (mode == 'compact') != (self._template_key() == 'compact'):
            # 按所选模式临时重排导出（不改主界面设置、不动当前拖好的排版）
            saved_var = self.template_var.get()
            saved_layout = getattr(self, 'layout_pages', None)
            saved_pages = getattr(self, 'batch_pages', None)
            try:
                self.template_var.set('紧凑拼版' if mode == 'compact' else '标准（每组一页）')
                lay2 = self.compute_layout()
                pages = self.render_pages(lay2) if lay2 else []
            finally:
                self.template_var.set(saved_var)
                self.layout_pages = saved_layout
                self.batch_pages = saved_pages
        if pages is None:
            if getattr(self, 'layout_pages', None):
                self.render_pages()
            pages = getattr(self, 'batch_pages', None) or self.compose_a4_pages()
        if not pages:
            messagebox.showwarning('提示', '还没有可导出的拼版，请先处理证件')
            return
        folder = filedialog.askdirectory(title='选择保存文件夹', initialdir=self._get_last_dir() or None)
        if not folder:
            return
        self._remember_dir(folder)
        saved, failed = 0, []
        for i, p in enumerate(pages):
            try:
                p.save(os.path.join(folder, f'证件拼版_{i + 1:02d}.jpg'), quality=95, dpi=(self.dpi, self.dpi))
                saved += 1
            except Exception as e:
                failed.append(f'第{i + 1}页：{e}')
        if failed:
            messagebox.showerror('导出不完整',
                                 f'已保存 {saved} 页，以下 {len(failed)} 页保存失败：\n' + '\n'.join(failed[:5]))
        elif saved:
            messagebox.showinfo('完成', f'已保存 {saved} 页 JPG 到所选文件夹')

    def _pages_to_pdf_bytes(self, pages):
        # The temp dir holds full-size renders of CUSTOMER ID pages; it
        # must not be left behind in %TEMP% (audit 3.4). Bytes are fully
        # materialized by img2pdf.convert before we clean up.
        tmpdir = tempfile.mkdtemp()
        try:
            paths = []
            for i, p in enumerate(pages):
                fp = os.path.join(tmpdir, f'page{i + 1}.jpg')
                p.save(fp, quality=95, dpi=(self.dpi, self.dpi))
                paths.append(fp)
            return img2pdf.convert(*paths)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _export_pages_pdf(self):
        if getattr(self, 'layout_pages', None):
            self.render_pages()
        pages = getattr(self, 'batch_pages', None) or self.compose_a4_pages()
        if not pages:
            messagebox.showwarning('提示', '还没有可导出的拼版，请先处理证件')
            return
        folder = filedialog.askdirectory(title='选择保存文件夹', initialdir=self._get_last_dir() or None)
        if not folder:
            return
        self._remember_dir(folder)
        try:
            with open(os.path.join(folder, '证件拼版.pdf'), 'wb') as f:
                f.write(self._pages_to_pdf_bytes(pages))
        except Exception as e:
            messagebox.showerror('导出失败', f'PDF 没能保存：{e}')
            return
        messagebox.showinfo('完成', '已保存 证件拼版.pdf 到所选文件夹')

    def _export_pages_pdf_single(self):
        """导出单页 PDF：每一页 A4 单独存一个 PDF 文件到所选文件夹"""
        if getattr(self, 'layout_pages', None):
            self.render_pages()
        pages = getattr(self, 'batch_pages', None) or self.compose_a4_pages()
        if not pages:
            messagebox.showwarning('提示', '还没有可导出的拼版，请先处理证件')
            return
        folder = filedialog.askdirectory(title='选择保存文件夹', initialdir=self._get_last_dir() or None)
        if not folder:
            return
        self._remember_dir(folder)
        saved, failed = 0, []
        for i, p in enumerate(pages):
            try:
                with open(os.path.join(folder, f'证件拼版_{i + 1:02d}.pdf'), 'wb') as f:
                    f.write(self._pages_to_pdf_bytes([p]))
                saved += 1
            except Exception as e:
                failed.append(f'第{i + 1}页：{e}')
        if failed:
            messagebox.showerror('导出不完整',
                                 f'已保存 {saved} 个，以下 {len(failed)} 个保存失败：\n' + '\n'.join(failed[:5]))
        elif saved:
            messagebox.showinfo('完成', f'已保存 {saved} 个单页 PDF 到所选文件夹')

    def _print_images_windows(self, page_images, job_name):
        """Windows printing with an explicit printer choice.

        The old code used os.startfile(path, 'print'), which silently
        targets the system DEFAULT printer -- in Qing's shop the
        default is the tracing-paper printer used for seal plates, so
        ID copies came out on the wrong machine and there was no way
        to choose (2026-11-12). This flow always asks first; the
        system default is never used as a silent target and is never
        changed. Route A: the standard Windows print dialog
        (pywin32/MFC PrintDlg -- printer list, copies, properties)
        plus GDI printing of the composed 300dpi page images.
        Route B (only if A fails before anything is spooled): a small
        Tk picker fed by winspool EnumPrintersW via ctypes, same GDI
        path. Cancelling prints nothing; any failure shows an error
        dialog. There is deliberately NO fallback to the default
        printer.
        """
        try:
            import win32ui  # noqa: F401  (pywin32, bundled by the build)
        except Exception:
            messagebox.showerror(
                '打印不可用',
                '打印组件（pywin32）没有随程序打包，无法选择打印机。\n本次没有打印任何内容。')
            return False
        try:
            return self._print_via_printdlg(page_images, job_name)
        except Exception:
            pass  # nothing was spooled yet; fall through to the picker
        try:
            return self._print_via_picker(page_images, job_name)
        except Exception as e:
            messagebox.showerror(
                '打印失败',
                f'没能把打印任务发出去：{e}\n本次没有打印到任何打印机。')
            return False

    def _spool_images_gdi(self, dc, page_images, job_name, copies):
        """GDI-print PIL page images (300dpi A4) through a win32ui DC.

        GetDeviceCaps indices 110-113 are the fixed Win32 ABI values
        PHYSICALWIDTH / PHYSICALHEIGHT / PHYSICALOFFSETX /
        PHYSICALOFFSETY; the image is stretched onto the physical
        page (the composed pages already carry their own margins).
        """
        from PIL import ImageWin
        hdc = dc.GetHandleOutput()
        phys_w = dc.GetDeviceCaps(110)
        phys_h = dc.GetDeviceCaps(111)
        off_x = dc.GetDeviceCaps(112)
        off_y = dc.GetDeviceCaps(113)
        try:
            for _ in range(max(1, int(copies))):
                dc.StartDoc(job_name)
                try:
                    for img in page_images:
                        dc.StartPage()
                        dib = ImageWin.Dib(img.convert('RGB'))
                        dib.draw(hdc, (-off_x, -off_y,
                                       -off_x + phys_w, -off_y + phys_h))
                        dc.EndPage()
                finally:
                    dc.EndDoc()
        finally:
            dc.DeleteDC()

    def _print_via_printdlg(self, page_images, job_name):
        """Route A: the standard Windows printer dialog (PrintDlg)."""
        import win32con
        import win32ui
        pd = win32ui.CreatePrintDialog(
            win32con.PD_RETURNDC | win32con.PD_NOSELECTION |
            win32con.PD_NOPAGENUMS)
        if pd.DoModal() != win32con.IDOK:
            self.update_status('已取消打印')
            return False
        # Still pre-spool: any failure above the StartDoc line may
        # raise out to the picker fallback; once spooling starts, a
        # failure is reported here instead of re-asking (which could
        # double-print).
        try:
            dc = pd.GetPrinterDC()
        except Exception:
            dc = pd.GetDC()
        try:
            copies = int(pd.GetCopies())
        except Exception:
            copies = 1
        copies = min(max(copies, 1), 20)
        try:
            self._spool_images_gdi(dc, page_images, job_name, copies)
        except Exception as e:
            try:
                dc.DeleteDC()
            except Exception:
                pass
            messagebox.showerror('打印失败',
                                 f'打印过程中出错：{e}\n请检查所选打印机后重试。')
            return False
        messagebox.showinfo('打印', '打印任务已发送到打印机')
        return True

    def _enum_printers_ctypes(self):
        """Installed printer names via winspool EnumPrintersW (stdlib
        ctypes only, so enumeration works even without pywin32)."""
        import ctypes
        from ctypes import wintypes

        class PRINTER_INFO_2(ctypes.Structure):
            _fields_ = [('pServerName', ctypes.c_void_p),
                        ('pPrinterName', ctypes.c_void_p),
                        ('pShareName', ctypes.c_void_p),
                        ('pPortName', ctypes.c_void_p),
                        ('pDriverName', ctypes.c_void_p),
                        ('pComment', ctypes.c_void_p),
                        ('pLocation', ctypes.c_void_p),
                        ('pDevMode', ctypes.c_void_p),
                        ('pSepFile', ctypes.c_void_p),
                        ('pPrintProcessor', ctypes.c_void_p),
                        ('pDatatype', ctypes.c_void_p),
                        ('pParameters', ctypes.c_void_p),
                        ('pSecurityDescriptor', ctypes.c_void_p),
                        ('Attributes', wintypes.DWORD),
                        ('Priority', wintypes.DWORD),
                        ('DefaultPriority', wintypes.DWORD),
                        ('StartTime', wintypes.DWORD),
                        ('UntilTime', wintypes.DWORD),
                        ('Status', wintypes.DWORD),
                        ('cJobs', wintypes.DWORD),
                        ('AveragePPM', wintypes.DWORD)]

        winspool = ctypes.WinDLL('winspool.drv')
        flags = 0x2 | 0x4  # PRINTER_ENUM_LOCAL | PRINTER_ENUM_CONNECTIONS
        needed = wintypes.DWORD(0)
        returned = wintypes.DWORD(0)
        winspool.EnumPrintersW(flags, None, 2, None, 0,
                               ctypes.byref(needed), ctypes.byref(returned))
        if needed.value == 0:
            return []
        buf = ctypes.create_string_buffer(needed.value)
        if not winspool.EnumPrintersW(flags, None, 2, buf, needed.value,
                                       ctypes.byref(needed),
                                       ctypes.byref(returned)):
            return []
        names = []
        base = ctypes.addressof(buf)
        stride = ctypes.sizeof(PRINTER_INFO_2)
        for i in range(returned.value):
            info = PRINTER_INFO_2.from_address(base + i * stride)
            if info.pPrinterName:
                names.append(ctypes.wstring_at(info.pPrinterName))
        return names

    def _default_printer_ctypes(self):
        """Current default printer name (only used to PRESELECT the
        picker's row; the user still confirms before anything is
        sent, and the default is never changed)."""
        try:
            import ctypes
            from ctypes import wintypes
            winspool = ctypes.WinDLL('winspool.drv')
            size = wintypes.DWORD(0)
            winspool.GetDefaultPrinterW(None, ctypes.byref(size))
            if size.value == 0:
                return None
            buf = ctypes.create_unicode_buffer(size.value)
            if winspool.GetDefaultPrinterW(buf, ctypes.byref(size)):
                return buf.value
        except Exception:
            pass
        return None

    def _print_via_picker(self, page_images, job_name):
        """Route B: own printer picker + GDI printing via win32ui."""
        import win32ui
        names = self._enum_printers_ctypes()
        if not names:
            raise RuntimeError('没有枚举到任何已安装的打印机')
        result = {'printer': None, 'copies': 1}
        dlg = ToplevelWithIcon(self.root)
        dlg.title('选择打印机')
        dlg.geometry('420x360')
        dlg.transient(self.root)
        dlg.grab_set()
        tk.Label(dlg, text='请选择打印机（不会使用默认打印机直接打印）：',
                 font=('Arial', 10)).pack(anchor='w', padx=14, pady=(12, 6))
        lb = tk.Listbox(dlg, font=('Arial', 10), exportselection=False)
        lb.pack(fill=tk.BOTH, expand=True, padx=14)
        for n in names:
            lb.insert(tk.END, n)
        default_name = self._default_printer_ctypes()
        sel = names.index(default_name) if default_name in names else 0
        lb.selection_set(sel)
        lb.see(sel)
        row = tk.Frame(dlg)
        row.pack(fill=tk.X, padx=14, pady=8)
        tk.Label(row, text='份数：', font=('Arial', 10)).pack(side=tk.LEFT)
        copies_var = tk.IntVar(value=1)
        tk.Spinbox(row, from_=1, to=9, width=4, textvariable=copies_var,
                   font=('Arial', 10)).pack(side=tk.LEFT)

        def _ok():
            cur = lb.curselection()
            if cur:
                result['printer'] = names[cur[0]]
                try:
                    result['copies'] = min(max(int(copies_var.get()), 1), 9)
                except Exception:
                    result['copies'] = 1
            dlg.destroy()

        brow = tk.Frame(dlg)
        brow.pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=12)
        tk.Button(brow, text='打印', bg='#2E7CF6', fg='white',
                  activebackground='#2E7CF6', activeforeground='white',
                  relief=tk.FLAT, padx=16, pady=4,
                  command=_ok).pack(side=tk.RIGHT)
        tk.Button(brow, text='取消', relief=tk.FLAT, padx=16, pady=4,
                  command=dlg.destroy).pack(side=tk.RIGHT, padx=8)
        dlg.wait_window()
        if not result['printer']:
            self.update_status('已取消打印')
            return False
        dc = win32ui.CreateDC('WINSPOOL', result['printer'], None)
        self._spool_images_gdi(dc, page_images, job_name, result['copies'])
        messagebox.showinfo('打印', '打印任务已发送到打印机')
        return True

    def _print_pages(self):
        if getattr(self, 'layout_pages', None):
            self.render_pages()
        pages = getattr(self, 'batch_pages', None) or self.compose_a4_pages()
        if not pages:
            messagebox.showwarning('提示', '还没有可打印的拼版，请先处理证件')
            return
        if platform.system() == 'Windows':
            # Ask-first printer flow (never the silent default).
            self._print_images_windows(pages, '证件拼版')
            return
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(self._pages_to_pdf_bytes(pages))
                temp_file = tmp.name
        except Exception as e:
            messagebox.showerror('打印失败', f'生成打印文件失败：{e}')
            return
        try:
            subprocess.run(['lp', temp_file], check=True)
        except Exception as e:
            messagebox.showerror('打印失败',
                                 f'没能把打印任务发出去：{e}\n（拼版文件已存到 {temp_file}，可手动打开打印）')
            return
        messagebox.showinfo('打印', '打印任务已发送到打印机')

    def load_image(self, side):
        """加载图片"""
        file_path = filedialog.askopenfilename(
            title=f"选择{side}图片", initialdir=self._get_last_dir() or None,
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            image = Image.open(file_path)
            
            # 限制图片大小
            max_size = (1200, 1200)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            self._remember_dir(file_path)
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
    
    def _close_mark_window(self):
        """Close the marking window AND clear the references (audit
        2.5). Cancel / window-X used to call destroy() directly, leaving
        self.mark_window pointing at a dead window; the identity guard
        in try_auto_finish then PASSED for the corpse and a late
        detection result could finish-mark whatever image the workspace
        held by then."""
        win = getattr(self, 'mark_window', None)
        self.mark_window = None
        self.mark_canvas = None
        self.is_marking = False
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass

    def show_image_for_marking(self, side, image, file_path, existing_points=None):
        """显示图片供用户标记 - 带缩放功能"""
        if self.mark_window is not None:
            self._close_mark_window()
        # 2026-11-12 fix: a marking window closed mid-adjust used to leave
        # adjust_mode/adjust_points stale; the next window then drew the
        # OLD card's quad and never showed the user's own points. Reset
        # all marking/adjust state on every window open.
        self.adjust_mode = False
        self.adjust_points = []
        self.dragging_point_index = None
        self._drag_idx = None
            
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
        scale_slider = ttk.Scale(scale_frame, from_=0.1, to=6.0, 
                                variable=self.scale_var, length=150)
        scale_slider.pack(side=tk.LEFT, padx=5)
        scale_slider.bind("<Motion>", self.on_scale_change)
        scale_slider.bind("<ButtonRelease-1>", self.on_scale_change)
        
        # 显示当前缩放比例
        self.scale_label = ttk.Label(scale_frame, text="100%")
        self.scale_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(scale_frame, text="适窗", command=self._fit_mark_window).pack(side=tk.LEFT, padx=2)
        
        # 控制按钮区域
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        # Stable refs for the adjust-mode button surgery: it rebuilds
        # ONLY this button row and never touches the scale row
        # (audit 1.2/1.3).
        self._mark_control_frame = control_frame
        self._mark_btn_frame = btn_frame
        
        # 添加自动识别按钮
        ttk.Button(btn_frame, text="自动识别", 
                  command=lambda: self.auto_detect_corners(side)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="重新开始", 
                  command=lambda: self.reset_marking(side)).pack(side=tk.LEFT, padx=5)
        
        # 创建完成标记按钮（初始状态）
        self.finish_marking_btn = tk.Button(btn_frame, text="完成标记",
                                          command=lambda: self.finish_marking(side),
                                          bg="#2E7CF6", fg="white", activebackground="#5B96F8",
                                          activeforeground="white", relief="flat", padx=12, pady=3, state="normal")
        self.finish_marking_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="取消", 
                  command=self._close_mark_window).pack(side=tk.LEFT, padx=5)
        
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
        self.mark_window.protocol("WM_DELETE_WINDOW", self._close_mark_window)
        
        # 保存原始图片
        self.original_image = image.copy()
        self.current_scale = 1.0
        # 2026-11-10: open already fitted/zoomed to the window
        # (subtraction: no manual 'fit' step after opening)
        # 2026-11-12: fit the whole photo into the window immediately and
        # synchronously -- opening at 100% made close-up photos huge with
        # corners off-screen; keep a delayed second fit as a safety.
        try:
            self.mark_window.update_idletasks()
            self._fit_mark_window()
        except Exception:
            pass
        self.mark_window.after(150, self._fit_mark_window)
        
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
        模型缺失或检测失败返回 None（对齐原版：失败即失败，交给手动标角，
        没有传统检测自动兜底）。"""
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
            img_rgb = np.array(image.convert("RGB"))
            img_cv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            self._ai_ran = True
            corners = detector.detect(img_cv)
            if not corners:
                return None
            # Aligned with the original IDCardMerge v2.2: the decoded
            # keypoints are used exactly as they come out -- NO
            # out-of-frame rejection, NO 2% clamping, NO quadrilateral
            # validator gate, NO 1.0% inset (all four were our own
            # additions; the original performs none of them). On tight
            # shots a point may sit a few px outside the frame; the warp
            # consumes it as-is, exactly like the original.
            result = [[float(x), float(y)] for x, y in corners]
            if REFINE_CORNERS:
                # refineCorners（原版 Detect 的紧接步骤）：只在原图上
                # 微调已检出的四角，不做任何门槛判断；精修自身出错时
                # 记录一行日志并原样使用未精修角，绝不让识别因此失败。
                try:
                    refined = _refine_corners_np(
                        img_rgb, np.asarray(result, dtype=np.float64))
                    refined = np.asarray(refined, dtype=np.float64)
                    if refined.shape == (4, 2) and np.isfinite(refined).all():
                        result = [[float(x), float(y)] for x, y in refined]
                except Exception as _refine_err:
                    print(f"[core] refine_corners failed, "
                          f"using raw corners: {_refine_err!r}")
            return result
        except Exception:
            return None

    def ai_detect_any(self, image):
        """识别入口（对齐原版 IDCardMerge v2.2）：只做一次全图 AI 识别。
        原版没有任何第二机会——没有横带裁剪、没有旋转重试、没有中心
        裁剪、没有分块放大、没有传统轮廓兜底；我们自创的那套补救链已
        整体拆除（分块阶段曾把卡内横带误报成整卡，即"半张卡"事故）。
        认不出就如实返回 None，交给手动标角。"""
        return self.ai_detect_corners(image)

    def _seed_preset_box(self, side):
        """Auto-detect failed: pre-place a draggable quad (12% image inset)
        so the user drags four green handles onto the card corners instead
        of clicking four points from scratch."""
        setattr(self, side + '_points_detected', False)
        try:
            if side == "front":
                image = self.front_image
            elif side == "back":
                image = self.back_image
            else:
                image = self.card_image
            if image is None:
                return
            w, h = image.size
            mx, my = max(4, int(w * 0.12)), max(4, int(h * 0.12))
            pts = [(mx, my), (w - mx, my), (w - mx, h - my), (mx, h - my)]
            if side == "front":
                self.front_points = pts
            elif side == "back":
                self.back_points = pts
            else:
                self.card_points = pts
            self.display_image_on_canvas()
            self.update_status("自动识别未成功：已预置一个框，把四个绿点拖到卡片四角，点'完成标记'即可（也可点'自动识别'再试）")
        except Exception:
            pass

    def try_auto_finish(self, side):
        """标记窗口打开后静默尝试自动识别（后台线程跑，界面不冻结）：
        成功直接完成该面处理（不弹窗确认）；失败不打扰用户，
        停留在手动标记流程。"""
        try:
            if getattr(self, '_skip_auto_finish', False):
                self._skip_auto_finish = False
                self._seed_preset_box(side)
                return
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
            _win = self.mark_window

            def _done(corners):
                try:
                    if self.mark_window is not _win or self.current_marking_side != side:
                        return
                    _pts = getattr(self, side + '_points', None)
                    if _pts is not None and len(_pts) == 4:
                        return
                    if not corners:
                        self._seed_preset_box(side)
                        return
                    if side == "front":
                        self.front_points = corners
                    elif side == "back":
                        self.back_points = corners
                    else:
                        self.card_points = corners
                    setattr(self, side + '_points_detected', True)
                    self.finish_marking(side)
                except Exception:
                    pass

            self._detect_async(image, _done)
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
        
        # 修改按钮：只销毁按钮行本身。旧实现按层级遍历找 Button，
        # 按钮实际住在第四层、一个都没拆掉，新旧两排并存，点旧的
        # "完成标记"会用调整前的角点处理、把调整无声丢弃（审计 1.2）。
        if getattr(self, '_mark_btn_frame', None) is not None:
            try:
                self._mark_btn_frame.destroy()
            except Exception:
                pass
        # The old 完成标记 button died with its frame; clear the ref or
        # canvas redraws (display_image_on_canvas) will config() a dead
        # widget and raise TclError. finish/cancel_adjust recreate it.
        self.finish_marking_btn = None
        control_frame = self._mark_control_frame
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        self._mark_btn_frame = btn_frame
        
        # 创建完成调整按钮
        self.finish_adjust_btn = tk.Button(btn_frame, text="完成调整",
                                          command=lambda: self.finish_adjust(side),
                                          bg="#2E7CF6", fg="white", activebackground="#5B96F8",
                                          activeforeground="white", relief="flat", padx=12, pady=3)
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
        colors = ["#2FBF71", "#2FBF71", "#2FBF71", "#2FBF71"]  # unified green handles, same as IDCardMerge
        labels = ["左上", "右上", "右下", "左下"]
        
        for i, (px, py) in enumerate(self.adjust_points):
            # 缩放坐标
            scaled_x = px * self.current_scale
            scaled_y = py * self.current_scale
            
            # 绘制点（可拖动的圆）
            point_id = self.mark_canvas.create_oval(
                scaled_x-11, scaled_y-11, scaled_x+11, scaled_y+11,
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
            _ids = self.mark_canvas.find_withtag(f"point_{i}")
            if not _ids:
                continue
            point_id = _ids[0]
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
        
        # 更新按钮：只重建按钮行。旧实现销毁 control_frame 下所有
        # Frame，把缩放滑块、"适窗"钮、百分比标签所在的 scale_frame
        # 一并拆掉且不再重建--用过一次微调它们就永久消失（审计 1.3）。
        if getattr(self, '_mark_btn_frame', None) is not None:
            try:
                self._mark_btn_frame.destroy()
            except Exception:
                pass
        control_frame = self._mark_control_frame
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        self._mark_btn_frame = btn_frame
        
        ttk.Button(btn_frame, text="自动识别", 
                  command=lambda: self.auto_detect_corners(side)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="重新开始", 
                  command=lambda: self.reset_marking(side)).pack(side=tk.LEFT, padx=5)
        
        # 重新创建完成标记按钮
        self.finish_marking_btn = tk.Button(btn_frame, text="完成标记",
                                          command=lambda: self.finish_marking(side),
                                          bg="#2E7CF6", fg="white", activebackground="#5B96F8",
                                          activeforeground="white", relief="flat", padx=12, pady=3)
        self.finish_marking_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="取消", 
                  command=self._close_mark_window).pack(side=tk.LEFT, padx=5)
        
        # 重新绑定键盘回车事件
        self.mark_window.unbind("<Return>")
        self.mark_window.bind("<Return>", lambda e: self.finish_marking(side))
        
        self.update_status(f"{side}角点调整完成，可以继续处理或完成标记")
        
        # 点了'完成调整'即处理，不再二次弹窗确认（减法）
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
        
        # 更新按钮：只重建按钮行。旧实现销毁 control_frame 下所有
        # Frame，把缩放滑块、"适窗"钮、百分比标签所在的 scale_frame
        # 一并拆掉且不再重建--用过一次微调它们就永久消失（审计 1.3）。
        if getattr(self, '_mark_btn_frame', None) is not None:
            try:
                self._mark_btn_frame.destroy()
            except Exception:
                pass
        control_frame = self._mark_control_frame
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(side=tk.RIGHT)
        self._mark_btn_frame = btn_frame
        
        ttk.Button(btn_frame, text="自动识别", 
                  command=lambda: self.auto_detect_corners(side)).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="重新开始", 
                  command=lambda: self.reset_marking(side)).pack(side=tk.LEFT, padx=5)
        
        # 重新创建完成标记按钮
        self.finish_marking_btn = tk.Button(btn_frame, text="完成标记",
                                          command=lambda: self.finish_marking(side),
                                          bg="#2E7CF6", fg="white", activebackground="#5B96F8",
                                          activeforeground="white", relief="flat", padx=12, pady=3)
        self.finish_marking_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="取消", 
                  command=self._close_mark_window).pack(side=tk.LEFT, padx=5)
        
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

        # 计算长宽比：按对边配对，与角点的起止位置、照片旋转方向无关
        # （角点是沿四边形一周的顺序即可；照片转 90°/270° 时宽高互换，
        # 用 max/min 归一后再与期望比例比较，避免横拍竖拍误杀）
        width1 = np.linalg.norm(corners[1] - corners[0])
        width2 = np.linalg.norm(corners[2] - corners[3])
        height1 = np.linalg.norm(corners[3] - corners[0])
        height2 = np.linalg.norm(corners[2] - corners[1])

        pair_a = (width1 + width2) / 2
        pair_b = (height1 + height2) / 2

        if pair_a == 0 or pair_b == 0:
            return False

        aspect_ratio = max(pair_a, pair_b) / min(pair_a, pair_b)
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
    
    def _fit_mark_window(self):
        """适窗：把整图缩放至刚好填满标记画布，并回到左上角"""
        try:
            if not getattr(self, 'mark_canvas', None) or getattr(self, 'original_image', None) is None:
                return
            cw = self.mark_canvas.winfo_width()
            ch = self.mark_canvas.winfo_height()
            if cw < 50:
                cw, ch = 880, 620
            iw, ih = self.original_image.size
            fit = min(cw / iw, ch / ih)
            fit = max(0.1, min(6.0, fit))
            self.scale_var.set(fit)
            self.on_scale_change()
            self.mark_canvas.xview_moveto(0)
            self.mark_canvas.yview_moveto(0)
        except Exception:
            pass

    def on_mousewheel(self, event):
        """鼠标滚轮缩放：乘法步进，光标下的图像点保持不动（锚定缩放）"""
        try:
            if event.num == 4 or (hasattr(event, 'delta') and getattr(event, 'delta', 0) > 0):
                factor = 1.25
            elif event.num == 5 or (hasattr(event, 'delta') and getattr(event, 'delta', 0) < 0):
                factor = 1 / 1.25
            else:
                return
            old = self.current_scale
            new = max(0.1, min(6.0, old * factor))
            if abs(new - old) < 1e-6:
                return
            cv = self.mark_canvas
            ix = cv.canvasx(event.x) / old
            iy = cv.canvasy(event.y) / old
            self.scale_var.set(new)
            self.on_scale_change()
            new = self.current_scale
            sr = cv.cget('scrollregion').split()
            if len(sr) == 4:
                sw, sh = float(sr[2]), float(sr[3])
                if sw > 0 and sh > 0:
                    cv.xview_moveto(max(0.0, min(1.0, (ix * new - event.x) / sw)))
                    cv.yview_moveto(max(0.0, min(1.0, (iy * new - event.y) / sh)))
        except Exception:
            pass

    def on_scale_change(self, event=None):
        """缩放改变事件"""
        scale = self.scale_var.get()
        _oi = getattr(self, 'original_image', None)
        if _oi is not None:
            _cap = 3600 / max(1, max(_oi.size))
            if scale > _cap:
                scale = _cap
                self.scale_var.set(scale)
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
                # 2026-11-12: cache the LANCZOS-resized image per
                # (source image, scale); a drag used to re-resize the
                # full photo on every mouse-motion event, which is what
                # made marking feel stuck on slow machines.
                _ckey = round(float(self.current_scale), 4)
                _cache = getattr(self, "_scaled_cache", None)
                if _cache and _cache[0] is self.original_image and _cache[1] == _ckey:
                    scaled_image = _cache[2]
                else:
                    scaled_image = self.original_image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                    self._scaled_cache = (self.original_image, _ckey, scaled_image)
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
                    scaled_x-11, scaled_y-11, scaled_x+11, scaled_y+11,
                    fill="#2FBF71", outline="white", width=2
                )
                
                # 显示点编号
                self.mark_canvas.create_text(
                    scaled_x+18, scaled_y-18,
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
            setattr(self, side + '_points_detected', False)
            points.append((int(img_x), int(img_y)))
            if len(points) == 4:
                points[:] = _order_quad_points(points)
            
            # 重新显示图片以更新标记点
            self.display_image_on_canvas()
            
            self.update_status(f"已选择 {len(points)}/4 个点")
            
            # 选满4个点：不再弹窗确认（减法），完成按钮已亮起，点它或按回车处理
            if len(points) == 4:
                # 自动启用完成标记按钮
                if self.finish_marking_btn:
                    self.finish_marking_btn.config(state="normal")
                
                # 不弹窗打扰：状态行提示即可
                self.update_status("已选满4个点：角点可拖动微调，点'完成标记'或按回车开始处理")
    

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

        # 角点顺序：识别（AI）给的点已按证件本身排好（左上→右上→右下
        # →左下，跟着证件走、不跟照片方向走），必须原样送拉正——竖拍
        # 照片若再按图像坐标重排，拉正出来就是侧躺的（2026-11-12 青
        # 现场实测＋复现坐实；原版 IDCardMerge 也是 Detect 的点直送
        # 拉正、从不重排）。只有手工标角的点（点击顺序任意）才走
        # 下面的几何排序。_points_detected 由识别回调置 True、由手工
        # 点点/清槽/载入存档等路径置 False（getattr 默认 False）。
        if not getattr(self, side + '_points_detected', False):
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
            
            # 关闭标记窗口（并清引用，见 _close_mark_window）
            self._close_mark_window()
            
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
            messagebox.showerror("错误", "处理失败：角点位置可能交叉或重合，请重新调整角点后再试")
    
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
        # *_image_path is only ever assigned inside
        # show_image_for_marking; after a pure auto-process (no
        # marking window opened for this side yet) the attribute does
        # not exist, and reading it raised AttributeError -- which Tk
        # swallows in the frozen exe, so clicking a processed
        # thumbnail looked completely dead (Qing, 2026-11-12).
        # Fall back to the open pair slot's stored path.
        def _path(side_):
            p = getattr(self, side_ + '_image_path', None)
            if p:
                return p
            idx = getattr(self, 'current_open', None)
            if idx is not None and 0 <= idx < len(self.pairs):
                return self.pairs[idx][side_].get('path')
            return None
        # 根据side获取原始图片和点
        if side == "front":
            if not hasattr(self, 'front_image') or self.front_image is None:
                messagebox.showwarning("警告", "没有找到原始图片，请重新上传")
                return
            image = self.front_image
            points = self.front_points
            file_path = _path("front")
            side_name = "身份证人像面"
        elif side == "back":
            if not hasattr(self, 'back_image') or self.back_image is None:
                messagebox.showwarning("警告", "没有找到原始图片，请重新上传")
                return
            image = self.back_image
            points = self.back_points
            file_path = _path("back")
            side_name = "身份证国徽面"
        else:  # card
            if not hasattr(self, 'card_image') or self.card_image is None:
                messagebox.showwarning("警告", "没有找到原始图片，请重新上传")
                return
            image = self.card_image
            points = self.card_points
            file_path = _path("card")
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

        system = platform.system()
        if system == 'Windows':
            # Ask-first printer flow (never the silent default).
            if self._print_images_windows([self.a4_layout_image], 'A4排版'):
                self.update_status("打印任务已发送")
            return

        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_file = tmp.name
                # 保存为PNG格式
                self.a4_layout_image.save(temp_file, dpi=(self.dpi, self.dpi))

            # 根据操作系统调用打印命令
            if system == 'Darwin':
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
        
        # 关闭标记窗口（并清引用，见 _close_mark_window）
        self._close_mark_window()
        
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