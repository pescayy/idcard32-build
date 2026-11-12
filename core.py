# IDCardMuse core: UI-free processing pipeline extracted verbatim from
# IDCardProcessor_ort.py (1110 fix build). Logic identical to the Tk app;
# only tk variable reads were replaced with plain attributes.
import os, sys, glob, gc, io, json, math, base64, tempfile, time, shutil
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter, ImageOps
try:
    import img2pdf
except Exception:
    img2pdf = None

# refineCorners 开关（对齐原版 IDCardMerge v2.2：Detect 出角后紧接着在
# 原图上精修四角）。置 False 则原样使用模型角点。
REFINE_CORNERS = True

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

def _enhance_idcard_clarity(pil_img):
    """身份证清晰档（2026-11-11 议定，见 CLARITY-SPEC.md）：去雾＋
    锐化（对比改善由去雾的光照归一化兼任，理由见下），全是纯像素
    数学——绝不重画/生成笔画（微信 AI 把标签字增强改形就是生成式
    重画的实证，本档从根上不用那类方法）。两道工序都只动亮度、
    按通道同比例缩放（色相不变），且带
    两道保护：高饱和像素（红章、烫金）权重自动收力、不被加深或吹
    白；近白纸面有白色肩部保护。劲宁小勿大：光照场增益顶 1.15、
    锐化量 0.35 且带噪点守卫（平坦区的高频多为噪点/纹理，不锐
    化；真字边才吃锐化）。配方经午后十六验收三轮实测收敛：首版
    （增益 1.5／锐化 0.55）把噪声图噪点描硬；次版收劲后 OCR 仍
    丢浅蓝标签字——二分定位到分位对比拉伸是主犯（三组参数下都
    丢字，它把纸面噪点与浅色笔画一起拉扯），故本档不做全局分
    位拉伸，对比改善只来自光照场归一化本身；去雾（增益 1.15＋
    软限幅）＋守卫锐化的组合在真实虚图上标签字完整、OCR 字数
    不减、锐度小升，即现行配方。"""
    arr = np.asarray(pil_img.convert('RGB')).astype(np.float32)
    y = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    h, w = y.shape

    def _sat_weight(rgb):
        mx = rgb.max(axis=2)
        mn = rgb.min(axis=2)
        s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
        t = np.clip((s - 0.20) / 0.25, 0.0, 1.0)
        return 1.0 - t * t * (3.0 - 2.0 * t)

    # 1) 去雾：亮度除以低频光照场（÷16 降采样求场，粗到看不见笔画，
    #    同 lic_enhance 的 48px 阶段思路）；同 lic_enhance 的墨守卫：
    #    远低于本地场的像素（笔画本体）少吃增益，不被提成灰。
    small = cv2.resize(y, (max(1, w // 16), max(1, h // 16)),
                       interpolation=cv2.INTER_AREA)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(bg, (0, 0), 8)
    bgv = np.maximum(bg, 24.0)
    gain = np.clip(232.0 / bgv, 0.97, 1.15)
    r = y / bgv
    gf = np.where(r >= 0.85, 1.0,
                  np.where(r <= 0.55, 0.25,
                           0.25 + 0.75 * (r - 0.55) / 0.30))
    # 高饱和像素（红章、皮肤、烫金）同样收力：增益不整幅硬乘。
    gain = 1.0 + (gain - 1.0) * gf * _sat_weight(arr)
    gained = arr * gain[..., None]
    # 软限幅：>235 的部分只走 35% 斜率，避免单通道硬削顶把浅色
    # 笔画冲向纸白（午后十六二分实测：硬削顶参与丢浅蓝标签字）。
    out = np.where(gained > 235.0,
                   235.0 + (gained - 235.0) * 0.35, gained)
    out = np.clip(out, 0, 255)

    # 2) 锐化：只在亮度上做 unsharp（不生色边），高饱和区收力，
    #    防红章边缘出光晕；噪点守卫：高频幅度小的像素多为噪点/
    #    纸纹，权重趋零，只有真边缘（字边）吃满锐化。
    y3 = out[..., 0] * 0.299 + out[..., 1] * 0.587 + out[..., 2] * 0.114
    blur = cv2.GaussianBlur(y3, (0, 0), 1.1)
    sharp = y3 + 0.35 * (y3 - blur)
    ratio = np.where(y3 >= 1.0, sharp / np.maximum(y3, 1.0), 1.0)
    edge = np.abs(y3 - blur)
    e_wgt = np.clip((edge - 1.5) / 8.0, 0.0, 1.0)
    wgt = (_sat_weight(out) * e_wgt)[..., None]
    corrected = np.clip(out * ratio[..., None], 0, 255)
    out = out + wgt * (corrected - out)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'RGB')

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

class Core:
    """Plain-state holder + portable processing methods from IDCardProcessor."""

    def __init__(self):
        self.id_width = 85.6
        self.id_height = 54.0
        self.card_width = 85.6
        self.card_height = 54.0
        self.enhance_mode = 'color'
        self.template = '标准（每组一页）'
        self.wm_enabled = False
        self.wm_text = ''
        self.wm_size = 7
        self.wm_color = '黑色'
        self.wm_opacity = 18
        self.wm_rows = 3
        self.pairs = []
        self.detector_ai = None
        self.ai_available = False
        self.current_marking_side = 'front'
        self.dpi = 300
        self.mm_to_pixel = self.dpi / 25.4
        self.a4_width = 210
        self.a4_height = 297
        self.corner_radius_mm = 3.18
        self.corner_radius = int(self.corner_radius_mm * self.mm_to_pixel)

    def _stash_workspace(self):
        # In the Tk app this syncs the open workspace pair into self.pairs;
        # Core keeps pairs as the single source of truth, so it is a no-op.
        return None

    def _template_key(self):
        v = self.template
        if '单列' in v or '清单' in v:
            return 'single'
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
        key = self._template_key()
        if key == 'compact':
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
        elif key == 'single':
            # 单列居中（清单）：全部卡片一列居中、自上而下，页边距/行距
            # 与紧凑模式同档；原版 v2.2 三排版模式之一（对照表 §5）。
            cards = [prep(im) for g in groups for im in g]
            margin = int(10 * s)
            gap_y = int(4 * s)
            rows = max(1, int((ph - 2 * margin + gap_y) // (ch + gap_y)))
            x = (pw - cw) // 2
            for start in range(0, len(cards), rows):
                placements = []
                for j, card in enumerate(cards[start:start + rows]):
                    placements.append((card, x, margin + j * (ch + gap_y)))
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
        if self.wm_enabled:
            text = self.wm_text.strip()
            if text:
                font_px = int(self.wm_size * s)
                _wm_colors = {'黑色': (0, 0, 0), '蓝色': (30, 90, 200), '红色': (220, 30, 30), '灰色': (110, 110, 110)}
                _wm_rgb = _wm_colors.get(self.wm_color, (0, 0, 0))
                pages = [_apply_watermark(p, text, font_px, self.wm_opacity, self.wm_rows, _wm_rgb) for p in pages]
        self.batch_pages = pages
        return pages

    def compose_a4_pages(self):
        """把所有组的处理结果拼成 A4 页（300dpi），返回 PIL 页列表"""
        layout = self.compute_layout()
        if not layout:
            return []
        return self.render_pages(layout)

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
            if mode == 'clarity':
                return _enhance_idcard_clarity(img)
        except Exception:
            return img
        return img

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


    def process_card(self, image, points, points_ordered=False):
        """UI-free equivalent of finish_marking's pipeline:
        order -> area guard -> warp -> enhance -> rounded corners.

        points_ordered=True: the points already stand in warp order
        [card TL, TR, BR, BL] -- AI detection output. The model's
        keypoints are anchored to the card itself, not to the photo
        (verified 2026-11-12: detect() on a photo rotated 90 deg CW/CCW
        returns the rotated original corners elementwise, max diff
        ~5 px, vs >1200 px for any cyclic shift), so they must NOT be
        re-sorted geometrically: on a portrait photo the geometric sort
        maps the image's top edge onto the output's top edge and the
        card comes out sideways. The original IDCardMerge never
        re-sorts Detect output either -- the points go straight into
        the warp. Free-clicked manual points (arbitrary order) keep
        the geometric sort (the default)."""
        if image is None or not points or len(points) != 4:
            return None
        if points_ordered:
            points = [tuple(p) for p in points]
        else:
            points = _order_quad_points([tuple(p) for p in points])
        pts = [(float(p[0]), float(p[1])) for p in points]
        area = 0.0
        for i in range(4):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % 4]
            area += x1 * y2 - x2 * y1
        if abs(area) / 2.0 < 200:
            return None
        processed = self.perspective_transform(image, points)
        if processed is None:
            return None
        processed = self.apply_enhancement(processed)
        return self.add_rounded_corners(processed)

    def auto_process(self, image):
        """Detect corners (AI only -- aligned with the original: no
        classical fallback stage) and process; None when not detected."""
        corners = self.ai_detect_any(image)
        if not corners:
            return None
        return self.process_card(image, corners, points_ordered=True)
