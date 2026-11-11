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

def _inset_quad(pts, frac=0.003):
    """Inset each edge of the quad along its inward normal by frac*card width,
    recomputing corners as intersections. Counters the model's outward corner
    bias (measured: 47% of edges include >2px background). On any degeneracy,
    returns the input unchanged."""
    try:
        p = [np.array(q, dtype=np.float64) for q in pts]
        w1 = np.hypot(*(p[1] - p[0])); w2 = np.hypot(*(p[2] - p[3]))
        d = frac * max(w1, w2)
        if d <= 0:
            return pts
        cen = sum(p) / 4.0
        lines = []
        for i in range(4):
            a, b = p[i], p[(i + 1) % 4]
            e = b - a
            L = np.hypot(*e)
            if L < 1:
                return pts
            n = np.array([-e[1], e[0]]) / L
            if np.dot(n, cen - a) < 0:
                n = -n
            lines.append((a + n * d, e))
        out = []
        for i in range(4):
            p1, e1 = lines[i - 1]
            p2, e2 = lines[i]
            det_ = e1[0] * e2[1] - e1[1] * e2[0]
            if abs(det_) < 1e-9:
                return pts
            t = ((p2[0] - p1[0]) * e2[1] - (p2[1] - p1[1]) * e2[0]) / det_
            out.append(p1 + e1 * t)
        return [(float(q[0]), float(q[1])) for q in out]
    except Exception:
        return pts

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
            self._sess = _ort.InferenceSession(onnxmodel, providers=['CPUExecutionProvider'])
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
        # 2026-11-10 fix: model keypoints carry card orientation (trained order =
        # card TL->TR->BR->BL); geometric re-sort flips upside-down shots.
        # Use raw order; fall back to geometric sort only on self-crossing quads.
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
        tmpdir = tempfile.mkdtemp()
        paths = []
        for i, p in enumerate(pages):
            fp = os.path.join(tmpdir, f'page{i + 1}.jpg')
            p.save(fp, quality=95, dpi=(self.dpi, self.dpi))
            paths.append(fp)
        return img2pdf.convert(*paths)

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
            # 2026-11-10: inset edges by 1.0% of card width before warping.
            # Calibrated on synthetic ground truth: model corners overshoot
            # ~0.9% of card width; 1.0% drives out-of-card pixels to ~0 while
            # eating only ~10 output px of the card's blank margin per edge.
            arr = np.array(_inset_quad([[float(x), float(y)] for x, y in arr], 0.010),
                           dtype=np.float32)
            return [[float(x), float(y)] for x, y in arr]
        except Exception:
            return None

    def _ai_band_crop(self, image):
        """找截图里照片所在的横带（上下大片黑边/纯色时）：返回 (裁剪图, y偏移)，不适用返回 (None, 0)"""
        try:
            g = np.asarray(image.convert("L"), dtype=np.float32)
            rs = g.std(axis=1)
            mask = rs > 10
            best, cur = (0, 0), None
            for i, v in enumerate(mask):
                if v and cur is None:
                    cur = i
                if (not v or i == len(mask) - 1) and cur is not None:
                    end = i if not v else i + 1
                    if end - cur > best[1] - best[0]:
                        best = (cur, end)
                    cur = None
            h = image.height
            if not (h * 0.12 < best[1] - best[0] < h * 0.85):
                return None, 0
            y0 = max(0, best[0] - 6)
            y1 = min(h, best[1] + 6)
            return image.crop((0, y0, image.width, y1)), y0
        except Exception:
            return None, 0

    def ai_detect_any(self, image):
        """深度识别：全图没认出时的补救链——横带裁剪→旋转90/270→中心裁剪→分块放大。
        每步仍走 ai_detect_corners（含四边形合理性校验）；全失败返回 None，交给手动标角。
        AI 模型缺失时第一步后直接返回，保持传统检测兜底的原有门闩不变。"""
        corners = self.ai_detect_corners(image)
        if corners or not getattr(self, "_ai_ran", False):
            return corners
        W, H = image.size
        import time as _time
        _t0 = _time.time()
        try:
            band, y0 = self._ai_band_crop(image)
            if band is not None:
                c = self.ai_detect_corners(band)
                if c:
                    return [[x, y + y0] for x, y in c]
        except Exception:
            pass
        if _time.time() - _t0 > 20:
            return None
        for ang in (90, 270):
            try:
                rot = image.rotate(ang, expand=True)
                c = self.ai_detect_corners(rot)
                if c:
                    if ang == 90:
                        return [[W - 1 - qy, qx] for qx, qy in c]
                    return [[qy, H - 1 - qx] for qx, qy in c]
            except Exception:
                pass
        if _time.time() - _t0 > 20:
            return None
        try:
            cw, ch = int(W * 0.79), int(H * 0.79)
            x0, y0 = (W - cw) // 2, (H - ch) // 2
            c = self.ai_detect_corners(image.crop((x0, y0, x0 + cw, y0 + ch)))
            if c:
                return [[x + x0, y + y0] for x, y in c]
        except Exception:
            pass
        if _time.time() - _t0 > 20:
            return None
        try:
            tw, th = int(W * 0.62), int(H * 0.62)
            fallback = None
            for (tx, ty) in [(0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th), ((W - tw) // 2, (H - th) // 2)]:
                if _time.time() - _t0 > 25:
                    break
                c = self.ai_detect_corners(image.crop((tx, ty, tx + tw, ty + th)))
                if not c:
                    continue
                pts = [[x + tx, y + ty] for x, y in c]
                touches = any(x < tw * 0.015 or x > tw * 0.985 or y < th * 0.015 or y > th * 0.985 for x, y in c)
                if not touches:
                    return pts
                if fallback is None:
                    fallback = pts
            if fallback:
                return fallback
        except Exception:
            pass
        return None

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


    def process_card(self, image, points):
        """UI-free equivalent of finish_marking's pipeline:
        order -> area guard -> warp -> enhance -> rounded corners."""
        if image is None or not points or len(points) != 4:
            return None
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
        """Detect corners (AI first, traditional fallback) and process."""
        corners = self.ai_detect_any(image)
        if not corners:
            return None
        return self.process_card(image, corners)
