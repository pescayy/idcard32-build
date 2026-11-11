# -*- coding: utf-8 -*-
"""全能版 OCR 并入（Qt 线）：RapidOCR 读字 + 身份证字段解析 + 信息导出。

口径逐函数复刻 IDCardProcessor_full.py（全能版）：
- 引擎懒加载单例 + 线程锁；打包（_MEIPASS）时先补 load_image.pyc 占位文件。
- _ocr_lines：白底 RGB 合成后送引擎，置信度 <0.5 丢弃，
  按 (round(中心y/20), 最小x) 排序成阅读顺序。
- 字段解析 _parse_id_front / _parse_id_back 与全能版一致，
  含 18 位号码正则、校验位（_id_checksum_ok）、出生日期回填。
- 导出列 INFO_COLS、Excel（openpyxl，表名「证件信息」）、
  TXT（utf-8-sig，「【第N组】」+「字段：值」，只写非空字段）均照全能版。
本模块不依赖 Qt；引擎不可用时 ocr_available() 返回 False，
由 UI 层降级并在状态行说明，绝不拖垮主程序。
"""
import os
import re as _re
import sys
import threading as _threading
import time
import traceback

from PIL import Image

_OCR_ENGINE = None
_OCR_LOCK = _threading.Lock()
_OCR_IMPORT_OK = None  # None=未探 / True / False（只探 import，不构造引擎）
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECKS = '10X98765432'

INFO_COLS = ['组号', '姓名', '性别', '民族', '出生', '住址', '公民身份号码',
             '号码校验', '签发机关', '有效期限']


def ocr_available():
    """只探 rapidocr_onnxruntime 能否 import（不构造引擎，构造留给首次读字）。"""
    global _OCR_IMPORT_OK
    if _OCR_IMPORT_OK is None:
        try:
            import rapidocr_onnxruntime  # noqa: F401
            _OCR_IMPORT_OK = True
        except Exception:
            _OCR_IMPORT_OK = False
    return _OCR_IMPORT_OK


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


def read_card(pil_img, side):
    """读一张已拉正卡面并解析字段；异常向上抛，由调用方记 log 并降级。"""
    texts = _ocr_lines(pil_img)
    return _parse_id_front(texts) if side == 'front' else _parse_id_back(texts)


def log_ocr_error(exc):
    """OCR 异常留痕：写到程序旁边的 ocr_error.log，避免窗口版无声失败。"""
    try:
        base = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base, 'ocr_error.log'), 'a', encoding='utf-8') as f:
            f.write(time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + repr(exc) + '\n')
            traceback.print_exc(file=f)
    except Exception:
        pass


def pair_info_rows(pairs):
    """把各组正反面 OCR 结果按组号合并成导出行（全能版 _pair_info_rows 口径）。"""
    rows = []
    for i, pair in enumerate(pairs):
        merged = {}
        for side in ('front', 'back'):
            merged.update(pair[side].get('ocr') or {})
        if merged:
            merged['组号'] = i + 1
            rows.append(merged)
    return rows


def export_info_excel(rows, path):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    ws.title = '证件信息'
    ws.append(INFO_COLS)
    for r in rows:
        ws.append([r.get(c, '') for c in INFO_COLS])
    wb.save(path)


def export_info_txt(rows, path):
    with open(path, 'w', encoding='utf-8-sig') as f:
        for r in rows:
            f.write('【第%s组】\n' % r.get('组号', ''))
            for c in INFO_COLS[1:]:
                if r.get(c):
                    f.write('%s：%s\n' % (c, r[c]))
            f.write('\n')
