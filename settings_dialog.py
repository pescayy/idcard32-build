# -*- coding: utf-8 -*-
"""IDCardMuse 设置窗（对照表 §7：IDCardMerge v2.2 有独立设置窗）。

三组与原版同名：画布基础设置 / 常见尺寸快捷设置 / 物理尺寸微调。
数值存在主窗属性上（canvas_w/h、card_w/h、print_scale），点「确定」
才写回并持久化进 idcard_settings.json；默认值与现行行为完全一致
（画布 210×297、卡片 85.6×54、打印缩放 100% = 打印路径零变换）。
"""
import os, sys
from PySide2.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                               QLabel, QComboBox, QDoubleSpinBox, QPushButton)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app_qt import PairRow

# 常见尺寸预设（mm）：身份证为现行规格；户口本单页尺寸见对照表 §8 的
# v2.2 本体字符串「户口本单页 (143 x 105)」；执照两规格为 ying 定案规格。
PRESETS = [
    ('身份证（85.6 × 54）', 85.6, 54.0),
    ('户口本单页（143 × 105）', 143.0, 105.0),
    ('营业执照副本（297 × 210）', 297.0, 210.0),
    ('营业执照正本（420 × 297）', 420.0, 297.0),
    ('自定义', None, None),
]


class SettingsDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.setWindowTitle('设置')
        self.resize(430, 430)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        # ---- 画布基础设置 ----
        g1 = QGroupBox('画布基础设置')
        f1 = QVBoxLayout(g1)
        r = QHBoxLayout()
        r.addWidget(QLabel('纸张宽度（毫米）'))
        self.canvas_w = self._spin(50, 600, win.canvas_w)
        r.addWidget(self.canvas_w)
        r.addWidget(QLabel('纸张高度（毫米）'))
        self.canvas_h = self._spin(50, 600, win.canvas_h)
        r.addWidget(self.canvas_h)
        f1.addLayout(r)
        lay.addWidget(g1)

        # ---- 常见尺寸快捷设置 ----
        g2 = QGroupBox('常见尺寸快捷设置')
        f2 = QVBoxLayout(g2)
        self.preset = QComboBox()
        self.preset.addItems([p[0] for p in PRESETS])
        self.preset.currentIndexChanged.connect(self._apply_preset)
        f2.addWidget(self.preset)
        r = QHBoxLayout()
        r.addWidget(QLabel('宽度（毫米）'))
        self.card_w = self._spin(20, 500, win.card_w)
        r.addWidget(self.card_w)
        r.addWidget(QLabel('高度（毫米）'))
        self.card_h = self._spin(20, 500, win.card_h)
        r.addWidget(self.card_h)
        f2.addLayout(r)
        lay.addWidget(g2)
        # 初值与某个预设吻合则选中该预设，否则停在「自定义」
        self.preset.blockSignals(True)
        idx = len(PRESETS) - 1
        for i, (_n, w, h) in enumerate(PRESETS):
            if w is not None and abs(w - win.card_w) < 0.01 \
                    and abs(h - win.card_h) < 0.01:
                idx = i
                break
        self.preset.setCurrentIndex(idx)
        self.preset.blockSignals(False)

        # ---- 物理尺寸微调 ----
        g3 = QGroupBox('物理尺寸微调')
        f3 = QVBoxLayout(g3)
        r = QHBoxLayout()
        r.addWidget(QLabel('打印缩放'))
        self.print_scale = QDoubleSpinBox()
        self.print_scale.setRange(90.0, 110.0)
        self.print_scale.setSingleStep(0.5)
        self.print_scale.setDecimals(1)
        self.print_scale.setSuffix(' %')
        self.print_scale.setValue(float(win.print_scale))
        r.addWidget(self.print_scale)
        r.addStretch(1)
        f3.addLayout(r)
        hint = QLabel('打印出来的证件偏大就往小调、偏小就往大调；100% 为不调整。'
                      '只影响打印，不影响导出文件。')
        hint.setWordWrap(True)
        hint.setStyleSheet('color: #5B6470; font-size: 12.5px;')
        f3.addWidget(hint)
        lay.addWidget(g3)

        lay.addStretch(1)
        btns = QHBoxLayout()
        btns.addStretch(1)
        ok_b = QPushButton('确定')
        ok_b.setStyleSheet(win._primary_style())
        ok_b.clicked.connect(self._accept)
        cancel_b = QPushButton('取消')
        cancel_b.setStyleSheet(PairRow._btn_style())
        cancel_b.clicked.connect(self.reject)
        btns.addWidget(ok_b)
        btns.addWidget(cancel_b)
        lay.addLayout(btns)

    @staticmethod
    def _spin(lo, hi, value):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(1)
        s.setValue(float(value))
        return s

    def _apply_preset(self, i):
        _n, w, h = PRESETS[i]
        if w is not None:
            self.card_w.setValue(w)
            self.card_h.setValue(h)

    def _accept(self):
        win = self.win
        win.canvas_w = float(self.canvas_w.value())
        win.canvas_h = float(self.canvas_h.value())
        win.card_w = float(self.card_w.value())
        win.card_h = float(self.card_h.value())
        win.print_scale = float(self.print_scale.value())
        win._sync_core_settings()
        win._save_settings()
        win.set_status('设置已保存')
        self.accept()
