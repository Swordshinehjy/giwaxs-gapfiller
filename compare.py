from pathlib import Path

import fabio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PySide6.QtCore import QCoreApplication, Qt
from PySide6.QtGui import QAction, QActionGroup, QIcon, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                               QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QSizePolicy, QSlider, QSpinBox, QToolBar,
                               QToolButton, QVBoxLayout, QWidget)

from custum import CompareCanvas


class CompareDialog(QDialog):

    def __init__(self,
                 parent=None,
                 data0=None,
                 data1=None,
                 data2=None,
                 cmap='jet',
                 vmax=2000):
        super().__init__(parent)
        self.setWindowTitle(
            QCoreApplication.translate("CompareDialog", "Compare"))
        self.setWindowIcon(QIcon(':/icon/compare.png'))
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(700)

        self._data0 = data0
        self._data1 = data1
        self._data2 = data2
        self._cmap = cmap
        self._vmax = vmax

        self.compare_layout = QVBoxLayout(self)
        self.compare_layout.setContentsMargins(5, 5, 5, 5)

        self.canvas_layout = QHBoxLayout(self)
        self.cv = CompareCanvas(self)
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.request_vmax_update)

        self.canvas_layout.addWidget(self.cv)
        self.canvas_layout.addWidget(self.slider)

        self.toolbar = QToolBar("Mask Tools", self)
        self.tool_action_group = QActionGroup(self)
        self.tool_action_group.setExclusive(True)

        pan_icon = QIcon(QPixmap(":/icon/pan.png"))
        self.pan_action = QAction(
            pan_icon, QCoreApplication.translate("CompareDialog", "Pan"), self)
        self.pan_action.setStatusTip(
            QCoreApplication.translate("CompareDialog",
                                       "Pan the canvas (Left-click & Drag)"))
        self.pan_action.setCheckable(True)
        self.pan_action.setChecked(True)
        self.pan_action.triggered.connect(lambda: self.cv.set_mode('pan'))
        self.toolbar.addAction(self.pan_action)
        self.tool_action_group.addAction(self.pan_action)

        draw_ruler_icon = QIcon(QPixmap(":/icon/ruler.png"))
        self.draw_ruler_action = QAction(
            draw_ruler_icon,
            QCoreApplication.translate("CompareDialog", "Ruler"), self)
        self.draw_ruler_action.setStatusTip(
            QCoreApplication.translate(
                "CompareDialog",
                "Draw a ruler (Click & Drag)"
            ))
        self.draw_ruler_action.setCheckable(True)
        self.draw_ruler_action.triggered.connect(
            lambda: self.cv.set_mode('ruler'))
        self.toolbar.addAction(self.draw_ruler_action)
        self.tool_action_group.addAction(self.draw_ruler_action)

        self.compare_buttons = QGroupBox(self)
        self.compare_gridLayout = QHBoxLayout(self.compare_buttons)

        self.btn_01 = QPushButton()
        self.btn_01.setAutoDefault(False)
        self.btn_01.setMinimumHeight(25)
        self.btn_01.setText(
            QCoreApplication.translate("CompareDialog", "Original/Move 1st"))
        self.btn_02 = QPushButton()
        self.btn_02.setAutoDefault(False)
        self.btn_02.setMinimumHeight(25)
        self.btn_02.setText(
            QCoreApplication.translate("CompareDialog", "Original/Move 2nd"))
        self.btn_all = QPushButton()
        self.btn_all.setAutoDefault(False)
        self.btn_all.setMinimumHeight(25)
        self.btn_all.setText(QCoreApplication.translate("CompareDialog", "All"))

        self.btn_01.clicked.connect(lambda: self.compare("move1"))
        self.btn_02.clicked.connect(lambda: self.compare("move2"))
        self.btn_all.clicked.connect(lambda: self.compare("all"))

        self.compare_gridLayout.addWidget(self.btn_01)
        self.compare_gridLayout.addWidget(self.btn_02)
        self.compare_gridLayout.addWidget(self.btn_all)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        ok_button = buttonBox.button(QDialogButtonBox.Ok)
        ok_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        buttonBox.accepted.connect(self.accept)

        self.compare_layout.addWidget(self.toolbar)
        self.compare_layout.addLayout(self.canvas_layout)
        self.compare_layout.addWidget(self.compare_buttons)
        self.compare_layout.addWidget(buttonBox)

        self.setLayout(self.compare_layout)

        self.draw_original()

    def draw_original(self):
        if self._data0 is not None:
            self.slider.setMaximum(self._data0.max())
            self.slider.setValue(min(self._vmax, self._data0.max()))
            ax = self.cv.axes
            ax.imshow(self._data0,
                      cmap=self._cmap,
                      interpolation='nearest',
                      vmin=0,
                      vmax=self._vmax)
            ax.set_aspect(aspect="equal")
            ax.set_axis_on()
            self.cv.draw()

    def compare(self, mode):
        if self._data0 is None:
            return
        ax = self.cv.axes
        ax.clear()
        ax.imshow(self._data0,
                  cmap=self._cmap,
                  interpolation='nearest',
                  vmin=0,
                  vmax=self._vmax,
                  alpha=0.33)

        if mode in ["move1", "all"] and self._data1 is not None:
            ax.imshow(self._data1,
                      cmap=self._cmap,
                      interpolation='nearest',
                      vmin=0,
                      vmax=self._vmax,
                      alpha=0.33)
        if mode in ['move2', 'all'] and self._data2 is not None:
            ax.imshow(self._data2,
                      cmap=self._cmap,
                      interpolation='nearest',
                      vmin=0,
                      vmax=self._vmax,
                      alpha=0.33)
        self.cv.draw()

    def request_vmax_update(self, value):
        try:
            self.cv.set_vlim(vmax=value)
            self.cv.draw_idle()
        except Exception as e:
            QMessageBox.warning(self, 'Error', str(e))
