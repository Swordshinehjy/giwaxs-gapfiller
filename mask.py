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

from custum import MaskCanvas, MyCanvas


class ShowMaskDialog(QDialog):

    def __init__(self, parent=None, data=None):
        super().__init__(parent)
        self.setWindowTitle(
            QCoreApplication.translate("ShowMaskDialog", "Mask"))
        self.setWindowIcon(QIcon(':/icon/mask.png'))
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(700)

        self.mask_layout = QVBoxLayout(self)
        self.mask_layout.setContentsMargins(5, 5, 5, 5)

        self.cv = MyCanvas(self)
        if data is not None:
            ax = self.cv.axes
            ax.imshow(data,
                      cmap='gray',
                      interpolation='nearest',
                      vmin=0,
                      vmax=1)
            ax.set_aspect(aspect="equal")
            ax.set_axis_on()
            ax.set_title('Mask')
            self.cv.draw()
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok)
        ok_button = buttonBox.button(QDialogButtonBox.Ok)
        ok_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        buttonBox.accepted.connect(self.accept)
        self.mask_layout.addWidget(self.cv)
        self.mask_layout.addWidget(buttonBox)

        self.setLayout(self.mask_layout)


class DrawMaskDialog(QDialog):

    def __init__(self, parent=None):  # Added initial_params for pre-filling
        super().__init__(parent)
        self.setWindowTitle(QCoreApplication.translate("DrawMaskDialog", "Mask"))
        self.setWindowIcon(QIcon(':/icon/mask.png'))
        self.setModal(True)  # Make the dialog modal
        self.setMinimumWidth(600)
        self.setMinimumHeight(700)

        self.mask_widget = QWidget(self)
        self.mask_layout = QVBoxLayout(self)
        self.mask_layout.setContentsMargins(5, 5, 5, 5)

        self.canvas_area = QWidget(self.mask_widget)
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.canvas_area.sizePolicy().hasHeightForWidth())
        self.canvas_area.setSizePolicy(sizePolicy1)
        self.canvas_area.setObjectName("canvas_area")
        self.canvas_area.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        self.canvas_layout = QHBoxLayout(self.canvas_area)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.cv = MaskCanvas(self.canvas_area)
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.request_vmax_update)

        self.canvas_layout.addWidget(self.cv)
        self.canvas_layout.addWidget(self.slider)

        self.toolbar = QToolBar("Mask Tools", self.mask_widget)
        self.tool_action_group = QActionGroup(self)
        self.tool_action_group.setExclusive(True)

        # Pan Action
        pan_icon = QIcon(QPixmap(":/icon/pan.png"))
        self.pan_action = QAction(
            pan_icon, QCoreApplication.translate("DrawMaskDialog", "Pan"), self)
        self.pan_action.setStatusTip(
            QCoreApplication.translate("DrawMaskDialog",
                                       "Pan the canvas (Left-click & Drag)"))
        self.pan_action.setCheckable(True)
        self.pan_action.setChecked(True)
        self.pan_action.triggered.connect(lambda: self.cv.set_mode('pan'))
        self.toolbar.addAction(self.pan_action)
        self.tool_action_group.addAction(self.pan_action)

        # Draw Circle Action
        draw_ellipse_icon = QIcon(QPixmap(":/icon/ellipse.png"))
        self.draw_ellipse_action = QAction(
            draw_ellipse_icon,
            QCoreApplication.translate("DrawMaskDialog", "Draw Ellipse"), self)
        self.draw_ellipse_action.setStatusTip(
            QCoreApplication.translate(
                "DrawMaskDialog",
                "Draw circles/ellipses (Left-click & Drag, Shift for perfect circle)"
            ))
        self.draw_ellipse_action.setCheckable(True)
        self.draw_ellipse_action.triggered.connect(
            lambda: self.cv.set_mode('draw_circle'))
        self.toolbar.addAction(self.draw_ellipse_action)
        self.tool_action_group.addAction(self.draw_ellipse_action)

        rectangle_icon = QIcon(QPixmap(":/icon/rectangle.png"))
        self.draw_rectangle_action = QAction(
            rectangle_icon,
            QCoreApplication.translate("DrawMaskDialog", "Draw Rectangle"), self)
        self.draw_rectangle_action.setStatusTip(
            QCoreApplication.translate("DrawMaskDialog",
                                       "Draw rectangles (Left-click & Drag)"))
        self.draw_rectangle_action.setCheckable(True)
        self.draw_rectangle_action.triggered.connect(
            lambda: self.cv.set_mode('draw_rectangle'))
        self.toolbar.addAction(self.draw_rectangle_action)
        self.tool_action_group.addAction(self.draw_rectangle_action)

        polygon_icon = QIcon(QPixmap(":/icon/polygon.png"))
        self.draw_polygon_action = QAction(
            polygon_icon,
            QCoreApplication.translate("DrawMaskDialog", "Draw Polygon"), self)
        self.draw_polygon_action.setStatusTip(
            QCoreApplication.translate(
                "DrawMaskDialog",
                "Draw polygons (Single click for vertex, Double-click to close)"
            ))
        self.draw_polygon_action.setCheckable(True)
        self.draw_polygon_action.triggered.connect(
            lambda: self.cv.set_mode('draw_polygon'))
        self.toolbar.addAction(self.draw_polygon_action)
        self.tool_action_group.addAction(self.draw_polygon_action)

        smudge_icon = QIcon(QPixmap(":/icon/painter.png"))
        self.draw_smudge_action = QAction(
            smudge_icon, QCoreApplication.translate("DrawMaskDialog", "Smudge"),
            self)
        self.draw_smudge_action.setStatusTip(
            QCoreApplication.translate("DrawMaskDialog",
                                       "Smudge (Click and drag)"))
        self.draw_smudge_action.setCheckable(True)
        self.draw_smudge_action.triggered.connect(
            lambda: self.cv.set_mode('smudge'))
        self.toolbar.addAction(self.draw_smudge_action)
        self.tool_action_group.addAction(self.draw_smudge_action)

        self.smudge_radius_slider = QSlider(Qt.Horizontal)
        self.smudge_radius_slider.setMinimum(1)
        self.smudge_radius_slider.setMaximum(
            20)  # Max radius, adjust as needed
        self.smudge_radius_slider.setValue(
            self.cv.smudge_radius)  # Get initial value from canvas
        self.smudge_radius_slider.setTickPosition(QSlider.TicksBelow)
        self.smudge_radius_slider.setTickInterval(1)
        self.smudge_radius_slider.setFixedWidth(120)
        self.smudge_radius_slider.valueChanged.connect(
            self.update_smudge_radius)
        
        self.coordinate_label = QLabel(self)
        self.pixel_value_label = QLabel(self)
        self.spacer_widget = QWidget(self)
        self.spacer_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Layout for toolbar and smudge slider (horizontal arrangement)
        tool_controls_layout = QHBoxLayout()
        tool_controls_layout.addWidget(self.toolbar)
        tool_controls_layout.addWidget(self.smudge_radius_slider)
        tool_controls_layout.addWidget(self.spacer_widget)
        tool_controls_layout.addWidget(self.coordinate_label)
        tool_controls_layout.addWidget(self.pixel_value_label)
        tool_controls_layout.addStretch()  # Push everything to the left

        self.cv.mouse_data_coords_changed.connect(self.update_mouse_info)

        self.make_mask = QGroupBox(self.canvas_area)
        self.gridLayout_setting = QGridLayout(self.make_mask)
        self.gridLayout_setting.setObjectName(u"gridLayout_calib")

        self.open_file = QPushButton(self.make_mask)
        self.open_file.setAutoDefault(False)
        self.open_file.setMinimumHeight(25)
        self.open_file.setText(
            QCoreApplication.translate("DrawMaskDialog", "Open File"))
        self.open_file.clicked.connect(self.openFile)
        self.gridLayout_setting.addWidget(self.open_file, 0, 0, 1, 1)

        self.save_file = QPushButton(self.make_mask)
        self.save_file.setAutoDefault(False)
        self.save_file.setMinimumHeight(25)
        self.save_file.setText(
            QCoreApplication.translate("DrawMaskDialog", "Save Mask"))
        self.save_file.clicked.connect(self.save_mask)
        self.gridLayout_setting.addWidget(self.save_file, 0, 1, 1, 1)

        self.label_intmax = QLabel(self.make_mask)
        self.label_intmax.setAlignment(Qt.AlignCenter)
        self.label_intmax.setText(
            QCoreApplication.translate("DrawMaskDialog", "Vmax:"))
        self.gridLayout_setting.addWidget(self.label_intmax, 0, 2, 1, 1)

        self.intmax_box = QDoubleSpinBox(self.make_mask)
        self.intmax_box.setRange(0, 999999999)
        self.intmax_box.setValue(2000)
        self.intmax_box.setSingleStep(250.0)
        self.intmax_box.setDecimals(0)
        self.intmax_box.valueChanged.connect(self.edit_vlim)
        self.gridLayout_setting.addWidget(self.intmax_box, 0, 3, 1, 1)

        self.mask_above = QCheckBox(self.make_mask)
        self.mask_above.setText(
            QCoreApplication.translate("DrawMaskDialog", "Mask above"))
        self.mask_above.setChecked(False)
        self.mask_above.setAutoRepeatInterval(0)
        self.gridLayout_setting.addWidget(self.mask_above, 1, 0, 1, 1)

        self.mask_above_box = QSpinBox(self.make_mask)
        self.mask_above_box.setRange(0, 999999)
        self.mask_above_box.setValue(10000)
        self.mask_above_box.setSingleStep(1000)
        self.gridLayout_setting.addWidget(self.mask_above_box, 1, 1, 1, 1)

        self.mask_below = QCheckBox(self.make_mask)
        self.mask_below.setText(
            QCoreApplication.translate("DrawMaskDialog", "Mask below"))
        self.mask_below.setChecked(False)
        self.mask_below.setAutoRepeatInterval(0)
        self.gridLayout_setting.addWidget(self.mask_below, 1, 2, 1, 1)

        self.mask_below_box = QSpinBox(self.make_mask)
        self.mask_below_box.setRange(-999999, 999999)
        self.mask_below_box.setValue(0)
        self.mask_below_box.setSingleStep(1000)
        self.gridLayout_setting.addWidget(self.mask_below_box, 1, 3, 1, 1)

        self.dialogButtonBox = QWidget(self.mask_widget)
        self.button_layout = QHBoxLayout(self.dialogButtonBox)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok
                                          | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        ok_button = self.buttonBox.button(QDialogButtonBox.Ok)
        ok_button.setAutoDefault(True)
        cancel_button = self.buttonBox.button(QDialogButtonBox.Cancel)

        if ok_button:
            ok_button.setSizePolicy(QSizePolicy.Expanding,
                                    QSizePolicy.Expanding)
        if cancel_button:
            cancel_button.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Expanding)
        self.button_layout.addWidget(ok_button, stretch=1)
        self.button_layout.addWidget(cancel_button, stretch=1)

        self.mask_layout.addLayout(tool_controls_layout)
        self.mask_layout.addWidget(self.canvas_area, 1)
        self.mask_layout.addWidget(self.make_mask, 0)
        self.mask_layout.addWidget(self.dialogButtonBox, 0)

        plt.rcParams['font.size'] = 5
        self.xCoords = []
        self.yCoords = []
        self.cid = None

    def update_smudge_radius(self, value):
        self.cv.set_smudge_radius(value)

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, QCoreApplication.translate("DrawMaskDialog", "Open Mask File"),
            "", "All Files (*)")
        if fileName:
            try:
                imarray = fabio.open(fileName).data
                max_value = np.percentile(imarray, 99.999)
                self.slider.setMaximum(max_value)
                slider_value = min(self.intmax_box.value(), max_value)
                self.slider.setValue(slider_value)
                self.cv.im = imarray
                ax = self.cv.axes
                ax.clear()
                ax.imshow(imarray,
                          cmap='jet',
                          interpolation='nearest',
                          vmin=0,
                          vmax=slider_value)
                ax.set_aspect(aspect='equal')
                ax.yaxis.set_ticks_position('left')
                ax.xaxis.set_ticks_position('bottom')
                # ax.set_axis_off()
                self.cv.draw()
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))
                return

    def request_vmax_update(self, value):
        try:
            for im in self.cv.axes.get_images():
                im.set_clim(vmin=0, vmax=value)
            self.intmax_box.setValue(value)
            self.cv.draw_idle()
        except Exception as e:
            QMessageBox.warning(self, 'Error', str(e))

    def edit_vlim(self):
        try:
            for im in self.cv.axes.get_images():
                im.set_clim(vmin=0, vmax=self.intmax_box.value())
            self.slider.setValue(self.intmax_box.value())
            self.cv.draw_idle()
        except Exception as e:
            QMessageBox.warning(self, 'Error', str(e))

        except Exception as e:
            QMessageBox.warning(self, 'Error', str(e))

    def update_mouse_info(self, x, y, value):
        """Updates the labels with mouse coordinates and pixel value."""
        self.coordinate_label.setText(f"({x}, {y})")
        self.pixel_value_label.setText(f"{value:.2f}")

    def get_mask(self):
        if self.mask_above.isChecked():
            self.cv.mask_above_value = self.mask_above_box.value()
        else:
            self.cv.mask_above_value = None
        if self.mask_below.isChecked():
            self.cv.mask_below_value = self.mask_below_box.value()
        else:
            self.cv.mask_below_value = None
        return self.cv.get_mask()

    def save_mask(self):
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            mask = self.get_mask()
            if mask is not None:
                savedFile, _ = QFileDialog.getSaveFileName(
                    self,
                    QCoreApplication.translate("DrawMaskDialog", "Save Mask file"),
                    str(Path.cwd()), 'tif Files (*.tif)')
                if savedFile:
                    Image.fromarray(mask).save(savedFile)
        except:
            QMessageBox.warning(self, "Error", "No mask found")
        finally:
            QApplication.restoreOverrideCursor()
