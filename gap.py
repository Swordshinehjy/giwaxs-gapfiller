import json
import multiprocessing
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import fabio
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import scipy.constants as sc
from fabio import (GEimage, HiPiCimage, OXDimage, adscimage, binaryimage,
                   bruker100image, brukerimage, cbfimage, dm3image, dtrekimage,
                   edfimage, eigerimage, esperantoimage, fabioimage,
                   fit2dimage, fit2dmaskimage, fit2dspreadsheetimage,
                   hdf5image, jpeg2kimage, jpegimage, kcdimage, limaimage,
                   mar345image, marccdimage, mpaimage, mrcimage, numpyimage,
                   openimage, pilatusimage, pixiimage, pnmimage, raxisimage,
                   sparseimage, speimage, templateimage, tifimage, xsdimage)
from PIL import Image
from PySide6.QtCore import (QCoreApplication, QDir, QFile, QModelIndex, Qt,
                            QTextStream, QTranslator)
from PySide6.QtGui import (QAction, QActionGroup, QDoubleValidator, QIcon,
                           QImage, QKeySequence, QPixmap, QShortcut)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox,
                               QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFileSystemModel,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QLineEdit, QListView, QMainWindow, QMenuBar,
                               QMessageBox, QPushButton, QRadioButton,
                               QSizePolicy, QSlider, QSpinBox, QSplitter,
                               QStatusBar, QToolBar, QToolButton, QTreeView,
                               QVBoxLayout, QWidget)
from scipy import integrate

import resources_rc
import utils
from compare import CompareDialog
from custum import MyGraph
from mask import DrawMaskDialog, ShowMaskDialog

warnings.filterwarnings("ignore", 'invalid value')
warnings.filterwarnings("ignore", 'divide by zero')


class MainWindow(QMainWindow):

    def __init__(self, initial_translator=None):
        super().__init__()
        self.resize(1200, 780)
        self.center_on_screen()
        self.setWindowIcon(QIcon(":/icon/scatter.png"))
        self._current_translator = initial_translator
        self.params = {
            "use_pixel": False,
            "x1": 3,
            "y1": -4,
            "x2": 6,
            "y2": -8,
            "detector": "Eiger1M",
            "folder": str(Path.cwd()),
            "flatfield": "",
            "cmap": "jet",
            "vmax": 2000,
            "theme": "default",
        }
        self.themes_qss_map = {
            "default": ":/qss/default.qss",
            "dark": ":/qss/dark.qss",
            "light": ":/qss/light.qss",
        }
        self.mask_data = None
        self.data_original = None
        self.data_first_move = None
        self.data_second_move = None
        self.data_original_path = None

        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Create left/right splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left side layout (vertical)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Calibration GroupBox setup
        self.imageparams = QGroupBox(left_widget)  # Parent explicitly set
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(
            self.imageparams.sizePolicy().hasHeightForWidth())
        self.imageparams.setSizePolicy(sizePolicy1)
        self.imageparams.setAlignment(Qt.AlignLeading | Qt.AlignLeft
                                      | Qt.AlignVCenter)
        self.gridLayout_image = QGridLayout(self.imageparams)
        self.gridLayout_image.setObjectName(u"gridLayout_image")

        self.open_folder = QPushButton(self.imageparams)
        sizePolicy1.setHeightForWidth(
            self.open_folder.sizePolicy().hasHeightForWidth())
        self.open_folder.setSizePolicy(sizePolicy1)
        self.open_folder.setAutoDefault(False)
        self.open_folder.setMinimumHeight(25)
        self.open_folder.clicked.connect(self.browse_folder)
        self.shortcut_browse = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_browse.activated.connect(self.browse_folder)
        self.gridLayout_image.addWidget(self.open_folder, 0, 0, 1, 1)

        self.folder_line = QLineEdit(self.imageparams)
        sizePolicy1.setHeightForWidth(
            self.folder_line.sizePolicy().hasHeightForWidth())
        self.folder_line.setSizePolicy(sizePolicy1)
        self.folder_line.editingFinished.connect(self.navigate_to_folder)
        self.gridLayout_image.addWidget(self.folder_line, 0, 1, 1, 3)

        self.open_flatfield = QPushButton(self.imageparams)
        sizePolicy1.setHeightForWidth(
            self.open_flatfield.sizePolicy().hasHeightForWidth())
        self.open_flatfield.setSizePolicy(sizePolicy1)
        self.open_flatfield.setMinimumHeight(25)
        self.open_flatfield.clicked.connect(self.browse_ff)
        self.gridLayout_image.addWidget(self.open_flatfield, 1, 0, 1, 1)

        self.ff_line = QLineEdit(self.imageparams)
        sizePolicy1.setHeightForWidth(
            self.ff_line.sizePolicy().hasHeightForWidth())
        self.ff_line.setSizePolicy(sizePolicy1)
        self.gridLayout_image.addWidget(self.ff_line, 1, 1, 1, 3)

        self.colormap = QLabel(self.imageparams)
        self.colormap.setAlignment(Qt.AlignCenter)
        self.gridLayout_image.addWidget(self.colormap, 2, 0, 1, 1)

        self.colormap_box = QComboBox(self.imageparams)
        self.colormap_box.addItems([
            "jet", "viridis", "magma", "plasma", "terrain", "bwr", "hot",
            "gray"
        ])
        self.colormap_box.currentTextChanged.connect(self.change_colormap)
        self.gridLayout_image.addWidget(self.colormap_box, 2, 1, 1, 1)

        self.label_vmax = QLabel(self.imageparams)
        self.label_vmax.setAlignment(Qt.AlignCenter)
        self.gridLayout_image.addWidget(self.label_vmax, 2, 2, 1, 1)

        self.intbox = QDoubleSpinBox(self.imageparams)
        self.intbox.setRange(0, 999999999)
        self.intbox.setValue(2000)
        self.intbox.setSingleStep(250.0)
        self.intbox.setDecimals(0)
        self.intbox.valueChanged.connect(self.edit_vlim_main)
        self.gridLayout_image.addWidget(self.intbox, 2, 3, 1, 1)

        self.detector_mask = QRadioButton(self.imageparams)
        self.detector_mask.setAutoRepeatInterval(0)
        self.detector_mask.setChecked(True)
        self.detector_mask.toggled.connect(self.toggle_mask_toolbar)
        self.gridLayout_image.addWidget(self.detector_mask, 3, 0, 1, 1)

        self.detector_label = QLabel(self.imageparams)
        self.detector_label.setAlignment(Qt.AlignCenter)
        self.gridLayout_image.addWidget(self.detector_label, 3, 1, 1, 1)

        self.detector = QComboBox(self.imageparams)
        self.detector.addItems([
            "Eiger1M", "Eiger4M", "Eiger9M", "Eiger16M", "Pilatus1M",
            "Pilatus2M", "Pilatus300K", "Pilatus300K-W"
        ])
        self.detector.currentTextChanged.connect(self.generate_detector_mask)
        self.detector.currentTextChanged.connect(self.update_pixel_size)
        self.gridLayout_image.addWidget(self.detector, 3, 2, 1, 2)

        self.custom_mask = QRadioButton(self.imageparams)
        self.custom_mask.setAutoRepeatInterval(0)
        self.custom_mask.setChecked(False)
        self.custom_mask.toggled.connect(self.toggle_mask_toolbar)
        self.gridLayout_image.addWidget(self.custom_mask, 4, 0, 1, 1)

        self.mask_toolbar = QToolBar("Mask", self.imageparams)
        self.mask_action_group = QActionGroup(self.imageparams)
        self.mask_action_group.setExclusive(True)

        self.open_mask = QAction("", self.imageparams)
        self.open_mask.setCheckable(True)
        self.open_mask.triggered.connect(self.browse_mask)
        self.mask_toolbar.addAction(self.open_mask)
        self.mask_action_group.addAction(self.open_mask)

        self.draw_mask = QAction("", self.imageparams)
        self.draw_mask.setCheckable(True)
        self.draw_mask.triggered.connect(self.open_mask_dialog)
        self.mask_toolbar.addAction(self.draw_mask)
        self.mask_action_group.addAction(self.draw_mask)

        self.no_mask = QAction("", self.imageparams)
        self.no_mask.setCheckable(True)
        self.no_mask.triggered.connect(self.clear_mask)
        self.mask_toolbar.addAction(self.no_mask)
        self.mask_action_group.addAction(self.no_mask)

        self.mask_toolbar.setEnabled(False)
        self.gridLayout_image.addWidget(self.mask_toolbar, 4, 1, 1, 3)

        self.gapFillParams = QGroupBox(left_widget)
        self.gridLayout_move = QGridLayout(self.gapFillParams)

        self.pixel_label = QLabel(self.gapFillParams)
        self.pixel_label.setAlignment(Qt.AlignCenter)
        self.pixel_label.setText("X1")
        self.gridLayout_move.addWidget(self.pixel_label, 0, 0, 1, 1)

        self.pixel = QLineEdit(self.gapFillParams)
        sizePolicy1.setHeightForWidth(
            self.pixel.sizePolicy().hasHeightForWidth())
        self.pixel.setSizePolicy(sizePolicy1)
        self.pixel.setValidator(
            QDoubleValidator(0,
                             1,
                             10,
                             notation=QDoubleValidator.StandardNotation))
        self.pixel.setText("0.075")
        self.gridLayout_move.addWidget(self.pixel, 0, 1, 1, 2)

        self.use_pixel = QRadioButton(self.gapFillParams)
        self.use_pixel.setAutoRepeatInterval(0)
        self.use_pixel.setChecked(False)
        self.use_pixel.toggled.connect(self.unit_conversion)
        self.gridLayout_move.addWidget(self.use_pixel, 0, 3, 1, 1)

        self.use_minimeter = QRadioButton(self.gapFillParams)
        self.use_minimeter.setAutoRepeatInterval(0)
        self.use_minimeter.setChecked(True)
        self.use_minimeter.toggled.connect(self.unit_conversion)
        self.gridLayout_move.addWidget(self.use_minimeter, 0, 4, 1, 1)

        self.move_first = QLabel(self.gapFillParams)
        self.move_first.setAlignment(Qt.AlignCenter)
        self.gridLayout_move.addWidget(self.move_first, 1, 0, 1, 1)

        self.x1_label = QLabel(self.gapFillParams)
        self.x1_label.setAlignment(Qt.AlignCenter)
        self.x1_label.setText("X1")
        self.gridLayout_move.addWidget(self.x1_label, 1, 1, 1, 1)

        self.x1 = QLineEdit(self.gapFillParams)
        sizePolicy1.setHeightForWidth(self.x1.sizePolicy().hasHeightForWidth())
        self.x1.setSizePolicy(sizePolicy1)
        self.x1.setValidator(
            QDoubleValidator(-999999,
                             999999,
                             5,
                             notation=QDoubleValidator.StandardNotation))
        self.x1.setText("3")
        self.gridLayout_move.addWidget(self.x1, 1, 2, 1, 1)

        self.y1_label = QLabel(self.gapFillParams)
        self.y1_label.setAlignment(Qt.AlignCenter)
        self.y1_label.setText("Y1")
        self.gridLayout_move.addWidget(self.y1_label, 1, 3, 1, 1)

        self.y1 = QLineEdit(self.gapFillParams)
        sizePolicy1.setHeightForWidth(self.y1.sizePolicy().hasHeightForWidth())
        self.y1.setSizePolicy(sizePolicy1)
        self.y1.setValidator(
            QDoubleValidator(-999999,
                             999999,
                             5,
                             notation=QDoubleValidator.StandardNotation))
        self.y1.setText("-4")
        self.gridLayout_move.addWidget(self.y1, 1, 4, 1, 1)

        self.move_second = QLabel(self.gapFillParams)
        self.move_second.setAlignment(Qt.AlignCenter)
        self.gridLayout_move.addWidget(self.move_second, 2, 0, 1, 1)

        self.x2_label = QLabel(self.gapFillParams)
        self.x2_label.setAlignment(Qt.AlignCenter)
        self.x2_label.setText("X2")
        self.gridLayout_move.addWidget(self.x2_label, 2, 1, 1, 1)

        self.x2 = QLineEdit(self.gapFillParams)
        sizePolicy1.setHeightForWidth(self.x2.sizePolicy().hasHeightForWidth())
        self.x2.setSizePolicy(sizePolicy1)
        self.x2.setValidator(
            QDoubleValidator(-999999,
                             999999,
                             5,
                             notation=QDoubleValidator.StandardNotation))
        self.x2.setText("6")
        self.gridLayout_move.addWidget(self.x2, 2, 2, 1, 1)

        self.y2_label = QLabel(self.gapFillParams)
        self.y2_label.setAlignment(Qt.AlignCenter)
        self.y2_label.setText("Y2")
        self.gridLayout_move.addWidget(self.y2_label, 2, 3, 1, 1)

        self.y2 = QLineEdit(self.gapFillParams)
        sizePolicy1.setHeightForWidth(self.y2.sizePolicy().hasHeightForWidth())
        self.y2.setSizePolicy(sizePolicy1)
        self.y2.setValidator(
            QDoubleValidator(-999999,
                             999999,
                             5,
                             notation=QDoubleValidator.StandardNotation))
        self.y2.setText("-8")
        self.gridLayout_move.addWidget(self.y2, 2, 4, 1, 1)

        self.file_group = QGroupBox()
        self.file_gridLayout = QGridLayout(self.file_group)
        up_icon = QIcon(QPixmap(":/icon/uparrow.png"))
        self.up_button = QPushButton(up_icon, "", self)
        self.up_button.setAutoDefault(False)
        self.up_button.setMinimumHeight(25)
        self.up_button.clicked.connect(self.navigate_up_directory)
        self.file_gridLayout.addWidget(self.up_button, 0, 0, 1, 1)

        send_to_original_icon = QIcon(QPixmap(":/icon/send0.png"))
        self.send_original_button = QPushButton(send_to_original_icon, "",
                                                self)
        self.send_original_button.setAutoDefault(False)
        self.send_original_button.setMinimumHeight(25)
        self.send_original_button.clicked.connect(
            lambda: self.send_data("original"))
        self.file_gridLayout.addWidget(self.send_original_button, 0, 1, 1, 1)

        send_to_first_icon = QIcon(QPixmap(":/icon/send1.png"))
        self.send_first_button = QPushButton(send_to_first_icon, "", self)
        self.send_first_button.setAutoDefault(False)
        self.send_first_button.setMinimumHeight(25)
        self.send_first_button.clicked.connect(lambda: self.send_data("first"))
        self.file_gridLayout.addWidget(self.send_first_button, 0, 2, 1, 1)

        send_to_second_icon = QIcon(QPixmap(":/icon/send2.png"))
        self.send_second_button = QPushButton(send_to_second_icon, "", self)
        self.send_second_button.setAutoDefault(False)
        self.send_second_button.setMinimumHeight(25)
        self.send_second_button.clicked.connect(
            lambda: self.send_data("second"))
        self.file_gridLayout.addWidget(self.send_second_button, 0, 3, 1, 1)

        send_all_icon = QIcon(QPixmap(":/icon/sendall.png"))
        self.send_all_button = QPushButton(send_all_icon, "", self)
        self.send_all_button.setAutoDefault(False)
        self.send_all_button.setMinimumHeight(25)
        self.send_all_button.clicked.connect(lambda: self.send_data("all"))
        self.file_gridLayout.addWidget(self.send_all_button, 0, 4, 1, 1)

        # File list view
        self.file_list = QListView()
        self.file_list.setResizeMode(QListView.Adjust)
        self.file_list.setWordWrap(True)
        self.file_list.setSpacing(4)
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_model = QFileSystemModel()
        self.file_model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot
                                  | QDir.Files)
        self.file_model.setRootPath("")
        self.file_list.setModel(self.file_model)
        self.file_list.doubleClicked.connect(
            self.handle_file_list_double_click)
        self.file_gridLayout.addWidget(self.file_list, 1, 0, 1, 5)

        left_layout.addWidget(self.imageparams)
        left_layout.addWidget(self.gapFillParams)
        left_layout.addWidget(self.file_group)

        # Right side image preview area
        preview_frame = QWidget(self)
        preview_frame.setObjectName("preview_frame")
        preview_layout = QVBoxLayout(preview_frame)
        canvas_layout = QGridLayout()

        self.original = MyGraph(preview_frame)
        self.moveFirst = MyGraph(preview_frame)
        self.moveSecond = MyGraph(preview_frame)
        self.gapfilled = MyGraph(preview_frame)

        canvas_layout.addWidget(self.original, 0, 0, 1, 1)
        canvas_layout.addWidget(self.moveFirst, 0, 1, 1, 1)
        canvas_layout.addWidget(self.moveSecond, 1, 0, 1, 1)
        canvas_layout.addWidget(self.gapfilled, 1, 1, 1, 1)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(0)
        preview_layout.addLayout(canvas_layout, stretch=1)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        preview_layout.setSpacing(5)

        self.button_layout = QHBoxLayout()  # Vertical layout for buttons

        self.btn_import_params = QPushButton()
        self.btn_export_params = QPushButton()
        self.btn_mask = QPushButton()
        self.btn_compare = QPushButton()
        self.btn_clear = QPushButton()
        self.btn_gapfill = QPushButton()

        self.btn_import_params.setMinimumHeight(25)
        self.btn_export_params.setMinimumHeight(25)
        self.btn_mask.setMinimumHeight(25)
        self.btn_compare.setMinimumHeight(25)
        self.btn_clear.setMinimumHeight(25)
        self.btn_gapfill.setMinimumHeight(25)

        self.button_layout.addWidget(self.btn_import_params)
        self.button_layout.addWidget(self.btn_export_params)
        self.button_layout.addWidget(self.btn_mask)
        self.button_layout.addWidget(self.btn_compare)
        self.button_layout.addWidget(self.btn_clear)
        self.button_layout.addWidget(self.btn_gapfill)

        preview_layout.addLayout(self.button_layout, stretch=1)

        self.btn_import_params.clicked.connect(self.load_settings)
        self.btn_export_params.clicked.connect(self.save_settings)
        self.btn_mask.clicked.connect(self.show_mask)
        self.btn_compare.clicked.connect(self.compare_images)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_gapfill.clicked.connect(self.gapfill)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add widgets to the main splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(preview_frame)
        splitter.setSizes([400, 700])

        # Set initial directory
        self.file_list.setRootIndex(self.file_model.index(""))
        self.status_bar.showMessage("")

        if platform.system() == "Windows":
            self.defaultdir = Path('')
        else:
            self.defaultdir = Path('/')
        # set  default parameters first
        self.set_default_params()
        self.setup_menu_bar()
        self.retranslateUi()
        if self.mask_data is None:
            self.mask_data = utils.generate_detector_mask(self.detector.currentText())

    def setup_menu_bar(self):
        """
        Configures the application's menu bar, including Settings.
        """
        self.main_menu_bar = self.menuBar()

        self.settings_menu = self.main_menu_bar.addMenu("")

        self.theme_submenu = self.settings_menu.addMenu("")
        self.theme_action_group = QActionGroup(self)
        self.theme_action_group.setExclusive(True)

        self.default_theme_action = QAction(self)
        self.default_theme_action.setCheckable(True)
        self.default_theme_action.setChecked(True)
        self.default_theme_action.triggered.connect(
            lambda: self.apply_theme("default"))
        self.theme_submenu.addAction(self.default_theme_action)
        self.theme_action_group.addAction(self.default_theme_action)

        self.dark_theme_action = QAction(self)
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.triggered.connect(
            lambda: self.apply_theme("dark"))
        self.theme_submenu.addAction(self.dark_theme_action)
        self.theme_action_group.addAction(self.dark_theme_action)

        self.light_theme_action = QAction(self)
        self.light_theme_action.setCheckable(True)
        self.light_theme_action.triggered.connect(
            lambda: self.apply_theme("light"))
        self.theme_submenu.addAction(self.light_theme_action)
        self.theme_action_group.addAction(self.light_theme_action)

        self.language_submenu = self.settings_menu.addMenu("")
        self.lang_action_group = QActionGroup(self)
        self.lang_action_group.setExclusive(True)

        self.chinese_lang_action = QAction(self)
        self.chinese_lang_action.setText("中文")
        self.chinese_lang_action.setCheckable(True)
        self.chinese_lang_action.setChecked(True)
        self.chinese_lang_action.triggered.connect(
            lambda: self.change_language("中文"))
        self.language_submenu.addAction(self.chinese_lang_action)
        self.lang_action_group.addAction(self.chinese_lang_action)

        self.english_lang_action = QAction(self)
        self.english_lang_action.setText("English")
        self.english_lang_action.setCheckable(True)
        self.english_lang_action.triggered.connect(
            lambda: self.change_language("English"))
        self.language_submenu.addAction(self.english_lang_action)
        self.lang_action_group.addAction(self.english_lang_action)

        self.theme_actions = {
            "default": self.default_theme_action,
            "dark": self.dark_theme_action,
            "light": self.light_theme_action,
        }
        action_to_check = self.theme_actions.get(self.params["theme"])
        if action_to_check:
            action_to_check.trigger()

        self.help_menu = self.main_menu_bar.addMenu("")

        self.help_doc_action = QAction(self)
        self.help_doc_action.setShortcut(QKeySequence("F1"))
        self.help_doc_action.triggered.connect(self.open_help_pdf)
        self.help_menu.addAction(self.help_doc_action)

        self.about_action = QAction(self)
        self.about_action.triggered.connect(self.show_about_dialog)
        self.help_menu.addAction(self.about_action)

    def retranslateUi(self):
        """
        Sets the translatable strings for the UI elements.
        """
        self.setWindowTitle(
            QCoreApplication.translate("MainWindow", "GIWAXS GapFiller"))
        self.settings_menu.setTitle(
            QCoreApplication.translate("MainWindow", "Menu"))
        self.theme_submenu.setTitle(
            QCoreApplication.translate("MainWindow", "Theme"))
        self.language_submenu.setTitle(
            QCoreApplication.translate("MainWindow", "Language"))
        self.default_theme_action.setText(
            QCoreApplication.translate("MainWindow", "Default Theme"))
        self.dark_theme_action.setText(
            QCoreApplication.translate("MainWindow", "Dark Theme"))
        self.light_theme_action.setText(
            QCoreApplication.translate("MainWindow", "Light Theme"))
        self.help_menu.setTitle(
            QCoreApplication.translate("MainWindow", "Help"))
        self.help_doc_action.setText(
            QCoreApplication.translate("MainWindow", "User Manual"))
        self.about_action.setText(
            QCoreApplication.translate("MainWindow", "About"))
        self.open_folder.setText(
            QCoreApplication.translate("MainWindow", "Folder:"))
        self.open_folder.setToolTip(
            QCoreApplication.translate("MainWindow", "Open Folder") +
            " (Ctrl + O)")
        self.open_flatfield.setText(
            QCoreApplication.translate("MainWindow", "Flat Field:"))
        self.colormap.setText(
            QCoreApplication.translate("MainWindow", "Colormap:"))
        self.label_vmax.setText(
            QCoreApplication.translate("MainWindow", "Vmax:"))
        self.detector_mask.setText(
            QCoreApplication.translate("MainWindow", "Detector Mask"))
        self.custom_mask.setText(
            QCoreApplication.translate("MainWindow", "Custom Mask"))
        self.open_mask.setText(
            QCoreApplication.translate("MainWindow", "Open Mask"))
        self.draw_mask.setText(
            QCoreApplication.translate("MainWindow", "Draw Mask"))
        self.no_mask.setText(
            QCoreApplication.translate("MainWindow", "Clear Mask"))
        self.up_button.setText(QCoreApplication.translate("MainWindow", "Up"))
        self.send_original_button.setText(
            QCoreApplication.translate("MainWindow", "Original"))
        self.send_original_button.setToolTip(
            QCoreApplication.translate("MainWindow", "Send to original image"))
        self.send_first_button.setText(
            QCoreApplication.translate("MainWindow", "Move 1"))
        self.send_first_button.setToolTip(
            QCoreApplication.translate("MainWindow", "Send to first move"))
        self.send_second_button.setText(
            QCoreApplication.translate("MainWindow", "Move 2"))
        self.send_second_button.setToolTip(
            QCoreApplication.translate("MainWindow", "Send to second move"))
        self.send_all_button.setText(
            QCoreApplication.translate("MainWindow", "All"))
        self.send_all_button.setToolTip(
            QCoreApplication.translate("MainWindow", "Send to all images"))
        self.btn_import_params.setText(
            QCoreApplication.translate("MainWindow", "Import Settings"))
        self.btn_export_params.setText(
            QCoreApplication.translate("MainWindow", "Export Settings"))
        self.btn_mask.setText(QCoreApplication.translate("MainWindow", "Mask"))
        self.btn_compare.setText(
            QCoreApplication.translate("MainWindow", "Compare"))
        self.btn_compare.setToolTip(
            QCoreApplication.translate("MainWindow", "Compare images"))
        self.btn_clear.setText(
            QCoreApplication.translate("MainWindow", "Clear"))
        self.btn_clear.setToolTip(
            QCoreApplication.translate("MainWindow", "Clear all images"))
        self.btn_gapfill.setText(
            QCoreApplication.translate("MainWindow", "Gap Fill"))
        self.use_pixel.setText(
            QCoreApplication.translate("MainWindow", "Unit: pixel"))
        self.use_minimeter.setText(
            QCoreApplication.translate("MainWindow", "Unit: mm"))
        self.pixel_label.setText(
            QCoreApplication.translate("MainWindow", "Pixel (mm)"))
        self.pixel_label.setToolTip(
            QCoreApplication.translate("MainWindow", "Pixel size"))
        self.move_first.setText(
            QCoreApplication.translate("MainWindow", "Move 1st"))
        self.move_first.setToolTip(
            QCoreApplication.translate(
                "MainWindow",
                "positive values for the center moving right and downward"))
        self.move_second.setText(
            QCoreApplication.translate("MainWindow", "Move 2nd"))
        self.move_second.setToolTip(
            QCoreApplication.translate(
                "MainWindow",
                "positive values for the center moving right and downward"))
        self.detector_label.setText(
            QCoreApplication.translate("MainWindow", "Detector"))
        self.original.label.setText(
            QCoreApplication.translate("MainWindow", "Original Data"))
        self.moveFirst.label.setText(
            QCoreApplication.translate("MainWindow", "Move 1st"))
        self.moveSecond.label.setText(
            QCoreApplication.translate("MainWindow", "Move 2nd"))
        self.gapfilled.label.setText(
            QCoreApplication.translate("MainWindow", "Gap Filled"))

    def open_help_pdf(self):
        """
        Opens a PDF help file using the default system PDF viewer.
        """
        # Define the path to your PDF help file
        # IMPORTANT: Replace 'path/to/your/help_file.pdf' with the actual path
        # You might want to place it in a 'docs' or 'help' folder within your project
        pdf_path = Path.cwd() / "GIWAXS-Gapfiller使用说明.pdf"

        if not pdf_path.exists():
            QMessageBox.critical(self, "Error",
                                 f"Help file not found at: {pdf_path}")
            print(f"Error: PDF help file not found at {pdf_path}")
            return

        try:
            if sys.platform == "win32":
                os.startfile(pdf_path)  # windows
            elif sys.platform == "darwin":  # macOS
                subprocess.run(["open", pdf_path])
            else:
                subprocess.run(["xdg-open", pdf_path])
            print(f"Opened help file: {pdf_path}")
        except Exception as e:
            self.status_bar.showMessage(f"Error opening help file: {e}")

    def show_about_dialog(self):
        """
        Displays an "About" dialog with software information.
        """
        # You can customize these details
        app_name = "GIWAXS GapFiller"
        version = "1.0.0"
        author = "Jianyao Huang"
        description = "A tool for filling the gap of GIWAXS data."
        license_info = "Licensed under the LGPL License."
        cite = """Please cite:\n
        Jianyao Huang, GIWAXS-Tools. https://gitee.com/swordshinehjy/giwaxs-script.
        """

        info_text = f"""
        <b>{app_name}</b><br>
        Version: {version}<br>
        Author: {author}<br><br>
        {description}<br><br>
        {license_info}<br><br>
        {cite}
        """

        QMessageBox.about(self, f"About {app_name}", info_text)

    def apply_theme(self, theme_key):
        """Applies the QSS from the specified theme file based on its internal key."""
        qss_file_path = self.themes_qss_map.get(theme_key)
        self.params["theme"] = theme_key
        if qss_file_path:
            qss_file = QFile(qss_file_path)
            if qss_file.open(QFile.ReadOnly | QFile.Text):
                stream = QTextStream(qss_file)
                _style = stream.readAll()
                QApplication.instance().setStyleSheet(_style)
                qss_file.close()
                self.status_bar.showMessage(
                    QCoreApplication.translate("MainWindow", "Theme applied"))
            else:
                self.status_bar.showMessage(
                    QCoreApplication.translate("MainWindow",
                                               "Failed to apply theme"))

    def change_language(self, lang):
        if lang == "中文":
            if hasattr(self,
                       '_current_translator') and self._current_translator:
                QApplication.instance().removeTranslator(
                    self._current_translator)

            self.translator = QTranslator()
            if self.translator.load(":/localization/zh_CN.qm"):
                QApplication.instance().installTranslator(self.translator)
                self._current_translator = self.translator
            else:
                if hasattr(self,
                           '_current_translator') and self._current_translator:
                    QApplication.instance().removeTranslator(
                        self._current_translator)
                    self._current_translator = None

        elif lang == "English":
            if hasattr(self,
                       '_current_translator') and self._current_translator:
                QApplication.instance().removeTranslator(
                    self._current_translator)
                self._current_translator = None
        self.retranslateUi()

    def center_on_screen(self):
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        center_x = screen_geometry.center().x()
        center_y = screen_geometry.center().y()
        x = center_x - (self.width() / 2)
        y = center_y - (self.height() / 2)
        self.move(int(x), int(y))

    def edit_vlim_main(self):
        try:
            self.original.update_vmax(self.intbox.value())
            self.moveFirst.update_vmax(self.intbox.value())
            self.moveSecond.update_vmax(self.intbox.value())
            self.gapfilled.update_vmax(self.intbox.value())
        except Exception as e:
            self.status_bar.showMessage(str(e))

    def change_colormap(self):
        try:
            widget = [
                self.original, self.moveFirst, self.moveSecond, self.gapfilled
            ]
            for w in widget:
                if w.canvas.im:
                    w.canvas.im.set_cmap(self.colormap_box.currentText())
                    w.canvas.draw_idle()
        except Exception as e:
            self.status_bar.showMessage(str(e))

    def set_default_params(self):
        default_file = Path.cwd() / "gapfiller_setting.json"
        if default_file.exists():
            try:
                with open(default_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key, value in data.items():
                    if key in self.params:
                        self.params[key] = value
                self.colormap_box.setCurrentText(self.params["cmap"])
                self.intbox.setValue(self.params["vmax"])
                self.update_display(Path(self.params["folder"]))
                self.ff_line.setText(self.params["flatfield"])
                self.ff_line.setCursorPosition(0)
                self.detector.setCurrentText(self.params["detector"])
                if self.mask_data is None:
                    self.mask_data = self.generate_detector_mask()
                self.use_pixel.setChecked(self.params["use_pixel"])
                self.x1.setText(str(self.params["x1"]))
                self.y1.setText(str(self.params["y1"]))
                self.x2.setText(str(self.params["x2"]))
                self.y2.setText(str(self.params["y2"]))
            except (json.JSONDecodeError, IOError, OSError) as e:
                self.status_bar.showMessage(f"Error: {str(e)}")
        

    def toggle_mask_toolbar(self, checked):
        if not checked:
            return
        if self.custom_mask.isChecked():
            self.mask_toolbar.setEnabled(True)
            self.no_mask.setChecked(True)
            self.mask_data = None
        else:
            self.mask_toolbar.setEnabled(False)
            self.mask_data = utils.generate_detector_mask(
                self.detector.currentText())

    def generate_detector_mask(self):
        if self.detector_mask.isChecked():
            self.mask_data = utils.generate_detector_mask(
                self.detector.currentText())

    def unit_conversion(self, checked):
        if not checked:
            return
        try:
            pixel_size = float(self.pixel.text())
            if pixel_size == 0:
                raise ValueError("Pixel size cannot be zero.")

            widgets = [self.x1, self.x2, self.y1, self.y2]

            for w in widgets:
                try:
                    val = float(w.text())
                except ValueError:
                    raise ValueError(f"Invalid input")

                if self.use_minimeter.isChecked():
                    w.setText(f"{(val * pixel_size):.3f}")
                else:
                    w.setText(f"{(val / pixel_size):.5f}")

        except ValueError as e:
            self.status_bar.showMessage(str(e))

    def browse_folder(self):
        try:
            # Use getExistingDirectory to select a folder
            folder = QFileDialog.getExistingDirectory(
                self.central_widget,
                QCoreApplication.translate("MainWindow", "Choose folder"),
                str(self.defaultdir))
        except Exception as e:
            self.status_bar.showMessage(str(e))
            return

        if folder:
            try:
                self.folder_line.clear()
                self.folder_line.setText(folder)
                self.folder_line.setCursorPosition(0)

                self.defaultdir = Path(folder)

                self.file_model.setRootPath(str(self.defaultdir))
                self.file_list.setRootIndex(
                    self.file_model.index(str(self.defaultdir)))

                self.status_bar.showMessage(
                    QCoreApplication.translate("MainWindow", "Folder loaded"))
            except Exception as e:
                self.status_bar.showMessage(f"Error loading folder: {e}")
        else:
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow", "No folder selected"))

    def browse_ff(self):
        try:
            ffFile, _ = QFileDialog.getOpenFileName(
                self.central_widget,
                QCoreApplication.translate("MainWindow",
                                           "Choose Flat Field File"),
                str(self.defaultdir), 'Flat Field (*.*)')
        except:
            return
        if ffFile:
            try:
                fabio.open(ffFile)
                self.ff_line.clear()
                self.ff_line.setText(ffFile)
                self.ff_line.setCursorPosition(0)
            except Exception as e:
                self.status_bar.showMessage(str(e))

    def browse_mask(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, QCoreApplication.translate("MainWindow", "Open Mask File"),
            "", "All Files (*)")
        if fileName:
            try:
                mask = fabio.open(fileName).data > 0
                if np.sum(mask) > mask.size / 2:
                    mask = ~mask
                self.mask_data = mask
                self.status_bar.showMessage(
                    QCoreApplication.translate("MainWindow",
                                               "Mask file loaded"))
            except Exception as e:
                self.status_bar.showMessage(str(e))
                self.no_mask.setChecked(True)
        else:
            self.mask_data = None
            self.status_bar.showMessage(
                QApplication.translate("MainWindow", "No mask file selected"))

    def clear_mask(self):
        self.mask_data = None
        print("mask cleared")

    def update_display(self, new_path: Path):
        """Updates the QLineEdit and QListView to show contents of new_path."""
        if platform.system() == "Windows" and new_path == Path(''):
            self.defaultdir = new_path  # Set defaultdir to Path('')
            self.folder_line.setText("")
            self.folder_line.setCursorPosition(0)
            self.file_model.setRootPath("")
            self.file_list.setRootIndex(QModelIndex())
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow",
                                           "Navigated to My Computer."))
            return

        if not new_path.is_dir():
            new_path = new_path.parent

        self.defaultdir = new_path
        self.folder_line.setText(str(self.defaultdir))
        self.folder_line.setCursorPosition(0)

        # For standard paths, setRootPath and setRootIndex as usual
        self.file_model.setRootPath(str(self.defaultdir))
        self.file_list.setRootIndex(self.file_model.index(str(
            self.defaultdir)))

    def handle_file_list_double_click(self, index: QModelIndex):
        """Handles double-clicks on items in the file_list."""
        file_path = Path(self.file_model.filePath(index))

        # If double-clicking from "My Computer" view onto a drive (e.g., C:\)
        if self.defaultdir == Path(
                '') and file_path.is_dir() and file_path.drive:
            self.update_display(file_path)
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow", "Entered drive: ") +
                f"{file_path.drive}")
        elif file_path.is_dir():
            self.update_display(file_path)
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow", "Entered folder: ") +
                f"{file_path.name}")
        elif file_path.is_file():
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow", "File selected: ") +
                f"{file_path.name}")

    def navigate_to_folder(self):
        """Navigates to the specified folder."""
        folder = self.folder_line.text()
        if not folder:
            self.go_to_system_root()
            return
        folder_path = Path(folder)
        if folder_path.exists():
            if not folder_path.is_dir():
                folder_path = folder_path.parent

            self.update_display(folder_path)
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow", "Navigated to: ") +
                f"{folder_path.name}")
        else:
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow",
                                           "Directory does not exist: ") +
                f"{folder_path}")

    def navigate_up_directory(self):
        """Navigates one level up in the directory structure."""
        if platform.system() == "Windows":
            if self.defaultdir == Path(''):
                self.status_bar.showMessage(
                    QCoreApplication.translate("MainWindow",
                                               "Already at the top level."))
                return
            if self.defaultdir.is_absolute() and \
                self.defaultdir.parent == self.defaultdir and \
                self.defaultdir.drive:
                self.go_to_system_root()
                return

            parent_dir = self.defaultdir.parent
            self.update_display(parent_dir)
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow", "Navigated up to: ") +
                f"{parent_dir.name if parent_dir.name else parent_dir.as_posix()}"
            )
        else:  # Linux/macOS (Unix-like behavior)
            if self.defaultdir == Path('/'):  # Already at '/' root
                self.status_bar.showMessage(
                    QCoreApplication.translate("MainWindow",
                                               "Already at the top level."))
            else:
                parent_dir = self.defaultdir.parent
                self.update_display(parent_dir)
                self.status_bar.showMessage(
                    QCoreApplication.translate("MainWindow",
                                               "Navigated up to: ") +
                    f"{parent_dir.name if parent_dir.name else parent_dir.as_posix()}"
                )

    def go_to_system_root(self):
        """Navigates to the system's root (My Computer on Windows, / on Unix-like)."""
        if platform.system() == "Windows":
            self.update_display(Path(""))
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow",
                                           "Navigated to My Computer."))
        else:
            self.update_display(Path('/'))
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow",
                                           "Navigated to root (/)."))

    def open_mask_dialog(self):
        """
        Opens the calibration settings dialog.
        """
        dialog = DrawMaskDialog(self)
        if dialog.exec() == QDialog.Accepted:
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.mask_data = dialog.get_mask()
            except Exception as e:
                QMessageBox.warning(self, 'Error', str(e))
            finally:
                QApplication.restoreOverrideCursor()
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow", "Mask updated"))
        else:
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow", "Mask not updated"))

    def update_pixel_size(self):
        self.pixel.setText(
            str(utils.get_detector_pixel_size(self.detector.currentText())))

    def send_data(self, mode):
        try:
            all_indexes = self.file_list.selectionModel().selectedIndexes()
            selected_paths = [
                self.file_model.filePath(index) for index in all_indexes
            ]
            if len(selected_paths) == 1:
                data = utils.load_data(selected_paths[0])
                if mode == "original":
                    self.data_original = data
                    self.data_original_path = Path(selected_paths[0])
                elif mode == "first":
                    self.data_first_move = data
                elif mode == "second":
                    self.data_second_move = data
                elif mode == "all":
                    self.data_original = data
                    self.data_original_path = Path(selected_paths[0])
                    self.data_first_move = None
                    self.data_second_move = None
            elif len(selected_paths) > 1 and mode == "all":
                all_data = [utils.load_data(path) for path in selected_paths]
                self.data_original = all_data[0]
                self.data_original_path = Path(selected_paths[0])
                if len(all_data) == 2:
                    self.data_first_move = all_data[1]
                    self.data_second_move = None
                if len(all_data) > 2:
                    self.data_first_move = all_data[1]
                    self.data_second_move = all_data[2]
            self.show_all_data()
        except Exception as e:
            self.status_bar.showMessage(str(e))
            self.data_original = None
            self.data_original_path = None
            self.data_first_move = None
            self.data_second_move = None
            self.show_all_data()

    def show_all_data(self):
        self.show_data(self.data_original, self.original)
        self.show_data(self.data_first_move, self.moveFirst)
        self.show_data(self.data_second_move, self.moveSecond)
        self.show_data(None, self.gapfilled)

    def show_data(self, data, graph: MyGraph):
        try:
            if data is not None:
                max_value = np.percentile(data, 99.999)
                graph.slider.setMaximum(max_value)
                slider_value = min(self.intbox.value(), max_value)
                graph.slider.setValue(slider_value)
                graph.canvas.set_image(data,
                                       cmap=self.colormap_box.currentText(),
                                       vmax=slider_value)
            else:
                graph.canvas.im = None
                graph.canvas.axes.clear()
                graph.canvas.axes.set_axis_off()
                graph.canvas.draw()
            self.status_bar.showMessage(
                QCoreApplication.translate("MainWindow", "Data updated"))
        except Exception as e:
            self.status_bar.showMessage(str(e))

    def clear_all(self):
        self.data_original = None
        self.data_original_path = None
        self.data_first_move = None
        self.data_second_move = None
        self.show_data(None, self.original)
        self.show_data(None, self.moveFirst)
        self.show_data(None, self.moveSecond)
        self.show_data(None, self.gapfilled)

    def show_mask(self):
        try:
            mask = ShowMaskDialog(self, data=self.mask_data)
            if mask.exec() == QDialog.Accepted:
                pass
        except Exception as e:
            self.status_bar.showMessage(str(e))

    def compare_images(self):
        try:
            compare = CompareDialog(self, data0=self.data_original,
                                    data1=self.data_first_move,
                                    data2=self.data_second_move,
                                    cmap=self.colormap_box.currentText(),
                                    vmax=self.intbox.value())
            if compare.exec() == QDialog.Accepted:
                pass
        except Exception as e:
            self.status_bar.showMessage(str(e))

    def gapfill(self):
        try:
            data0 = self.data_original
            path = self.data_original_path
            data1 = self.data_first_move
            data2 = self.data_second_move
            mask = self.mask_data
            if data0 is None or data1 is None or mask is None:
                self.show_data(None, self.gapfilled)
                self.status_bar.showMessage(
                    QCoreApplication.translate("MainWindow", "Missing data"))
                return
            if self.ff_line.text():
                ff = fabio.open(self.ff_line.text()).data
            else:
                ff = np.ones_like(data0)
            x1, y1, x2, y2 = map(float, (self.x1.text(), self.y1.text(),
                                         self.x2.text(), self.y2.text()))
            if self.use_minimeter.isChecked():
                pixelsize = float(self.pixel.text())
                x1 /= pixelsize
                y1 /= pixelsize
                x2 /= pixelsize
                y2 /= pixelsize
            filled = utils.fill_gap(ff, mask, data0, data1, data2, x1, y1, x2,
                                    y2)
            self.show_data(filled, self.gapfilled)
            name = 0
            Image.fromarray(filled).save(path.parent /
                                         f"filled_{path.stem}.tif")
        except Exception as e:
            self.status_bar.showMessage(str(e))

    def load_settings(self):
        try:
            jsonfile, ok = QFileDialog.getOpenFileName(
                self.central_widget,
                QCoreApplication.translate("MainWindow",
                                           "Choose setting file"),
                str(self.defaultdir), 'json Files (*.json)')
            if jsonfile:
                with open(jsonfile, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key, value in data.items():
                    if key in self.params:
                        self.params[key] = value
                self.apply_theme(self.params["theme"])
                self.intbox.setValue(self.params["vmax"])
                self.folder_line.setText(self.params["folder"])
                self.folder_line.setCursorPosition(0)
                self.ff_line.setText(self.params["flatfield"])
                self.ff_line.setCursorPosition(0)
                self.colormap_box.setCurrentText(self.params["cmap"])
                self.detector.setCurrentText(self.params["detector"])
                if not self.detector_mask.isChecked():
                    self.detector_mask.setChecked(True)
                    self.mask_data = self.generate_detector_mask()
                self.update_display(Path(self.params["folder"]))
                self.use_pixel.setChecked(self.params["use_pixel"])
                self.x1.setText(str(self.params["x1"]))
                self.y1.setText(str(self.params["y1"]))
                self.x2.setText(str(self.params["x2"]))
                self.y2.setText(str(self.params["y2"]))
        except Exception as e:
            self.status_bar.showMessage(str(e))

    def save_settings(self):
        try:
            self.params["folder"] = self.folder_line.text()
            self.params["flatfield"] = self.ff_line.text()
            self.params["cmap"] = self.colormap_box.currentText()
            self.params["vmax"] = self.intbox.value()
            self.params["detector"] = self.detector.currentText()
            self.params["use_pixel"] = self.use_pixel.isChecked()
            self.params["x1"] = float(self.x1.text())
            self.params["y1"] = float(self.y1.text())
            self.params["x2"] = float(self.x2.text())
            self.params["y2"] = float(self.y2.text())
            savedFile, ok = QFileDialog.getSaveFileName(
                self.central_widget,
                QCoreApplication.translate("MainWindow", "Save setting file"),
                str(self.defaultdir / "default.json"), 'json Files (*.json)')
            if savedFile:
                with open(savedFile, "w", encoding='utf-8') as f:
                    json.dump(self.params, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.status_bar.showMessage(str(e))


if __name__ == "__main__":

    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    initial_translator = None
    loaded_translator = QTranslator()
    if loaded_translator.load(":/localization/zh_CN.qm"):
        app.installTranslator(loaded_translator)
        initial_translator = loaded_translator
    browser = MainWindow(initial_translator=initial_translator)
    browser.show()
    sys.exit(app.exec())
