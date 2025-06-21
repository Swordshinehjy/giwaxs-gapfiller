from pathlib import Path

import fabio
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon, Rectangle
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QSizePolicy, QSlider,
                               QToolBar, QVBoxLayout, QWidget)


class ScrollableMixin:

    def _setup_scroll(self):
        # This method assumes the class using the mixin has 'self.axes' and 'self.mpl_connect'
        self.mpl_connect('scroll_event', self.on_scroll)

    def on_scroll(self, event):
        if event.inaxes != self.axes:
            return

        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()
        x_data, y_data = event.xdata, event.ydata

        if event.button == 'up':
            scale_factor = 0.8
        elif event.button == 'down':
            scale_factor = 1.25
        else:
            return

        new_x_min = x_data - (x_data - x_min) * scale_factor
        new_x_max = x_data + (x_max - x_data) * scale_factor
        new_y_min = y_data - (y_data - y_min) * scale_factor
        new_y_max = y_data + (y_max - y_data) * scale_factor

        self.axes.set_xlim(new_x_min, new_x_max)
        self.axes.set_ylim(new_y_min, new_y_max)
        self.draw()


class MyCanvas(FigureCanvas, ScrollableMixin):

    def __init__(self, parent=None, width=3, height=4, dpi=150):
        self.fig = plt.figure(figsize=(width, height),
                              dpi=dpi,
                              tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        self.axes = self.fig.add_subplot(111)
        self.setParent(parent)
        self.axes.set_axis_off()
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self._setup_scroll()
        self.pan_start_x = None
        self.pan_start_y = None
        self.ispanning = False
        self.im = None
        self.mpl_connect('button_press_event', self.on_mouse_press)
        self.mpl_connect('button_release_event', self.on_mouse_release)
        self.mpl_connect('motion_notify_event', self.on_mouse_motion)

    def on_mouse_press(self, event):
        if event.button == 1 and event.inaxes == self.axes:
            self.ispanning = True
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata

    def on_mouse_release(self, event):
        if event.button == 1:
            self.ispanning = False
            self.pan_start_x = None
            self.pan_start_y = None

    def on_mouse_motion(self, event):
        if self.ispanning and event.inaxes == self.axes:
            if event.xdata is None or event.ydata is None:
                return
            dx = event.xdata - self.pan_start_x
            dy = event.ydata - self.pan_start_y
            x_min, x_max = self.axes.get_xlim()
            y_min, y_max = self.axes.get_ylim()
            self.axes.set_xlim(x_min - dx, x_max - dx)
            self.axes.set_ylim(y_min - dy, y_max - dy)
            self.draw()

    def set_vlim(self, vmin=None, vmax=None):
        if self.im:
            current_vmin, current_vmax = self.im.get_clim()
            new_vmin = vmin if vmin is not None else current_vmin
            new_vmax = vmax if vmax is not None else current_vmax
            self.im.set_clim(vmin=new_vmin, vmax=new_vmax)
            self.draw()

    def set_image(self, data, cmap='jet', vmax=2000):
        try:
            if self.im:
                self.im.set_data(data)
                self.im.set_cmap(cmap)
                self.im.set_clim(vmin=0, vmax=vmax)
            else:
                self.im = self.axes.imshow(data,
                                           cmap=cmap,
                                           interpolation='nearest',
                                           vmin=0,
                                           vmax=vmax)
            self.axes.set_axis_off()
            self.draw()
        except Exception as e:
            print(f"Error loading data: {e}")


class MyGraph(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.canvas = MyCanvas()
        self.hbl = QHBoxLayout()
        self.vbl = QVBoxLayout()

        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(0, 4000)
        self.slider.setValue(2000)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.update_vmax)

        self.hbl.addWidget(self.canvas)
        self.hbl.addWidget(self.slider)
        self.vbl.addWidget(self.label)
        self.vbl.addLayout(self.hbl)
        self.setLayout(self.vbl)

        self.vbl.setContentsMargins(0, 0, 0, 0)
        self.vbl.setSpacing(0)
        self.hbl.setContentsMargins(0, 0, 0, 0)
        self.hbl.setSpacing(0)

    def update_vmax(self, value):
        self.canvas.set_vlim(vmax=value)
        self.slider.setValue(value)


class CompareCanvas(FigureCanvas, ScrollableMixin):

    def __init__(self, parent=None, width=3, height=4, dpi=150):
        self.fig = plt.figure(figsize=(width, height),
                              dpi=dpi,
                              tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        self.axes = self.fig.add_subplot(111)
        self.setParent(parent)
        self.axes.set_axis_off()
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self._setup_scroll()
        self.mode = 'pan'

        self.pan_start_x = None
        self.pan_start_y = None
        self.is_panning = False

        self.ruler_start_x = None
        self.ruler_start_y = None
        self.ruler_line = None
        self.ruler_text = None
        self.is_measuring = False

        self.mpl_connect('button_press_event', self.on_button_press)
        self.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.mpl_connect('button_release_event', self.on_button_release)

    def set_mode(self, mode):
        self.mode = mode
        self.is_panning = False
        self.is_measuring = False
        if self.ruler_line:
            self.ruler_line.remove()
            self.ruler_line = None
        if self.ruler_text:
            self.ruler_text.remove()
            self.ruler_text = None
        self.draw_idle()

    def on_button_press(self, event):
        self.setFocus()
        if event.inaxes != self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        if self.mode == 'pan' and event.button == 1:
            self.is_panning = True
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata
        elif self.mode == 'ruler' and event.button == 1:
            self.is_measuring = True
            self.ruler_start_x = event.xdata
            self.ruler_start_y = event.ydata
            # Initialize line and text, will update in motion
            self.ruler_line, = self.axes.plot([], [], 'r--', lw=1)
            self.ruler_text = self.axes.text(0,
                                             0,
                                             '',
                                             color='red',
                                             fontsize=8,
                                             ha='left',
                                             va='bottom')
            self.draw_idle()

    def on_mouse_motion(self, event):
        if event.inaxes != self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Panning
        if self.is_panning and self.mode == 'pan':
            dx = event.xdata - self.pan_start_x
            dy = event.ydata - self.pan_start_y
            x_min, x_max = self.axes.get_xlim()
            y_min, y_max = self.axes.get_ylim()

            new_x_min = x_min - dx
            new_x_max = x_max - dx
            new_y_min = y_min - dy
            new_y_max = y_max - dy

            self.axes.set_xlim(new_x_min, new_x_max)
            self.axes.set_ylim(new_y_min, new_y_max)
            self.draw_idle()
        elif self.is_measuring and self.mode == 'ruler':
            current_x = event.xdata
            current_y = event.ydata

            self.ruler_line.set_data([self.ruler_start_x, current_x],
                                     [self.ruler_start_y, current_y])

            dx_data = current_x - self.ruler_start_x
            dy_data = current_y - self.ruler_start_y

            text_x = (self.ruler_start_x + current_x) / 2
            text_y = (self.ruler_start_y + current_y) / 2

            self.ruler_text.set_position((text_x, text_y))
            self.ruler_text.set_text(
                f'dX: {dx_data:.2f} px\ndY: {dy_data:.2f} px')
            self.draw_idle()

    def on_button_release(self, event):
        if self.mode == 'pan' and event.button == 1:
            self.is_panning = False
            self.pan_start_x = None
            self.pan_start_y = None
        elif self.mode == 'ruler' and event.button == 1:
            self.is_measuring = False

    def set_vlim(self, vmin=None, vmax=None):
        for im in self.axes.get_images():
            current_vmin, current_vmax = im.get_clim()
            new_vmin = vmin if vmin is not None else current_vmin
            new_vmax = vmax if vmax is not None else current_vmax
            im.set_clim(vmin=new_vmin, vmax=new_vmax)


class MaskCanvas(FigureCanvas, ScrollableMixin):
    mouse_data_coords_changed = Signal(int, int, float)

    def __init__(self, parent=None, width=4, height=4, dpi=150):
        self.fig = plt.figure(figsize=(width, height),
                              dpi=dpi,
                              tight_layout=True)
        FigureCanvas.__init__(self, self.fig)
        self.axes = self.fig.add_subplot(111)
        self.setParent(parent)
        self.axes.set_axis_off()
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.im = None
        self.mask_above_value = None
        self.mask_below_value = None
        self._setup_scroll()
        # Default mode
        self.mode = 'pan'
        # Panning attributes
        self.pan_start_x = None
        self.pan_start_y = None
        self.is_panning = False
        # Drawing attributes (shared for ellipse/rectangle)
        self.drawing_active = False
        self.start_x = None
        self.start_y = None
        self.current_shape = None
        self.shift_is_pressed = False
        # Polygon specific attributes
        self.polygon_points = []
        self.current_polygon_line = None  # Line connecting last point to mouse
        # Smudge tool attributes
        self.is_smudging = False
        self.smudge_points = []
        self.current_smudge_line = None
        self.smudge_radius = 5  # Default smudge radius
        self.smudge_lines = []

        self.mpl_connect('button_press_event', self.on_button_press)
        self.mpl_connect('motion_notify_event', self.on_mouse_motion)
        self.mpl_connect('button_release_event', self.on_button_release)
        self.mpl_connect('key_press_event', self.on_key_press)
        self.mpl_connect('key_release_event', self.on_key_release)
        self.mpl_connect('motion_notify_event', self._on_mouse_move)

    def set_mode(self, mode):
        """Sets the current operating mode of the canvas."""
        # Clean up only active drawing elements when changing mode

        # Clear polygon drawing state if it's in progress
        if self.mode == 'draw_polygon' and self.polygon_points:
            print("Clearing polygon drawing state when switching mode")
            # clear polygon drawing state
            self.clear_unfinished_polygon()

        # Clear unfinished ellipse/rectangle
        if self.current_shape and self.drawing_active:
            print("Removing unfinished shape when switching mode")
            self.current_shape.remove()
            self.current_shape = None
            self.drawing_active = False
        # Clear smudge drawing state if it's in progress
        if self.mode == 'smudge' and self.is_smudging:
            self.is_smudging = False
            self.smudge_points = []
            if self.current_smudge_line:
                self.current_smudge_line.remove()
                self.current_smudge_line = None
        self.mode = mode
        self.is_panning = False
        self.draw_idle()
        # print(f"Canvas mode set to: {self.mode}")

    def set_smudge_radius(self, radius):
        """Sets the radius for the smudge tool."""
        self.smudge_radius = radius
        # print(f"Smudge radius set to: {self.smudge_radius}")

    def on_button_press(self, event):
        self.setFocus()
        if event.inaxes != self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Panning logic
        if self.mode == 'pan' and event.button == 1:
            self.is_panning = True
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata

        # Circle drawing logic
        elif self.mode == 'draw_circle' and event.button == 1:
            self.drawing_active = True
            self.start_x = event.xdata
            self.start_y = event.ydata
            self.current_shape = Ellipse((self.start_x, self.start_y),
                                         0,
                                         0,
                                         facecolor='lightgray',
                                         edgecolor='none',
                                         linewidth=1,
                                         linestyle='-',
                                         alpha=0.5)
            self.axes.add_patch(self.current_shape)
            self.draw_idle()

        # Rectangle drawing logic
        elif self.mode == 'draw_rectangle' and event.button == 1:
            self.drawing_active = True
            self.start_x = event.xdata
            self.start_y = event.ydata
            self.current_shape = Rectangle((self.start_x, self.start_y),
                                           0,
                                           0,
                                           facecolor='lightgray',
                                           edgecolor='none',
                                           linewidth=1,
                                           linestyle='-',
                                           alpha=0.5)
            self.axes.add_patch(self.current_shape)
            self.draw_idle()

        # Polygon drawing logic
        elif self.mode == 'draw_polygon' and event.button == 1:
            x, y = event.xdata, event.ydata

            # double click to close polygon
            if event.dblclick:
                if len(self.polygon_points) >= 3:
                    # Remove temporary line
                    if self.current_polygon_line:
                        self.current_polygon_line.remove()
                        self.current_polygon_line = None

                    # add the last line
                    last_pt = self.polygon_points[-1]
                    first_pt = self.polygon_points[0]

                    # check if the first point is the same as the last point
                    line_exists = False
                    for line in self.axes.lines:
                        data = line.get_xydata()
                        if len(data) == 2:
                            if (np.array_equal(data[0], last_pt) and
                                np.array_equal(data[1], first_pt)) or \
                            (np.array_equal(data[0], first_pt) and
                                np.array_equal(data[1], last_pt)):
                                line_exists = True
                                break

                    if not line_exists:
                        self.axes.plot([last_pt[0], first_pt[0]],
                                       [last_pt[1], first_pt[1]],
                                       color='lightgray',
                                       linestyle='-',
                                       alpha=0.5)

                    final_polygon = Polygon(self.polygon_points,
                                            facecolor='lightgray',
                                            edgecolor='none',
                                            linewidth=1,
                                            alpha=0.5)
                    self.axes.add_patch(final_polygon)
                    self.polygon_points = []  # Reset for next polygon
                    self.draw_idle()
                else:
                    print("Need at least 3 points to close a polygon.")
            else:  # Single click for a new vertex
                self.polygon_points.append((x, y))
                self.axes.plot(x, y, color='lightgray')

                if len(self.polygon_points) > 1:
                    last_x, last_y = self.polygon_points[-2]
                    self.axes.plot([last_x, x], [last_y, y],
                                   color='lightgray',
                                   linestyle='-',
                                   alpha=0.5)

                # Create/update temporary line from last point to mouse cursor
                if self.current_polygon_line:
                    self.current_polygon_line.remove()
                self.current_polygon_line = Line2D([x, x], [y, y],
                                                   color='lightgray',
                                                   linestyle='--',
                                                   alpha=0.5)
                self.axes.add_line(self.current_polygon_line)

                self.draw_idle()
        # Smudge drawing logic
        elif self.mode == 'smudge' and event.button == 1:
            self.is_smudging = True
            self.smudge_points = [(event.xdata, event.ydata)]
            # Use self.smudge_radius for linewidth
            self.current_smudge_line = Line2D([], [],
                                              color='lightgray',
                                              linewidth=self.smudge_radius,
                                              solid_capstyle='round',
                                              alpha=0.5)
            self.axes.add_line(self.current_smudge_line)
            self.draw_idle()

    def on_mouse_motion(self, event):
        if event.inaxes != self.axes:
            return
        if event.xdata is None or event.ydata is None:
            return

        # Panning
        if self.is_panning and self.mode == 'pan':
            dx = event.xdata - self.pan_start_x
            dy = event.ydata - self.pan_start_y
            x_min, x_max = self.axes.get_xlim()
            y_min, y_max = self.axes.get_ylim()

            new_x_min = x_min - dx
            new_x_max = x_max - dx
            new_y_min = y_min - dy
            new_y_max = y_max - dy

            self.axes.set_xlim(new_x_min, new_x_max)
            self.axes.set_ylim(new_y_min, new_y_max)
            self.draw_idle()

        # Circle drawing
        elif self.drawing_active and self.mode == 'draw_circle':
            end_x, end_y = event.xdata, event.ydata

            if self.shift_is_pressed:
                dx = end_x - self.start_x
                dy = end_y - self.start_y
                max_delta = max(abs(dx), abs(dy))
                square_end_x = self.start_x + max_delta * np.sign(dx)
                square_end_y = self.start_y + max_delta * np.sign(dy)
                width = max_delta
                height = max_delta
                center_x = (self.start_x + square_end_x) / 2
                center_y = (self.start_y + square_end_y) / 2

            else:
                center_x = (self.start_x + end_x) / 2
                center_y = (self.start_y + end_y) / 2
                width = abs(end_x - self.start_x)
                height = abs(end_y - self.start_y)

            self.current_shape.center = (center_x, center_y)
            self.current_shape.width = width
            self.current_shape.height = height
            self.draw_idle()

        # Rectangle drawing
        elif self.drawing_active and self.mode == 'draw_rectangle':
            end_x, end_y = event.xdata, event.ydata
            width = abs(end_x - self.start_x)
            height = abs(end_y - self.start_y)
            rect_x = min(self.start_x, end_x)
            rect_y = min(self.start_y, end_y)

            self.current_shape.set_xy((rect_x, rect_y))
            self.current_shape.set_width(width)
            self.current_shape.set_height(height)
            self.draw_idle()

        # Polygon drawing - update temporary line
        elif self.mode == 'draw_polygon' and self.polygon_points:
            if self.current_polygon_line:
                last_x, last_y = self.polygon_points[-1]
                self.current_polygon_line.set_data([last_x, event.xdata],
                                                   [last_y, event.ydata])
                self.draw_idle()
        # Smudge drawing
        elif self.is_smudging and self.mode == 'smudge':
            self.smudge_points.append((event.xdata, event.ydata))
            x_data, y_data = zip(*self.smudge_points)
            self.current_smudge_line.set_data(x_data, y_data)
            self.draw_idle()

    def on_button_release(self, event):
        # Only pan and continuous drawing modes have release actions here
        if self.mode == 'pan' and event.button == 1:
            self.is_panning = False
            self.pan_start_x = None
            self.pan_start_y = None

        elif (self.mode == 'draw_circle'
              or self.mode == 'draw_rectangle') and event.button == 1:
            self.drawing_active = False
            if self.current_shape:
                self.current_shape.set_linestyle('-')
                self.current_shape.set_edgecolor('none')
                # self.current_shape = None # Uncomment to clear temporary shape after drawing
            self.draw_idle()
        # Smudge tool release
        elif self.mode == 'smudge' and event.button == 1:
            self.is_smudging = False
            if self.current_smudge_line:
                self.smudge_lines.append({
                    'line': self.current_smudge_line,
                    'radius': self.smudge_radius
                })
                self.current_smudge_line = None
            self.smudge_points = []

    def clear_unfinished_polygon(self):
        if self.current_polygon_line:
            self.current_polygon_line.remove()
            self.current_polygon_line = None

        for artist in self.axes.get_children():
            if isinstance(artist,
                          Line2D) and artist.get_color() == 'lightgray':
                if artist != self.current_polygon_line:
                    if len(artist.get_xdata()) == 2:
                        artist.remove()
                    elif len(artist.get_xdata()) == 1:
                        artist.remove()

        self.polygon_points = []

    def on_key_press(self, event):
        if event.key == 'shift':
            self.shift_is_pressed = True
            if self.drawing_active and self.mode == 'draw_circle' and self.current_shape:
                pass

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_pressed = False
            if self.drawing_active and self.mode == 'draw_circle' and self.current_shape:
                pass

    def _on_mouse_move(self, event):
        if self.im is None:
            return
        if event.inaxes:
            x_data = event.xdata
            y_data = event.ydata
            if 0 <= int(y_data) < self.im.shape[0] and \
               0 <= int(x_data) < self.im.shape[1]:
                pixel_value = self.im[int(y_data), int(x_data)]
            else:
                pixel_value = np.nan
            self.mouse_data_coords_changed.emit(x_data, y_data,
                                                float(pixel_value))

    def get_mask(self):
        if self.im is None:
            return None
        height, width = self.im.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        x_coords = np.arange(width)
        y_coords = np.arange(height)
        X, Y = np.meshgrid(x_coords, y_coords)
        points = np.vstack([X.ravel(), Y.ravel()]).T

        for artist in self.axes.patches:
            if isinstance(artist, Ellipse):
                center = artist.center
                width_ellipse = artist.width
                height_ellipse = artist.height
                angle = artist.angle * np.pi / 180

                x_rotated = (X - center[0]) * np.cos(angle) + (
                    Y - center[1]) * np.sin(angle)
                y_rotated = -(X - center[0]) * np.sin(angle) + (
                    Y - center[1]) * np.cos(angle)

                in_ellipse = ((x_rotated / (width_ellipse / 2))**2 +
                              (y_rotated / (height_ellipse / 2))**2 <= 1)
                mask |= in_ellipse.astype(np.uint8)

            elif isinstance(artist, Rectangle):
                xy = artist.get_xy()
                rect_width = artist.get_width()
                rect_height = artist.get_height()

                in_rect = ((X >= xy[0]) & (X <= xy[0] + rect_width) &
                           (Y >= xy[1]) & (Y <= xy[1] + rect_height))
                mask |= in_rect.astype(np.uint8)

            elif isinstance(artist, Polygon):
                polygon_points = artist.get_xy()
                polygon_path = mpath.Path(polygon_points)
                in_polygon = polygon_path.contains_points(points)
                mask |= in_polygon.reshape(height, width).astype(np.uint8)

        pil_mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(pil_mask)

        for smudge_data in self.smudge_lines:
            line = smudge_data['line']
            radius = smudge_data['radius']

            x_data, y_data = line.get_xdata(), line.get_ydata()

            display_radius = radius
            data_radius = display_radius * width / self.axes.bbox.width

            thickness = max(1, int(round(data_radius * 2)))
            int_radius = max(1, int(round(data_radius)))

            points_xy = [(x, y) for x, y in zip(x_data, y_data)
                         if np.isfinite(x) and np.isfinite(y)]

            if not points_xy:
                continue

            # step 1: draw connecting line segments
            if len(points_xy) > 1:
                draw.line(points_xy, fill=1, width=thickness)

            # step 2: draw filled circles at each vertex
            for point in points_xy:
                x, y = point
                # defined by the center and radius
                bbox = [
                    x - int_radius, y - int_radius, x + int_radius,
                    y + int_radius
                ]
                draw.ellipse(bbox, fill=1)

        mask = np.array(pil_mask)
        if self.mask_above_value is not None:
            mask[self.im > self.mask_above_value] = 1
        if self.mask_below_value is not None:
            mask[self.im < self.mask_below_value] = 1
            print(np.sum(mask))
        return mask

    def clear_all_smudges(self):
        for smudge_data in self.smudge_lines:
            if smudge_data['line'] in self.axes.lines:
                smudge_data['line'].remove()
        self.smudge_lines = []
        self.draw_idle()
