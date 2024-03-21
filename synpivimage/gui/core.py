import pathlib
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal
from scipy.fft import rfft2, irfft2, fftshift

import synpivimage
import synpivimage as spi
from lib.corr import CorrelationPlane
from lib.plotting import gauss3ptfit
from src.main import Ui_MainWindow

__this_dir__ = pathlib.Path(__file__).parent
INIT_DIR = __this_dir__


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def get_window(window_function, shape):
    if window_function in ('gaussian50', 'gaussian25'):
        ny, nx = shape
        if window_function == 'gaussian50':
            sy = ny / 2
            sx = nx / 2
        else:
            sy = ny / 4
            sx = nx / 4
    elif window_function == 'tophat':
        raise NotImplementedError()
    else:
        raise KeyError(f'Unknown window function: {window_function}')
    return np.sqrt(np.outer(signal.gaussian(ny, sy), signal.gaussian(nx, sx)))


def generate_correlation(imgA, imgB, window_function=None):
    if window_function != 'uniform':
        try:
            win = get_window(window_function, imgA.shape)
            f2a = np.conj(rfft2(win * imgA))
            f2b = rfft2(win * imgB)
        except (KeyError, NotImplementedError) as e:
            warnings.warn(f'Unknown window function: {window_function}')
            f2a = np.conj(rfft2(imgA))
            f2b = rfft2(imgB)
    else:
        f2a = np.conj(rfft2(imgA))
        f2b = rfft2(imgB)
    return fftshift(irfft2(f2a * f2b).real, axes=(-2, -1))


class Ui(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, root_dir):
        self.curr_img_index = 0

        super(Ui, self).__init__()
        self.setupUi(self)

        self.setWindowTitle('Synthetic PIV Image Generator')

        self.nx.setMinimum(4)
        self.nx.setValue(16)
        self.ny.setMinimum(4)
        self.ny.setValue(16)
        self.nx.setMaximum(1000)
        self.ny.setMaximum(1000)
        self.particle_density.setMinimum(0.000001)
        self.particle_density.setMaximum(1)
        self.particle_density.setValue(0.1)

        self.particle_image_diameter.setMinimum(0.5)
        self.particle_image_diameter.setMaximum(10)
        self.particle_image_diameter.setValue(2.5)

        self.laser_width.setMinimum(0)
        self.laser_width.setValue(2)
        self.laser_shape_factor.setMinimum(1)
        self.laser_shape_factor.setMaximum(10 ** 6)
        self.laser_shape_factor.setValue(10000)
        self.laser_shape_factor.setToolTip('gaussian: 2, top hat: 10000')

        self.particle_count.setMinimum(1)
        self.particle_count.setMaximum(2 ** 16)
        self.particle_count.setValue(1000)

        self.dx.setMinimum(-100)
        self.dx.setMaximum(100)

        self.dy.setMinimum(-100)
        self.dy.setMaximum(100)

        self.dx.setValue(0.6)
        self.dy.setValue(0.3)

        self.dz.setMinimum(-100)
        self.dz.setMaximum(100)
        self.dz.setValue(0.0)

        self.n_imgs.setMinimum(1)
        self.n_imgs.setValue(1)
        self.n_imgs.setMaximum(10 ** 6)

        self.baseline.setMinimum(0)
        self.baseline.setMaximum(2 ** 16)
        self.baseline.setValue(50)

        self.darknoise.setMinimum(0)
        self.darknoise.setMaximum(2 ** 16 / 2)
        self.darknoise.setValue(10)

        self.bit_depth.setMinimum(4)
        self.bit_depth.setValue(16)
        self.bit_depth.setMaximum(32)
        n_imgs = self.n_imgs.value()
        nx = self.nx.value()
        ny = self.ny.value()
        self.imgsA = np.empty(shape=(n_imgs, ny, nx))
        self.imgsB = np.empty(shape=(n_imgs, ny, nx))
        self.correlations = np.empty(shape=(n_imgs, ny, nx))
        self.particle_dataA = [{}] * n_imgs
        self.particle_dataB = [{}] * n_imgs

        plotting_layout1 = QHBoxLayout(self.plotwidget1)
        plotting_layout2 = QHBoxLayout(self.plotwidget2)

        dummy_arr = np.random.random((self.nx.value(), self.ny.value()))

        self.figures = []
        self.axes = []
        self.canvas = []
        self.ims = []
        self.cax = []
        for i in range(3):
            figure, ax = plt.subplots(tight_layout=True)
            self.figures.append(figure)
            self.axes.append(ax)
            self.canvas.append(FigureCanvas(figure))
            plotting_layout1.addWidget(self.canvas[-1])

            im = ax.imshow(dummy_arr, cmap='gray')
            divider = make_axes_locatable(ax)
            self.cax.append(divider.append_axes("right", size="5%", pad=0.05))
            _ = plt.colorbar(im, cax=self.cax[-1])

            self.ims.append(im)

        figure, ax = plt.subplots(2, 1, sharex=True, tight_layout=True)
        self.figures.append(figure)
        self.axes.append(ax)
        self.canvas.append(FigureCanvas(figure))
        plotting_layout2.addWidget(self.canvas[-1])
        ax[0].plot([1, 2, 3])
        ax[1].plot([1, 2, 3])

        for i in range(2):
            figure, ax = plt.subplots(tight_layout=True)
            self.figures.append(figure)
            self.axes.append(ax)
            self.canvas.append(FigureCanvas(figure))
            plotting_layout2.addWidget(self.canvas[-1])
            ax.plot([1, 2, 3])

        self.update()

        self._root_dir = root_dir

        self.btn_update.clicked.connect(self.update)
        # self.particle_number.valueChanged.connect(self.update)
        # self.particle_number.editingFinished.connect(self.update)
        # self.apply_gauss_window.clicked.connect(self.update_with_existing_images)

        self.show()

    def keyPressEvent(self, event):
        # This function will be called when any key is pressed
        if event.key() == Qt.Key_F5:
            # Handle F5 key press
            self.update()
        elif event.key() == Qt.Key_F6:
            self.update_with_existing_images()
        elif event.key() == Qt.Key_F8:
            # Handle F6 key press
            if self.curr_img_index == 0:
                self.curr_img_index = self.imgsA.shape[0] - 1
            else:
                self.curr_img_index -= 1
            self.update_plot()
        elif event.key() == Qt.Key_F9:
            # Handle F6 key press
            if self.curr_img_index == self.imgsA.shape[0] - 1:
                self.curr_img_index = 0
            else:
                self.curr_img_index += 1
            self.update_plot()

    def get_camera(self):
        cam = synpivimage.Camera(
            nx=self.nx.value(),
            ny=self.ny.value(),
            bit_depth=self.bit_depth.value(),
            dark_noise=self.darknoise.value(),
            baseline_noise=self.baseline.value(),
            shot_noise=self.shotnoise.isChecked(),
            fill_ratio_x=1.0,
            fill_ratio_y=1.0,
            particle_image_diameter=self.particle_image_diameter.value(),
            qe=1,
            sensitivity=1.,
        )
        return cam

    def get_laser(self):
        laser = synpivimage.Laser(
            shape_factor=self.laser_shape_factor.value(),
            width=self.laser_width.value()
        )
        return laser

    def generate_images(self, take_existing_particles=False):
        cam = self.get_camera()
        laser = self.get_laser()

        n_imgs = self.n_imgs.value()
        imgs_shape = (n_imgs, cam.ny, cam.nx)
        if self.imgsA.shape != imgs_shape:
            print('reallocate image arrays')
            self.imgsA = np.empty(shape=imgs_shape)
            self.imgsB = np.empty(shape=imgs_shape)
            self.correlations = np.empty(shape=imgs_shape)
            self.particle_dataA = [{}] * n_imgs
            self.particle_dataB = [{}] * n_imgs

        dx = self.dx.value()
        dy = self.dy.value()
        dz = self.dz.value()

        if dx > 0:
            dx_max = [0, dx]
        else:
            dx_max = [dx, 0]

        if dy > 0:
            dy_max = [0, dy]
        else:
            dy_max = [dy, 0]

        if dz > 0:
            dz_max = [0, dz]
        else:
            dz_max = [dz, 0]

        for i in range(n_imgs):
            if take_existing_particles:
                particle_dataA = self.particle_dataA[i]
            else:
                particle_dataA = synpivimage.particles.Particles.generate(
                    dx_max=dx_max, dy_max=dy_max, dz_max=dz_max,
                    camera=cam,
                    laser=laser,
                    ppp=self.particle_density.value(),
                )

            imgA, partA = synpivimage.take_image(
                camera=cam,
                laser=laser,
                particles=particle_dataA,
                particle_peak_count=self.particle_count.value(),
            )

            if take_existing_particles:
                particle_dataB = self.particle_dataA[i]
            else:
                particle_dataB = particle_dataA.displace(dx=self.dx.value(), dy=self.dy.value(), dz=self.dz.value())

            imgB, partB = synpivimage.take_image(
                camera=cam,
                laser=laser,
                particles=particle_dataB,
                particle_peak_count=self.particle_count.value(),
            )

            self.imgsA[i, ...] = imgA
            self.imgsB[i, ...] = imgB

            self.particle_dataA[i] = partA
            self.particle_dataB[i] = partB

    def update_with_existing_images(self):
        # compute correlations:
        # if the particle size definition has changed, the images need to be regenerated
        if self.particle_size_definition.currentText() != self.current_config.particle_size_definition:
            self.current_config = self.get_config()
            self.generate_images(take_existing_particles=True)
        for i in range(self.imgsA.shape[0]):
            self.correlations[i, ...] = generate_correlation(
                self.imgsA[i, ...],
                self.imgsB[i, ...],
                self.windowing.currentText()
            )
        # plot images
        self.update_plot()

    def update(self):
        self.curr_img_index = 0
        # generate images
        self.generate_images()
        # compute correlations:
        for i in range(self.imgsA.shape[0]):
            self.correlations[i, ...] = generate_correlation(
                self.imgsA[i, ...],
                self.imgsB[i, ...],
                self.windowing.currentText()
            )
        # plot images
        self.update_plot()

    def update_plot(self):
        self._plot_imgA()
        self._plot_imgB()
        self._plot_correlation()

    def _plot_imgA(self):
        # plot imgA to self.plot11 widget
        self.axes[0].cla()
        if self.windowing.currentText() != 'uniform':
            win = get_window(self.windowing.currentText(), self.imgsA[self.curr_img_index].shape)
            img = win * self.imgsA[self.curr_img_index]
        else:
            img = self.imgsA[self.curr_img_index]
        im = self.axes[0].imshow(normalize(img), cmap='gray')
        plt.colorbar(im, cax=self.cax[0])
        self.canvas[0].draw()

    def _plot_imgB(self):
        # plot_img(self.imgB, self.axes[1])
        self.axes[1].cla()
        if self.windowing.currentText() != 'uniform':
            win = get_window(self.windowing.currentText(), self.imgsB[self.curr_img_index].shape)
            img = win * self.imgsB[self.curr_img_index]
        else:
            img = self.imgsB[self.curr_img_index]
        im = self.axes[1].imshow(normalize(img), cmap='gray')
        plt.colorbar(im, cax=self.cax[1])
        self.canvas[1].draw()

    def _plot_correlation(self):
        self.axes[5].cla()
        self.axes[2].cla()
        im = self.axes[2].imshow(normalize(self.correlations[self.curr_img_index]), cmap='gray')
        plt.colorbar(im, cax=self.cax[2])
        self.canvas[2].draw()
        corr = CorrelationPlane(data=self.correlations[self.curr_img_index],
                                fit=self.peak_find.currentText())
        corr.data[corr.j, :].plot(ax=self.axes[5], color='r')
        corr.data[:, corr.i].plot(ax=self.axes[5], color='b')
        ny, nx = corr.data.shape

        peak_area = corr.data[corr.j - 1:corr.j + 2, corr.i - 1:corr.i + 2]

        peak_area[1, :].plot.scatter(ax=self.axes[5], color='r')
        self.axes[5].scatter(corr.i, np.max(peak_area.data), color='r', marker='^')

        peak_area[:, 1].plot.scatter(ax=self.axes[5], color='b')
        self.axes[5].scatter(corr.j, np.max(peak_area.data), color='b', marker='^')

        self.axes[5].vlines(corr.i_sub, 0, 1, color='r')
        self.axes[5].vlines(corr.j_sub, 0, 1, color='b')

        self.axes[2].vlines(corr.i_sub, 0, nx - 1, color='r')
        self.axes[2].hlines(corr.j_sub, 0, ny - 1, color='b')

        self.axes[2].scatter(corr.i,
                             corr.j,
                             marker='+',
                             color='r')

        nx = self.nx.value()
        ny = self.ny.value()

        try:
            g1 = gauss3ptfit(peak_area[1, :])
            _x = np.linspace(0, nx, nx * 4)
            self.axes[5].plot(_x, g1(_x - corr.i), color='r', linestyle='--')
        except RuntimeError as e:
            print(e)
        try:
            g2 = gauss3ptfit(peak_area[:, 1])

            _y = np.linspace(0, ny, ny * 4)
            self.axes[5].plot(_y, g2(_y - corr.j), color='b', linestyle='--')
        except RuntimeError as e:
            print(e)
        _min = min(corr.i, corr.j) - 5
        _max = max(corr.i, corr.j) + 5
        self.axes[5].set_xlim(_min, _max)
        self.canvas[5].draw()

        # canvas 3: plot scatter of displacements:
        for ax in self.axes[3]:
            ax.cla()

        self.axes[4].cla()

        estimated_dx = []  # np.empty(shape=(self.imgsA.shape[0]))
        estimated_dy = []  # np.empty(shape=(self.imgsA.shape[0]))
        for i in range(self.imgsA.shape[0]):
            try:
                corr = CorrelationPlane(self.correlations[i, ...],
                                        fit=self.peak_find.currentText())
            except ValueError as e:
                print(e)
                continue
            # self.axes[4].scatter(corr.j + corr.highest_peak.j_sub, corr.i + corr.highest_peak.i_sub, color='k',
            #                      marker='+')

            dx = corr.i_sub - nx / 2  # - corr.i
            dy = corr.j_sub - ny / 2  # - corr.j
            estimated_dx.append(dx)
            estimated_dy.append(dy)

            # sub_dx = dx - round(dx)
            # sub_dy = dy - round(dy)
        estimated_dx = np.array(estimated_dx)
        estimated_dy = np.array(estimated_dy)
        # sub_pixel_dx = estimated_dx
        # sub_pixel_dy = estimated_dy
        self.axes[4].scatter(estimated_dx, estimated_dy, color='r', marker='o', alpha=0.5)
        if len(estimated_dx) > 1:
            self.axes[4].scatter(estimated_dx[self.curr_img_index], estimated_dy[self.curr_img_index], color='k',
                                 marker='o')

        true_dx, true_dy = self.dx.value(), self.dy.value()
        self.axes[4].scatter(true_dx, true_dy, color='k', marker='+')
        self.axes[4].set_xlim(true_dx - 1, true_dx + 1)
        self.axes[4].set_ylim(true_dy - 1, true_dy + 1)

        # compute RMS values
        rms_x = np.sqrt(np.sum((estimated_dx - true_dx) ** 2) / (max(1, len(estimated_dx) - 1)))
        rms_y = np.sqrt(np.sum((estimated_dy - true_dy) ** 2) / (max(1, len(estimated_dy) - 1)))
        rms = np.sqrt(rms_x ** 2 + rms_y ** 2)
        self.axes[4].text(true_dx + 0.25, true_dy + 0.25, f'{rms:.3f}')
        self.axes[4].text(true_dx + 0.25, true_dy + 0.45, f'x: {rms_x:.3f}')
        self.axes[4].text(true_dx + 0.25, true_dy + 0.65, f'y: {rms_y:.3f}')

        #
        # xlims = self.axes[4].get_xlim()
        # ylims = self.axes[4].get_ylim()
        # self.axes[4].hlines(xlims[0], xlims[1], self.dx.value() - self.nx.value() / 2, color='k', linestyle='--')
        # self.axes[4].vlines(self.dy.value() - self.ny.value() / 2, ylims[0], ylims[1], color='k', linestyle='--')
        # self.canvas[3].draw()
        # self.canvas[4].draw()
        if False:
            z = np.linspace(-2 * self.laser_width.value(), 2 * self.laser_width.value(), 1000)
            laser_intensity = particle_intensity(z=z,
                                                 beam_width=self.laser_width.value(),  # laser beam width
                                                 s=self.laser_shape_factor.value(),  # shape factor
                                                 )
            self.axes[3][0].plot(z, laser_intensity)
            # self.axes[4][0].plot(z, laser_intensity)

            self.axes[3][1].scatter(self.particle_dataA[self.curr_img_index].z, self.particle_dataA[self.curr_img_index].y,
                                    color='b', alpha=0.5, s=10, marker='o')
            # draw vlines for laser
            self.axes[3][1].vlines(-self.laser_width.value() / 2, 0, self.nx.value(), color='k', linestyle='--')
            self.axes[3][1].vlines(self.laser_width.value() / 2, 0, self.nx.value(), color='k', linestyle='--')

            self.axes[3][1].scatter(self.particle_dataB[self.curr_img_index].z, self.particle_dataB[self.curr_img_index].y,
                                    color='r', alpha=0.5, s=10, marker='o')
            self.axes[3][1].vlines(-self.laser_width.value() / 2, 0, self.nx.value(), color='k', linestyle='--')
            self.axes[3][1].vlines(self.laser_width.value() / 2, 0, self.nx.value(), color='k', linestyle='--')

        self.canvas[2].draw()
        self.canvas[3].draw()
        self.canvas[4].draw()


def start(*args, wd=None, console: bool = True):
    """call the gui"""
    print('Preparing piv2hdf gui ...')
    if wd is None:
        root_dir = pathlib.Path.cwd()
    else:
        root_dir = pathlib.Path(wd)

    if not root_dir.exists():
        print(f'cannot start gui on that path: {root_dir}')

    if console:
        print('Initializing gui from console...')
        app = QtWidgets.QApplication([*args, ])
    else:
        if len(args[0]) == 2:
            root_dir = pathlib.Path(args[0][0]).parent
            root_dir = root_dir.joinpath(args[0][1])
        else:
            root_dir = INIT_DIR
        print('Initializing gui from python script...')
        app = QtWidgets.QApplication(*args, )

    _ = Ui(root_dir)

    print('Starting gui ...')
    app.exec_()
    print('Closed gui ...')


if __name__ == "__main__":
    start(sys.argv)
