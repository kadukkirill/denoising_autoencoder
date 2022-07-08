import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from tensorflow.keras.models import Sequential, load_model
from main_window import Ui_MainWindow  # main window
from sound_window import Ui_Form  # sound create window


class MyWin(QtWidgets.QMainWindow):

    def __init__(self):
        super(MyWin, self).__init__()
        self.main_window = Ui_MainWindow()
        self.main_window.setupUi(self)
        self.sound = Sound()  # second tab

        # files with sound
        self.data_pure = None
        self.data_noise = None
        # y-axis of sound
        self.y_pure = None
        self.y_noise = None
        # normalized y-axis
        self.y_noise_normal = None
        self.y_pure_normal = None
        # prepared data for autoencoder
        self.percent_training = None
        self.noise_input = None
        self.noise_input_test = None
        self.pure_input = None
        self.pure_input_test = None
        # autoencoder
        self.model = None

        # disable buttons
        self.main_window.pushButton_3.setEnabled(False)
        self.main_window.pushButton_4.setEnabled(False)
        self.main_window.pushButton_5.setEnabled(False)
        self.main_window.pushButton_6.setEnabled(False)

        # set default values
        self.main_window.lineEdit_dur.setText("1")
        self.main_window.lineEdit_sampling_rate.setText("2048")
        self.main_window.lineEdit_batch_size.setText("64")
        self.main_window.lineEdit_epochs.setText("3")
        self.main_window.lineEdit_test_split.setText("0.3")
        self.main_window.lineEdit_val_split.setText("0.2")
        self.main_window.lineEdit_kernel.setText("4")
        self.main_window.lineEdit_num_visualize.setText("3")
        self.main_window.lineEdit_pure_file.setText("waves")
        self.main_window.lineEdit_noise_file.setText("noise_waves")
        self.main_window.lineEdit_first_layer.setText("128")
        self.main_window.lineEdit_second_layer.setText("32")
        self.main_window.lineEdit_weights_file.setText("weights")

        # button press functions
        self.main_window.pushButton.clicked.connect(self.show_dialog)
        self.main_window.pushButton_2.clicked.connect(self.load_data)
        self.main_window.pushButton_3.clicked.connect(self.load_weights)
        self.main_window.pushButton_4.clicked.connect(self.train_ae)
        self.main_window.pushButton_5.clicked.connect(self.predict_data)
        self.main_window.pushButton_6.clicked.connect(self.save_weights)

    # load pure and noise files for AE
    def load_data(self):
        try:
            if not 0 < float(self.main_window.lineEdit_test_split.text()) <= 0.9:
                raise ValueError
            test_split = float(self.main_window.lineEdit_test_split.text())
            waves = self.main_window.lineEdit_noise_file.text()
            name_file_noise_waves = f"./waves/{waves}.npy"
            waves = self.main_window.lineEdit_pure_file.text()
            name_file_pure_waves = f"./waves/{waves}.npy"

            self.data_noise = np.load(name_file_noise_waves)
            self.y_noise = self.data_noise[:, 1]
            self.data_pure = np.load(name_file_pure_waves)
            self.y_pure = self.data_pure[:, 1]

            # normalize data
            self.y_noise_normal = []
            self.y_pure_normal = []
            for i in range(0, len(self.y_noise)):
                noise_sample = self.y_noise[i].copy()
                pure_sample = self.y_pure[i].copy()
                noise_sample = (noise_sample - np.min(noise_sample)) / \
                               (np.max(noise_sample) - np.min(noise_sample))
                pure_sample = (pure_sample - np.min(pure_sample)) / \
                              (np.max(pure_sample) - np.min(pure_sample))
                self.y_noise_normal.append(noise_sample)
                self.y_pure_normal.append(pure_sample)

            self.y_noise_normal = np.array(self.y_noise_normal)
            self.y_pure_normal = np.array(self.y_pure_normal)
            self.noise_input = self.y_noise_normal.reshape(
                (self.y_noise_normal.shape[0], self.y_noise_normal.shape[1], 1))
            self.pure_input = self.y_pure_normal.reshape((self.y_pure_normal.shape[0], self.y_pure_normal.shape[1], 1))

            self.percent_training = round((1 - test_split) * len(self.noise_input))

            self.noise_input, self.noise_input_test = self.noise_input[:self.percent_training], \
                                                      self.noise_input[self.percent_training:]
            self.pure_input, self.pure_input_test = self.pure_input[:self.percent_training], \
                                                    self.pure_input[self.percent_training:]

            self.main_window.pushButton_3.setEnabled(True)
            self.main_window.pushButton_4.setEnabled(True)
            self.main_window.label_error.setText("")
        except FileNotFoundError:
            self.main_window.label_error.setText("Files not found")
        except ValueError:
            self.main_window.label_error.setText("Input error")

    # show second tab
    def show_dialog(self):
        self.sound.show()

    def train_ae(self):
        try:
            self.main_window.label_error.setText("")
            if not 0 < float(self.main_window.lineEdit_val_split.text()) <= 0.9:
                raise ValueError
            if int(self.main_window.lineEdit_epochs.text()) <= 0:
                raise ValueError
            if int(self.main_window.lineEdit_second_layer.text()) > int(self.main_window.lineEdit_first_layer.text()):
                raise ValueError
            # reading data from lineEdit
            validation_split = float(self.main_window.lineEdit_val_split.text())
            duration = int(self.main_window.lineEdit_dur.text())
            sampling_rate = int(self.main_window.lineEdit_sampling_rate.text())
            input_shape = (duration * sampling_rate, 1)
            batch_size = int(self.main_window.lineEdit_batch_size.text())
            num_epochs = int(self.main_window.lineEdit_epochs.text())
            kernel_size = int(self.main_window.lineEdit_kernel.text())
            self.main_window.label_error.setText("")
            first_layer = int(self.main_window.lineEdit_first_layer.text())
            second_layer = int(self.main_window.lineEdit_second_layer.text())

            # creating model
            self.model = Sequential()
            self.model.add(
                Conv1D(first_layer, kernel_size=kernel_size, activation='relu',
                       kernel_initializer='he_uniform', input_shape=input_shape))
            self.model.add(Conv1D(second_layer, kernel_size=kernel_size,
                                  activation='relu', kernel_initializer='he_uniform'))
            self.model.add(Conv1DTranspose(second_layer, kernel_size=kernel_size,
                                           activation='relu', kernel_initializer='he_uniform'))
            self.model.add(Conv1DTranspose(first_layer, kernel_size=kernel_size,
                                           activation='relu', kernel_initializer='he_uniform'))
            self.model.add(Conv1D(1, kernel_size=kernel_size, activation='sigmoid', padding='same'))
            self.model.compile(optimizer='adam', loss='huber_loss')

            self.model.summary()
            # var of stack of ae from 1 to 3
            stack_ae = self.main_window.spinBox.value()

            self.model.fit(self.noise_input, self.pure_input, epochs=num_epochs,
                           batch_size=batch_size, validation_split=validation_split)

            # we take the output of the previous autoencoder and combine it
            # with the original noisy data, and we feed the
            # output of the previous autoencoder + a clean signal to the output
            if stack_ae >= 2:
                autoencoder_2_input = self.model.predict(self.noise_input)
                autoencoder_2_input = np.concatenate((autoencoder_2_input, self.noise_input))
                autoencoder_2_output = np.concatenate((self.pure_input, self.pure_input))
                self.model.fit(autoencoder_2_input, autoencoder_2_output, epochs=num_epochs,
                               batch_size=batch_size, validation_split=validation_split)
            if stack_ae == 3:
                autoencoder_3_input = self.model.predict(self.noise_input)
                autoencoder_3_input = np.concatenate((autoencoder_3_input, autoencoder_2_input))
                autoencoder_3_output = np.concatenate((autoencoder_2_output, self.pure_input))
                self.model.fit(autoencoder_3_input, autoencoder_3_output, epochs=num_epochs,
                               batch_size=batch_size, validation_split=validation_split)

            self.main_window.pushButton_5.setEnabled(True)
            self.main_window.pushButton_6.setEnabled(True)
        except ValueError:
            self.main_window.label_error.setText("Input error")

    # work check ae
    def predict_data(self):
        try:
            # Generate reconstructions
            num_reconstructions = int(self.main_window.lineEdit_num_visualize.text())
            samples = self.noise_input_test[:num_reconstructions]
            reconstructions = self.model.predict(samples)
            # Plot ae output
            j = 1
            for i in range(0, num_reconstructions):
                prediction_index = i + self.percent_training
                original = self.y_noise[prediction_index]
                pure = self.y_pure[prediction_index]
                reconstruction = np.array(reconstructions[i])
                reconstruction = reconstruction.reshape(2048, 1)
                plt.figure(3, figsize=(10, 5))
                plt.subplot(num_reconstructions, 3, j)
                plt.plot(pure[:100])
                plt.xlabel('Час', fontsize=12)
                plt.ylabel('Амплітуда', fontsize=12)
                plt.title("Pure")
                j += 1
                plt.subplot(num_reconstructions, 3, j)
                plt.plot(original[:100])
                plt.xlabel('Час', fontsize=12)
                plt.ylabel('Амплітуда', fontsize=12)
                plt.title("Noise")
                j += 1
                plt.subplot(num_reconstructions, 3, j)
                plt.plot(reconstruction[:100])
                plt.xlabel('Час', fontsize=12)
                plt.ylabel('Амплітуда', fontsize=12)
                plt.title('Denoised AE')
                j += 1
                # #TODO
                # corr_matrix = np.corrcoef(pure, reconstruction)
                # plt.text(0, 0, "corr coef: " + str(round(corr_matrix[0, 1], 3)), size='medium', weight="bold",
                #          backgroundcolor="0.7")
                plt.show()
                self.main_window.label_error.setText("")
        except ValueError:
            self.main_window.label_error.setText("Input correct num visualize")
        except IndexError:
            self.main_window.label_error.setText("Num visualize out of range ")

    # load weights of model from .h5 file
    def load_weights(self):
        try:
            weights = self.main_window.lineEdit_weights_file.text()
            self.main_window.lineEdit_weights_file.setText("")
            self.model = load_model(f"./weights/{weights}.h5")
            self.model.summary()
            self.main_window.pushButton_5.setEnabled(True)
            self.main_window.pushButton_6.setEnabled(False)
            self.main_window.label_error.setText("")
        except OSError:
            self.main_window.label_error.setText("Input correct weights name")

    # save model to .h5 file
    def save_weights(self):
        try:
            # do not save file without name
            if len(self.main_window.lineEdit_weights_file.text()) == 0:
                raise ValueError
            weights = self.main_window.lineEdit_weights_file.text()
            self.main_window.lineEdit_weights_file.setText("")
            self.model.save(f"./weights/{weights}.h5")
            self.main_window.label_error.setText("")
        except ValueError:
            self.main_window.label_error.setText("Not enough letters")
        except FileNotFoundError:
            os.mkdir("./weights")
            self.model.save(f"./weights/{weights}.h5")
            self.main_window.label_error.setText("")


class Sound(QtWidgets.QWidget):
    def __init__(self):
        super(Sound, self).__init__()
        self.sound_window = Ui_Form()
        self.sound_window.setupUi(self)

        # generated sound
        self.samples = None
        # noised sound
        self.noise_samples = None

        self.sound_window.pushButton_2.setEnabled(False)
        self.sound_window.pushButton_3.setEnabled(False)
        self.sound_window.pushButton_4.setEnabled(False)
        self.sound_window.pushButton_5.setEnabled(False)
        self.sound_window.pushButton_6.setEnabled(False)

        self.sound_window.lineEdit.setText("1")
        self.sound_window.lineEdit_2.setText("2048")
        self.sound_window.lineEdit_3.setText("10")
        self.sound_window.lineEdit_4.setText("3")
        self.sound_window.lineEdit_5.setText("0.6")
        self.sound_window.lineEdit_16.setText("waves")
        self.sound_window.lineEdit_17.setText("noise_waves")

        self.sound_window.pushButton.clicked.connect(self.create_waves)
        self.sound_window.pushButton_2.clicked.connect(self.show_waves)
        self.sound_window.pushButton_3.clicked.connect(self.saving_data)
        self.sound_window.pushButton_4.clicked.connect(self.add_noise)
        self.sound_window.pushButton_5.clicked.connect(self.show_noise_waves)
        self.sound_window.pushButton_6.clicked.connect(self.saving_noise_data)

    def create_waves(self):
        # create sin
        def gen_sine_wave(frequency, sample_rate, dur):
            x_value = np.linspace(0, dur, sample_rate * dur, endpoint=False)  # point generation
            y_value = np.sin((2 * np.pi) * frequency * x_value)
            return x_value, y_value

        try:
            duration = int(self.sound_window.lineEdit.text())
            sampling_rate = int(self.sound_window.lineEdit_2.text())
            num_samples = int(self.sound_window.lineEdit_3.text())
            if num_samples <= 0:
                raise ValueError
            # creation and saving sin
            self.samples = []
            for i in range(num_samples):
                freq = round(np.random.uniform(0.001, 100), 3)
                x_val, y_val = gen_sine_wave(freq, sampling_rate, duration)
                self.samples.append((x_val, y_val))
            self.samples = np.array(self.samples)
            # if checkBox is checked data will be complicated by adding a wave each samples
            if self.sound_window.checkBox.isChecked():
                # creation and saving sin
                second_waves = []
                for i in range(num_samples):
                    freq = round(np.random.uniform(0.001, 100), 3)
                    x_val, y_val = gen_sine_wave(freq, sampling_rate, duration)
                    second_waves.append((x_val, y_val))

                second_waves = np.array(second_waves)
                second_waves = second_waves[:, 1]
                new_samples = []

                x_second, y_second = self.samples[:, 0], self.samples[:, 1]

                # complicate data
                for i in range(len(x_second)):
                    temp_sine = y_second[i].copy()
                    temp_sine += second_waves[i]
                    new_samples.append([x_second[i], temp_sine])

                self.samples = np.array(new_samples)
            self.sound_window.pushButton_2.setEnabled(True)
            self.sound_window.pushButton_3.setEnabled(True)
            self.sound_window.pushButton_4.setEnabled(True)
            self.sound_window.label_21.setText("")
        except ValueError:
            self.sound_window.label_21.setText("Input error")

    def show_waves(self):
        try:
            # show samples
            num_visualize = int(self.sound_window.lineEdit_4.text())
            for i in range(0, num_visualize):
                x_val, y_val = self.samples[i]
                plt.figure(1, figsize=(8, 5))
                plt.subplot(1, num_visualize, i + 1)
                plt.plot(x_val[:100], y_val[:100])
                plt.xlabel('Час', fontsize=12)
                plt.ylabel('Амплітуда', fontsize=12)
                plt.show()
                self.sound_window.label_21.setText("")
        except ValueError:
            self.sound_window.label_21.setText("Input correct num visualize")
        except IndexError:
            self.sound_window.label_21.setText("Num visualize out of range ")

    def saving_data(self):
        # Save data to file
        try:
            if len(self.sound_window.lineEdit_16.text()) == 0:
                raise ValueError
            waves = self.sound_window.lineEdit_16.text()
            np.save(f"./waves/{waves}.npy", self.samples)
            self.sound_window.pushButton_3.setEnabled(False)
            self.sound_window.label_21.setText("")
        except ValueError:
            self.sound_window.label_21.setText("Not enough letters")
        except FileNotFoundError:
            os.mkdir("./waves")
            np.save(f"./waves/{waves}.npy", self.samples)
            self.sound_window.pushButton_3.setEnabled(False)
            self.sound_window.label_21.setText("")

    def add_noise(self):
        try:
            x_data, y_data = self.samples[:, 0], self.samples[:, 1]
            noise_coeff = float(self.sound_window.lineEdit_5.text())
            self.noise_samples = []
            for i in range(0, len(x_data)):
                temp_sine = np.array(y_data[i]).copy()
                noise = np.random.normal(0, 1, temp_sine.shape) * noise_coeff
                temp_sine += noise
                self.noise_samples.append([x_data[i], temp_sine])
            self.sound_window.pushButton_5.setEnabled(True)
            self.sound_window.pushButton_6.setEnabled(True)
            self.sound_window.label_21.setText("")
        except ValueError:
            self.sound_window.label_21.setText("Input error")

    def show_noise_waves(self):
        # show noise samples
        try:
            num_visualize = int(self.sound_window.lineEdit_4.text())
            for i in range(0, num_visualize):
                x_val, y_val = self.noise_samples[i]
                plt.figure(2, figsize=(8, 5))
                plt.subplot(1, num_visualize, i + 1)
                plt.plot(x_val[:100], y_val[:100])
                plt.xlabel('Час', fontsize=12)
                plt.ylabel('Амплітуда', fontsize=12)
                plt.show()
                self.sound_window.label_21.setText("")
        except ValueError:
            self.sound_window.label_21.setText("Input correct num visualize")
        except IndexError:
            self.sound_window.label_21.setText("Num visualize out of range ")

    def saving_noise_data(self):
        try:
            if len(self.sound_window.lineEdit_16.text()) == 0:
                raise ValueError
            waves = self.sound_window.lineEdit_17.text()
            np.save(f"./waves/{waves}.npy", self.noise_samples)
            self.sound_window.pushButton_6.setEnabled(False)
            self.sound_window.label_21.setText("")
        except ValueError:
            self.sound_window.label_21.setText("Not enough letters")
        except FileNotFoundError:
            os.mkdir("./waves")
            np.save(f"./waves/{waves}.npy", self.noise_samples)
            self.sound_window.pushButton_6.setEnabled(False)
            self.sound_window.label_21.setText("")


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    application = MyWin()
    application.show()

    sys.exit(app.exec())
