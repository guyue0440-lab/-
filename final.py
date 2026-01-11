import os
import sys
import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QLabel,
                             QMessageBox, QLineEdit, QGroupBox, QSlider)
from PyQt5.QtCore import Qt, QTimer

# --- 环境修复逻辑 ---
try:
    import PyQt5

    dirname = os.path.dirname(PyQt5.__file__)
    plugin_path = os.path.join(dirname, 'Qt5', 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
except Exception:
    pass

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class FinalAudioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音信号处理实验平台 - 功能增强版")
        self.setGeometry(100, 100, 1400, 900)

        self.fs = 44100
        self.audio_raw = None  # 原始备份
        self.current_signal = None  # 当前信号
        self.is_user_sliding = False
        self.volume = 1.0  # 默认音量

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_slider)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ================= 左侧：控制面板 =================
        ctrl_panel = QVBoxLayout()
        ctrl_panel.setSpacing(10)

        # 1. 信号采集与导出
        group_input = QGroupBox("1. 信号管理")
        l1 = QVBoxLayout()
        self.btn_open = self.create_btn("导入音频 (WAV)", "#E3F2FD")
        self.btn_record = self.create_btn("语音录制 (5s)", "#F1F8E9")
        self.btn_save = self.create_btn("导出/保存处理后的音频", "#E1F5FE")
        self.btn_reset = self.create_btn("恢复/重置原始音频", "#FFEBEE")
        l1.addWidget(self.btn_open)
        l1.addWidget(self.btn_record)
        l1.addWidget(self.btn_save)
        l1.addWidget(self.btn_reset)
        group_input.setLayout(l1)
        ctrl_panel.addWidget(group_input)

        # 2. 基础音量调节 (新增)
        group_vol = QGroupBox("2. 实时音量控制")
        l_vol = QVBoxLayout()
        self.vol_label = QLabel("当前音量: 100%")
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 200)  # 0% - 200%
        self.vol_slider.setValue(100)
        self.vol_slider.valueChanged.connect(self.action_volume_change)
        self.btn_mute = self.create_btn("一键静音", "#CFD8DC")
        l_vol.addWidget(self.vol_label)
        l_vol.addWidget(self.vol_slider)
        l_vol.addWidget(self.btn_mute)
        group_vol.setLayout(l_vol)
        ctrl_panel.addWidget(group_vol)

        # 3. 播放特效模块
        group_effect = QGroupBox("3. 播放特效控制")
        l2 = QVBoxLayout()
        self.btn_speed = self.create_btn("2倍速播放", "#FFF9C4")
        self.btn_reverse = self.create_btn("倒放音频处理", "#F3E5F5")
        l2.addWidget(self.btn_speed)
        l2.addWidget(self.btn_reverse)
        group_effect.setLayout(l2)
        ctrl_panel.addWidget(group_effect)

        # 4. 滤波器设计模块
        group_filter = QGroupBox("4. 巴特沃斯滤波器设计")
        l3 = QVBoxLayout()
        fc_lay = QHBoxLayout()
        fc_lay.addWidget(QLabel("截止频率(Hz):"))
        self.edit_fc = QLineEdit("2000")
        fc_lay.addWidget(self.edit_fc)
        l3.addLayout(fc_lay)
        self.btn_lp = self.create_btn("执行低通滤波", "#E8F5E9")
        self.btn_hp = self.create_btn("执行高通滤波", "#E8F5E9")
        l3.addWidget(self.btn_lp)
        l3.addWidget(self.btn_hp)
        group_filter.setLayout(l3)
        ctrl_panel.addWidget(group_filter)

        ctrl_panel.addStretch()
        main_layout.addLayout(ctrl_panel, 1)

        # ================= 右侧：展示与交互区 =================
        viz_panel = QVBoxLayout()

        self.status_label = QLabel("状态：请导入或录制音频")
        self.status_label.setStyleSheet("font-size: 11pt; font-weight: bold; color: #1565C0;")

        # 新增信息展示栏
        self.info_label = QLabel("时长: 0s | 采样率: 0Hz | 峰值: 0.0")
        self.info_label.setStyleSheet("color: #666; font-size: 9pt;")

        viz_panel.addWidget(self.status_label)
        viz_panel.addWidget(self.info_label)

        # 播放控制与可拖动进度条
        play_ctrl = QVBoxLayout()
        self.btn_play = QPushButton("▶ 播放当前处理后的音频")
        self.btn_play.setFixedHeight(50)
        self.btn_play.setStyleSheet("background-color: #2E7D32; color: white; font-weight: bold; border-radius: 5px;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)

        play_ctrl.addWidget(self.btn_play)
        play_ctrl.addWidget(self.slider)
        viz_panel.addLayout(play_ctrl)

        # 绘图区域
        self.fig, (self.ax_wave, self.ax_spec) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.tight_layout(pad=4.0)
        self.canvas = FigureCanvas(self.fig)
        viz_panel.addWidget(self.canvas)

        main_layout.addLayout(viz_panel, 4)

        # 事件绑定
        self.btn_open.clicked.connect(self.action_load)
        self.btn_record.clicked.connect(self.action_record)
        self.btn_save.clicked.connect(self.action_save_audio)
        self.btn_reset.clicked.connect(self.action_reset)
        self.btn_speed.clicked.connect(self.action_speed)
        self.btn_reverse.clicked.connect(self.action_reverse)
        self.btn_lp.clicked.connect(lambda: self.action_filter("low"))
        self.btn_hp.clicked.connect(lambda: self.action_filter("high"))
        self.btn_play.clicked.connect(self.action_play)
        self.btn_mute.clicked.connect(lambda: self.vol_slider.setValue(0))

        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)

    def create_btn(self, t, c):
        b = QPushButton(t)
        b.setStyleSheet(
            f"background-color: {c}; padding: 8px; border-radius: 4px; border: 1px solid #ccc; font-size: 9pt;")
        return b

    def update_info(self):
        """更新统计信息"""
        if self.current_signal is not None:
            duration = len(self.current_signal) / self.fs
            peak = np.max(np.abs(self.current_signal))
            self.info_label.setText(f"时长: {duration:.2f}s | 采样率: {self.fs}Hz | 峰值幅度: {peak:.2f}")

    def draw_plots(self, data, title):
        self.ax_wave.clear()
        self.ax_spec.clear()

        # 1. 时域波形
        time_x = np.linspace(0, len(data) / self.fs, len(data))
        self.ax_wave.plot(time_x, data, color='#1976D2', linewidth=0.5)
        self.ax_wave.set_title(f"{title} - 时域波形图", fontsize=10)
        self.ax_wave.set_xlabel("时间 (s)")
        self.ax_wave.set_ylabel("幅度")
        self.ax_wave.grid(True, alpha=0.3)

        # 2. 频谱转换 (FFT)
        n = len(data)
        freq_x = np.fft.fftfreq(n, 1 / self.fs)
        fft_y = np.abs(np.fft.fft(data))
        self.ax_spec.plot(freq_x[:n // 2], fft_y[:n // 2], color='#D32F2F', linewidth=0.5)
        self.ax_spec.set_title(f"{title} - 频谱转换图", fontsize=10)
        self.ax_spec.set_xlabel("频率 (Hz)")
        self.ax_spec.set_ylabel("强度")
        self.ax_spec.set_xlim(0, 8000)
        self.ax_spec.grid(True, alpha=0.3)

        self.canvas.draw()
        self.update_info()

    # --- 功能实现 ---

    def action_load(self):
        p, _ = QFileDialog.getOpenFileName(self, "选择音频", "", "音频 (*.wav)")
        if p:
            self.audio_raw, self.fs = sf.read(p)
            if len(self.audio_raw.shape) > 1: self.audio_raw = self.audio_raw[:, 0]
            self.current_signal = self.audio_raw.copy()
            self.draw_plots(self.current_signal, "导入原始信号")
            self.status_label.setText("状态：音频文件已导入")
            self.slider.setValue(0)

    def action_record(self):
        self.status_label.setText("状态：正在录制，请发言...")
        QApplication.processEvents()
        rec = sd.rec(int(5 * self.fs), samplerate=self.fs, channels=1)
        sd.wait()
        self.audio_raw = rec.flatten()
        self.current_signal = self.audio_raw.copy()
        self.draw_plots(self.current_signal, "录制原始信号")
        self.status_label.setText("状态：录音完成")
        self.slider.setValue(0)

    def action_reset(self):
        """重置音频"""
        if self.audio_raw is not None:
            self.current_signal = self.audio_raw.copy()
            self.draw_plots(self.current_signal, "信号已重置")
            self.status_label.setText("状态：已恢复原始波形")

    def action_save_audio(self):
        """导出WAV"""
        if self.current_signal is None: return
        p, _ = QFileDialog.getSaveFileName(self, "保存处理后的音频", "processed_audio.wav", "音频 (*.wav)")
        if p:
            sf.write(p, self.current_signal, self.fs)
            QMessageBox.information(self, "成功", "音频已成功保存到本地！")

    def action_volume_change(self):
        """实时调整音量百分比"""
        self.volume = self.vol_slider.value() / 100.0
        self.vol_label.setText(f"当前音量: {int(self.volume * 100)}%")

    def action_filter(self, ftype):
        if self.current_signal is None: return
        try:
            fc = float(self.edit_fc.text())
            nyq = 0.5 * self.fs
            b, a = signal.butter(5, fc / nyq, btype=ftype)
            self.current_signal = signal.filtfilt(b, a, self.current_signal)
            label = "低通" if ftype == "low" else "高通"
            self.draw_plots(self.current_signal, f"巴特沃斯{label}滤波")
            self.status_label.setText(f"状态：已完成{label}滤波处理")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"滤波器参数有误: {e}")

    def action_speed(self):
        if self.current_signal is None: return
        sd.stop()
        # 变速通过调整采样率实现播放
        sd.play(self.current_signal * self.volume, self.fs * 2)
        self.status_label.setText("状态：正在以2倍速播放音频")

    def action_reverse(self):
        if self.current_signal is None: return
        self.current_signal = self.current_signal[::-1]
        self.draw_plots(self.current_signal, "倒放信号")
        self.status_label.setText("状态：信号已反转")

    def action_play(self):
        if self.current_signal is None: return
        sd.stop()
        start_frame = int((self.slider.value() / 1000) * len(self.current_signal))
        # 播放时乘以当前音量系数
        sd.play(self.current_signal[start_frame:] * self.volume, self.fs)
        self.timer.start(100)

    def on_slider_pressed(self):
        self.is_user_sliding = True
        self.timer.stop()

    def on_slider_released(self):
        self.is_user_sliding = False
        self.action_play()

    def update_slider(self):
        if self.is_user_sliding: return
        if not sd.get_stream().active:
            self.timer.stop()
            return
        val = self.slider.value() + 2
        if val <= 1000:
            self.slider.setValue(val)
        else:
            self.timer.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用统一的现代界面风格
    win = FinalAudioApp()
    win.show()
    sys.exit(app.exec_())