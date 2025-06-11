# Copyright 2025 Zichong Li, ETH Zurich

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np
import os
import sys
import torch

import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPen
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{normal_repr(self)} \n {self.shape}"
np.set_printoptions(edgeitems=3, linewidth=1000, threshold=100, formatter={"float": "{: 0.3f}".format}, suppress=True)
torch.set_printoptions(edgeitems=3, linewidth=1000, threshold=100)

SCALING = 0.6

GOAL_SCALE_Y = 22 / 60
GOAL_SCALE_X = 39 / 45

CIRCLE_SCALE = 75 / 450

PENALTY_SCALE_Y = 4 / 6
PENALTY_SCALE_X = 285 / 450


class TimeLine(QtCore.QObject):
    frameChanged = QtCore.pyqtSignal(int)

    def __init__(self, interval=80, loopCount=1, parent=None):
        super().__init__(parent)
        self._startFrame = 0
        self._endFrame = 0
        self._loopCount = loopCount
        self._timer = QtCore.QTimer(self, timeout=self.on_timeout)
        self._counter = 0
        self._loop_counter = 0
        self.setInterval(interval)

    def on_timeout(self):
        if self._startFrame <= self._counter < self._endFrame:
            self.frameChanged.emit(self._counter)
            self._counter += 1
        else:
            self._counter = 0
            self._loop_counter += 1

        if self._loopCount > 0:
            if self._loop_counter >= self.loopCount():
                self._timer.stop()

    def reset(self):
        self._counter = 0

    def setLoopCount(self, loopCount):
        self._loopCount = loopCount

    def loopCount(self):
        return self._loopCount

    interval = QtCore.pyqtProperty(int, fget=loopCount, fset=setLoopCount)

    def setInterval(self, interval):
        self._timer.setInterval(interval)

    def interval(self):
        return self._timer.interval()

    interval = QtCore.pyqtProperty(int, fget=interval, fset=setInterval)

    def setFrameRange(self, startFrame, endFrame):
        self._startFrame = startFrame
        self._endFrame = endFrame

    @QtCore.pyqtSlot()
    def start(self):
        self._counter = 0
        self._loop_counter = 0
        self._timer.start()


class GUIBox(QWidget):
    def __init__(self, minimum=1, maximum=2, parent=None):
        super().__init__(parent=parent)

        self.minimum = minimum
        self.maximum = maximum

        self.horizontalLayout = QHBoxLayout(self)
        self.verticalLayout = QVBoxLayout()

        self.spacerItemL = QSpacerItem(500, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.spacerItemM = QSpacerItem(60, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.spacerItemR = QSpacerItem(600, 0, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.verticalLayoutSpeed = QVBoxLayout()
        self.labelSpeed = QLabel("Replay Speed: 5x")
        self.sliderSpeed = QSlider(Qt.Vertical)
        self.sliderSpeed.setRange(1, 5)
        self.sliderSpeed.setValue(5)
        self.sliderSpeed.setSingleStep(1)
        self.sliderSpeed.setFixedHeight(150)
        self.sliderSpeed.setFixedWidth(50)
        self.verticalLayoutSpeed.addWidget(self.labelSpeed)
        self.verticalLayoutSpeed.addWidget(self.sliderSpeed)

        self.verticalLayoutR = QVBoxLayout()
        self.labelR = QLabel("Red Agents 1")
        self.sliderR = QSlider(Qt.Vertical)
        self.sliderR.setSingleStep(1)
        self.sliderR.setRange(minimum, maximum)
        self.sliderR.setFixedHeight(150)
        self.sliderR.setFixedWidth(50)
        self.verticalLayoutR.addWidget(self.labelR)
        self.verticalLayoutR.addWidget(self.sliderR)

        self.verticalLayoutB = QVBoxLayout()
        self.labelB = QLabel("Blue Agents 1")
        self.sliderB = QSlider(Qt.Vertical)
        self.sliderB.setRange(minimum, maximum)
        self.sliderB.setSingleStep(1)
        self.sliderB.setFixedHeight(150)
        self.sliderB.setFixedWidth(50)
        self.verticalLayoutB.addWidget(self.labelB)
        self.verticalLayoutB.addWidget(self.sliderB)

        self.sliderPlay = QSlider(Qt.Horizontal)
        self.sliderPlay.setMinimum(1)
        self.sliderPlay.setMaximum(10)
        self.sliderPlay.setValue(0)
        self.sliderPlay.setTickInterval(1)
        self.sliderPlay.setTickPosition(QSlider.TicksBelow)
        self.sliderPlay.setFixedHeight(25)
        self.sliderPlay.setFixedWidth(300)
        self.slider_label_play = QLabel("Log Steps (Move slider to the left to auto play)")
        self.verticalLayout.addWidget(self.slider_label_play)
        self.verticalLayout.addWidget(self.sliderPlay)

        self.sliderEnv = QSlider(Qt.Horizontal)
        self.sliderEnv.setMinimum(1)
        self.sliderEnv.setMaximum(10)
        self.sliderEnv.setValue(0)
        self.sliderEnv.setTickInterval(1)
        self.sliderEnv.setTickPosition(QSlider.TicksBelow)
        self.sliderEnv.setFixedHeight(25)
        self.sliderEnv.setFixedWidth(300)
        self.slider_label_env = QLabel("Env ID")
        self.verticalLayout.addWidget(self.slider_label_env)
        self.verticalLayout.addWidget(self.sliderEnv)

        self.sliderEpoch = QSlider(Qt.Horizontal)
        self.sliderEpoch.setMinimum(0)
        self.sliderEpoch.setValue(0)
        self.sliderEpoch.setTickInterval(1)
        self.sliderEpoch.setTickPosition(QSlider.TicksBelow)
        self.sliderEpoch.setFixedHeight(25)
        self.sliderEpoch.setFixedWidth(300)
        self.slider_label = QLabel("Epoch")
        self.button_add = QPushButton("Add")
        self.button_del = QPushButton("Delete")
        self.checkbox_auto_iterate = QCheckBox("Auto Iterate Over Highlights")

        self.verticalLayout_selector = QVBoxLayout()
        self.horizontal_buttons = QHBoxLayout()
        self.horizontal_buttons.addWidget(self.button_add)
        self.horizontal_buttons.addWidget(self.button_del)
        self.horizontal_buttons.addWidget(self.checkbox_auto_iterate)

        self.experiment_layout = QVBoxLayout()
        # self.experiment_layout.setAlignment(Qt.AlignTop)

        self.experiment_folder = []
        self.dropdown_experiment_folders = QComboBox()
        self.dropdown_experiment_folders.setFixedWidth(150)
        self.dropdown_experiment_folders.addItems(self.experiment_folder)
        experiment_folder_label = QLabel("Experiment Folder Selector")
        experiment_folder_label.setFixedHeight(25)
        self.experiment_layout.addWidget(experiment_folder_label)
        self.experiment_layout.addWidget(self.dropdown_experiment_folders)

        self.experiments = []
        self.dropdown_experiments = QComboBox()
        self.dropdown_experiments.setFixedWidth(300)
        self.dropdown_experiments.addItems(self.experiments)
        experiment_label = QLabel("Experiment Selector")
        experiment_label.setFixedHeight(25)
        self.experiment_layout.addWidget(experiment_label)
        self.experiment_layout.addWidget(self.dropdown_experiments)

        self.dropdown_highlights = QComboBox()
        self.items = []
        self.current_selector_id = 0
        self.dropdown_highlights.addItems(self.items)
        highlight_label = QLabel("Highlight Manager")
        highlight_label.setFixedHeight(25)
        self.verticalLayout_selector.addWidget(highlight_label)
        self.verticalLayout_selector.addWidget(self.dropdown_highlights)
        self.verticalLayout_selector.addLayout(self.horizontal_buttons)

        self.verticalLayout.addWidget(self.slider_label)
        self.verticalLayout.addWidget(self.sliderEpoch)

        self.horizontalLayout.addItem(self.spacerItemL)
        self.horizontalLayout.addLayout(self.experiment_layout)
        self.horizontalLayout.addLayout(self.verticalLayoutSpeed)
        self.horizontalLayout.addItem(self.spacerItemM)
        self.horizontalLayout.addLayout(self.verticalLayoutR)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.horizontalLayout.addLayout(self.verticalLayoutB)
        self.horizontalLayout.addLayout(self.verticalLayout_selector)
        self.horizontalLayout.addItem(self.spacerItemR)

        # self.layout.addWidget(self.slider_label_env)

        # self.layout.addWidget(self.slider_label)

    def update_experiment_folders(self, experiment_folders):
        self.dropdown_experiment_folders.clear()
        self.dropdown_experiment_folders.addItems(experiment_folders)

    def update_agent_number(self, blue_range=[1, 3], red_range=[1, 3]):
        self.sliderB.setRange(blue_range[0], blue_range[1])
        self.sliderR.setRange(red_range[0], red_range[1])


class TrajAnalyser(QMainWindow):
    def __init__(self, folder_path):
        super().__init__()
        self.setWindowTitle("Training Trajectories Analyser")
        self.folder_path = folder_path
        self.full_path_to_eval_traj = None

        self.experiments_folder_list = [folder for folder in os.listdir(folder_path) if (folder[0].isdigit())]
        self.experiments_folder_list.sort(key=lambda m: "{:0>10}".format(m.split("_")[0]))

        self.current_folder_id = len(self.experiments_folder_list) - 1
        self.current_folder_name = self.experiments_folder_list[self.current_folder_id]

        self.experiments = []

        self.load_experiment_folder()
        selected_experiment = self.experiments[len(self.experiments) - 1]
        self.full_path_to_eval_traj = os.path.join(
            folder_path, self.current_folder_name, selected_experiment, "eval_traj"
        )
        self.load_traj()

        self.field_length = 4.5 * SCALING
        self.field_width = 3 * SCALING

        self.team_config_env_id = None

        self.world_state = None

        self.setGeometry(100, 100, 800, 1200)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.gui_box = GUIBox()

        self.gui_box.dropdown_experiment_folders.addItems(self.experiments_folder_list)

        self.border_offset = 0.5
        self.field_length = 4.5 * SCALING
        self.field_width = 3 * SCALING
        self.goal_width = 1.2 * SCALING
        self.goal_depth = 0.1

        # self.resize(self.sizeHint())
        self.tolerance = 0.4
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setXRange(-self.field_length - self.tolerance, self.field_length + self.tolerance)
        self.plot_widget.setYRange(-self.field_width - self.tolerance, self.field_width + self.tolerance)
        self.plot_widget.setAspectLocked(True)

        self.plot_widget.setBackground(None)  # Set transparent background
        self.plot_widget.setBackground(QColor(Qt.darkGreen))

        self.layout.addWidget(self.plot_widget, 5)
        self.layout.addWidget(self.gui_box, 1)

        self.add_football_field()

        self.scatter = pg.ScatterPlotItem()
        self.scatter_direction = pg.ScatterPlotItem()
        self.scatter_live = pg.ScatterPlotItem()
        self.plot_widget.addItem(self.scatter)
        self.plot_widget.addItem(self.scatter_live)
        self.plot_widget.addItem(self.scatter_direction)

        self.curve_live = self.plot_widget.plot()

        self.gui_box.dropdown_experiment_folders.activated[str].connect(self.update_experiment_folders)
        self.gui_box.dropdown_experiments.activated[str].connect(self.update_experiments)
        self.gui_box.sliderR.valueChanged.connect(self.setLabelValueR)
        self.gui_box.sliderB.valueChanged.connect(self.setLabelValueB)
        self.gui_box.sliderEpoch.valueChanged.connect(self.update_checkpoint)
        self.gui_box.sliderEnv.valueChanged.connect(self.updateEnvID)

        self.gui_box.sliderPlay.valueChanged.connect(self.plot_scatter_live_maual)
        self.enable_auto_iterate_over_hightlights = False
        self.current_item = 0
        self.max_number_of_items = 0
        self.gui_box.dropdown_highlights.activated[str].connect(self.itemSelected)
        self.gui_box.button_add.clicked.connect(self.addItem)
        self.gui_box.button_del.clicked.connect(self.deleteItem)
        self.gui_box.checkbox_auto_iterate.stateChanged.connect(self.auto_iterate_over_hightlights)

        self._timeline = TimeLine(loopCount=0, interval=10)
        self._timeline.setFrameRange(0, self.max_log_steps)
        self.auto_play = True
        self._timeline.frameChanged.connect(self.plot_scatter_live_automatic)
        self._timeline.start()
        self.gui_box.sliderSpeed.valueChanged.connect(self.changeSpeed)

        self.background_dot_size = 5
        self.live_dot_size = 10
        self.look_ahead_steps = 20
        self.plot_time_step = 2

        self.current_checkpoint_key = self.traj_dict_key_sorted[self.gui_box.sliderEpoch.value()]
        self.current_env_id = self.gui_box.sliderEnv.value()
        self.num_red = self.gui_box.sliderR.value()
        self.num_blue = self.gui_box.sliderB.value()
        self.update_core_state()
        # Add football field background image
        self.move(1200, 0)

    def load_experiment_folder(self):
        folder_path = os.path.join(self.folder_path, self.current_folder_name)
        self.experiments = [folder for folder in os.listdir(folder_path)]
        self.experiments.sort()
        # selected_experiment = experiments[len(experiments) - 1]
        # self.load_traj(os.path.join(folder_path, selected_experiment, "eval_traj"))

    def load_traj(self):
        folder_path = self.full_path_to_eval_traj
        traj_files = [file for file in os.listdir(folder_path) if "eval_traj" in file]
        traj_dict = dict()
        for traj_file in traj_files:
            traj_dict[traj_file.split(".")[0]] = torch.load(os.path.join(folder_path, traj_file), weights_only=True)
        self.traj_dict = traj_dict
        team_config = next(iter(self.traj_dict.values()))["infos"]["team_config"]
        min_team_size = torch.min(team_config, dim=0)[0]
        max_team_size = torch.max(team_config, dim=0)[0]
        self.blue_team_size = [min_team_size[0].item(), max_team_size[0].item()]
        self.red_team_size = [min_team_size[1].item(), max_team_size[1].item()]
        self.max_num_agents = torch.max(max_team_size).item()

        self.traj_dict_key_sorted = list(traj_dict.keys())
        self.traj_dict_key_sorted.sort(key=lambda m: "{:0>10}".format(m.split("_")[2]))

        self.num_checkpoints = len(self.traj_dict_key_sorted)

        self.current_traj = self.traj_dict[self.traj_dict_key_sorted[0]]
        self.max_log_steps = self.current_traj["infos"]["log_steps"]
        self.episode_len = torch.clip(self.current_traj["infos"]["episode_len"], 0, self.max_log_steps)

    def save_selection(self):
        with open(self.full_path_to_eval_traj + "/selection.json", "w") as f:
            json.dump(self.gui_box.items, f)

    def load_selection(self):
        try:
            with open(self.full_path_to_eval_traj + "/selection.json") as f:
                loaded_list = json.load(f)
                self.gui_box.dropdown_highlights.clear()
                self.gui_box.items = loaded_list
                self.gui_box.dropdown_highlights.addItems(loaded_list)
                self.current_item = 0
                self.max_number_of_items = len(loaded_list)
        except FileNotFoundError:
            with open(self.full_path_to_eval_traj + "/selection.json", "w") as f:
                json.dump([], f)

    def changeSpeed(self, value):
        self._timeline.setInterval(50 // value)
        self.gui_box.labelSpeed.setText(f"Replay Speed: {value}x")

    def addItem(self, item):
        epoch_num = self.current_checkpoint_key.split("_")[2]
        item = f"Epoch:{epoch_num}_R:{self.num_red}_B:{self.num_blue}_Env:{self.current_env_id}"
        if item not in self.gui_box.items:
            self.gui_box.items.append(item)
            self.gui_box.dropdown_highlights.addItem(item)
            self.save_selection()

    def deleteItem(self):
        if len(self.gui_box.items) > 0 and (0 <= self.gui_box.current_selector_id < len(self.gui_box.items)):
            del self.gui_box.items[self.gui_box.current_selector_id]
            self.gui_box.dropdown_highlights.removeItem(self.gui_box.current_selector_id)
            self.save_selection()

    def auto_iterate_over_hightlights(self):
        self.enable_auto_iterate_over_hightlights = self.gui_box.checkbox_auto_iterate.isChecked()
        self.load_selection()

    def itemSelected(self, text):
        self.gui_box.current_selector_id = self.gui_box.items.index(text)

        current_checkpoint_id = 0
        for i, key in enumerate(self.traj_dict_key_sorted):
            if text.split("_")[0].split(":")[1] in key:
                current_checkpoint_id = i
                break
        self.gui_box.sliderEpoch.setValue(current_checkpoint_id)
        self.update_checkpoint(current_checkpoint_id)

        self.num_red = int(text.split("_")[1].split(":")[1])
        self.num_blue = int(text.split("_")[2].split(":")[1])
        self.gui_box.sliderR.setValue(self.num_red)
        self.gui_box.sliderB.setValue(self.num_blue)

        self.current_env_id = int(text.split("_")[3].split(":")[1])
        self.gui_box.sliderEnv.setValue(self.current_env_id)
        self.updateEnvID(self.current_env_id)

        self.gui_box.sliderPlay.setMaximum(self.max_log_steps)
        self.update_labels()
        self.update_core_state()

    def update_experiment_folders(self, text):
        experiment_folder_id = self.experiments_folder_list.index(text)
        self.current_folder_name = self.experiments_folder_list[experiment_folder_id]
        self.load_experiment_folder()
        self.gui_box.dropdown_experiments.clear()
        self.gui_box.dropdown_experiments.addItems(self.experiments)

    def update_experiments(self, text):
        experiment_id = self.experiments.index(text)
        selected_experiment = self.experiments[experiment_id]
        self.full_path_to_eval_traj = os.path.join(
            self.folder_path, self.current_folder_name, selected_experiment, "eval_traj"
        )
        self.load_traj()
        self.load_selection()

        self.reset_gui_sliders_and_states()

    def reset_gui_sliders_and_states(self):
        self.gui_box.sliderR.setValue(self.blue_team_size[0])
        self.gui_box.sliderB.setValue(self.red_team_size[0])
        self.gui_box.sliderEpoch.setValue(0)
        self.gui_box.sliderEnv.setValue(0)
        self.gui_box.sliderPlay.setValue(0)
        self.current_checkpoint_key = None
        self.current_env_id = None

        self.update_checkpoint(0)
        self.updateEnvID(0)

        self.update_labels()
        self.update_core_state()

    def setLabelValueR(self, value):
        self.num_red = value
        self.gui_box.labelR.setText(f"Red Agents {self.num_red:.4g}")
        self.update_core_state()

    def setLabelValueB(self, value):
        self.num_blue = value
        self.gui_box.labelB.setText(f"Blue Agents {self.num_blue:.4g}")
        self.update_core_state()

    def updateEnvID(self, value):
        self.current_env_id = value
        self.gui_box.slider_label_env.setText(f"Env : {self.current_env_id}")
        self.update_core_state()

    def update_checkpoint(self, value):
        self.current_checkpoint_key = self.traj_dict_key_sorted[value]
        self.gui_box.slider_label.setText(f"Epoch: {self.current_checkpoint_key}")
        self.update_core_state()

    def update_labels(self):
        self.gui_box.labelR.setText(f"Red Agents {self.num_red:.4g}")
        self.gui_box.labelB.setText(f"Blue Agents {self.num_blue:.4g}")
        self.gui_box.slider_label_env.setText(f"Env : {self.current_env_id}")
        self.gui_box.slider_label.setText(f"Epoch: {self.current_checkpoint_key}")

    def update_core_state(self):
        self.gui_box.sliderEpoch.setMaximum(self.num_checkpoints - 1)
        self.gui_box.sliderPlay.setMaximum(self.max_log_steps)

        self.gui_box.update_agent_number(self.blue_team_size, self.red_team_size)
        if self.current_checkpoint_key is None:
            self.current_checkpoint_key = self.traj_dict_key_sorted[self.gui_box.sliderEpoch.value()]
        if self.current_env_id is None:
            self.current_env_id = self.gui_box.sliderEnv.value()

        self.current_traj = self.traj_dict[self.current_checkpoint_key]
        self.episode_len = torch.clip(self.current_traj["infos"]["episode_len"], 0, self.max_log_steps)

        team_config = self.current_traj["infos"]["team_config"]
        if "play" in self.full_path_to_eval_traj:
            self.num_blue = team_config[0, 0]
            self.num_red = team_config[0, 1]
        self.team_config_env_id = torch.logical_and(
            team_config[:, 0] == self.num_blue, team_config[:, 1] == self.num_red
        )
        self.num_related_envs = len(self.team_config_env_id.nonzero())
        if self.num_related_envs > 0:
            self.gui_box.sliderEnv.setRange(0, self.num_related_envs - 1)
        else:
            self.gui_box.sliderEnv.setRange(0, 0)
        self.current_env_id = min(self.current_env_id, self.num_related_envs - 1)
        sub_episode_len = self.episode_len[self.team_config_env_id]
        self.world_state = self.current_traj["world_state_traj"][:, self.team_config_env_id, :][
            -sub_episode_len[self.current_env_id] :, self.current_env_id, :
        ]
        self._timeline.setFrameRange(0, sub_episode_len[self.current_env_id])
        self._timeline.reset()
        self.plot_scatter()

    def plot_scatter(self):
        world_state = self.world_state.cpu()
        red_pos_w = (
            world_state[:, 3 * self.max_num_agents : -2]
            .reshape(-1, self.max_num_agents, 3)[:, : self.num_red, :]
            .numpy()
        )
        blue_pos_w = (
            world_state[:, : 3 * self.max_num_agents].reshape(-1, self.max_num_agents, 3)[:, : self.num_blue, :].numpy()
        )
        ball_pos_w = world_state[:, -2:]

        num_red_agents = red_pos_w.shape[1]
        num_blue_agents = blue_pos_w.shape[1]

        data = []
        data_len = world_state.shape[0]
        plot_range = range(0, data_len, self.plot_time_step)
        for i in range(num_red_agents):
            data += [
                {
                    "pos": red_pos_w[j, i, :2],
                    "brush": pg.mkBrush(255, 0, 0),
                    "pen": pg.mkPen(None),
                    "size": self.background_dot_size,
                }
                for j in plot_range
            ]
        for i in range(num_blue_agents):
            data += [
                {
                    "pos": blue_pos_w[j, i, :2],
                    "brush": pg.mkBrush(0, 0, 255),
                    "pen": pg.mkPen(None),
                    "size": self.background_dot_size,
                }
                for j in plot_range
            ]

        data += [
            {
                "pos": ball_pos_w[j, :2],
                "brush": pg.mkBrush(255, 255, 255),
                "pen": pg.mkPen(None),
                "size": self.background_dot_size,
            }
            for j in plot_range
        ]
        self.scatter.setData(data)

    def plot_scatter_live_maual(self, current_time_step):
        if current_time_step != 1:
            self.auto_play = False
            self.plot_scatter_live(current_time_step)
        else:
            self.auto_play = True

    def plot_scatter_live_automatic(self, current_time_step):
        if not self.auto_play:
            return
        self.plot_scatter_live(current_time_step)

    def plot_scatter_live(self, current_time_step):
        def map_to_alpha(value):
            return min(
                int((self.look_ahead_steps - value) * 200 / (self.look_ahead_steps)) + 55,
                255,
            )

        world_state = self.world_state.cpu()
        red_pos_w = (
            world_state[:, 3 * self.max_num_agents : -2]
            .reshape(-1, self.max_num_agents, 3)[:, : self.num_red, :]
            .numpy()
        )
        blue_pos_w = (
            world_state[:, : 3 * self.max_num_agents].reshape(-1, self.max_num_agents, 3)[:, : self.num_blue, :].numpy()
        )
        ball_pos_w = world_state[:, -2:]

        num_red_agents = red_pos_w.shape[1]
        num_blue_agents = blue_pos_w.shape[1]

        data_live = []
        plot_range = range(0, min(int(self.look_ahead_steps), current_time_step), self.plot_time_step)
        for i in range(num_red_agents):
            data_live += [
                {
                    "pos": red_pos_w[max(0, current_time_step - j), i, :2],
                    "brush": pg.mkBrush(255, 0, 0, map_to_alpha(j)),
                    "pen": pg.mkPen(None),
                    "size": self.live_dot_size * 1.5,
                }
                for j in plot_range
            ] + [
                {
                    "pos": red_pos_w[current_time_step, i, :2],
                    "brush": pg.mkBrush(255, 0, 0, 200),
                    "pen": pg.mkPen(None),
                    "size": self.live_dot_size * 4,
                }
            ]

        for i in range(num_blue_agents):
            data_live += [
                {
                    "pos": blue_pos_w[max(0, current_time_step - j), i, :2],
                    "brush": pg.mkBrush(0, 0, 255, map_to_alpha(j)),
                    "pen": pg.mkPen(None),
                    "size": self.live_dot_size * 1.5,
                }
                for j in plot_range
            ] + [
                {
                    "pos": blue_pos_w[current_time_step, i, :2],
                    "brush": pg.mkBrush(0, 0, 255, 200),
                    "pen": pg.mkPen(None),
                    "size": self.live_dot_size * 4,
                }
            ]
        data_live += [
            {
                "pos": ball_pos_w[max(0, current_time_step - j), :2],
                "brush": pg.mkBrush(255, 255, 255, map_to_alpha(j)),
                "pen": pg.mkPen(None),
                "size": self.live_dot_size * 1.5,
            }
            for j in plot_range
        ] + [
            {
                "pos": ball_pos_w[current_time_step, :2],
                "brush": pg.mkBrush(255, 255, 255, 200),
                "pen": pg.mkPen(None),
                "size": self.live_dot_size * 4,
            }
        ]
        self.scatter_live.setData(data_live)

        agent_pose = np.concatenate(
            (
                blue_pos_w[current_time_step],
                red_pos_w[current_time_step],
            ),
            axis=0,
        )
        B = np.concatenate(
            (
                np.expand_dims(np.cos(agent_pose[:, 2]), axis=1),
                np.expand_dims(np.sin(agent_pose[:, 2]), axis=1),
            ),
            axis=1,
        )
        scatter_position = agent_pose[:, :2] + B * 0.06

        direction_data = []
        for i in range(scatter_position.shape[0]):
            direction_data += [
                {
                    "pos": scatter_position[i],
                    "brush": pg.mkBrush(255, 255, 255, 250),
                    "pen": pg.mkPen(None),
                    "size": 8,
                }
            ]

        self.scatter_direction.setData(direction_data)
        if (
            self.enable_auto_iterate_over_hightlights
            and current_time_step == self.episode_len[self.team_config_env_id][self.current_env_id] - 1
        ):
            self.current_item += 1
            self.itemSelected(self.gui_box.items[self.current_item])

    def add_football_field(self):
        line_width = 0.025
        # Add lines to represent field boundaries
        penW = QPen(Qt.white)
        penW.setWidthF(line_width)
        penR = QPen(Qt.red)
        penR.setWidthF(line_width * 3)
        penB = QPen(Qt.blue)
        penB.setWidthF(line_width * 3)

        linesB = [
            [
                (self.field_length + self.goal_depth, -self.goal_width),
                (self.field_length + self.goal_depth, self.goal_width),
            ],
            [
                (self.field_length + line_width, -self.goal_width),
                (self.field_length + self.goal_depth, -self.goal_width),
            ],
            [(self.field_length + line_width, self.goal_width), (self.field_length + self.goal_depth, self.goal_width)],
        ]
        linesR = [
            [
                (-self.field_length - self.goal_depth, -self.goal_width),
                (-self.field_length - self.goal_depth, self.goal_width),
            ],
            [
                (-self.field_length - line_width, -self.goal_width),
                (-self.field_length - self.goal_depth, -self.goal_width),
            ],
            [
                (-self.field_length - line_width, self.goal_width),
                (-self.field_length - self.goal_depth, self.goal_width),
            ],
        ]
        linesW = [
            [(self.field_length, -self.field_width), (self.field_length, self.field_width)],
            [(0, -self.field_width), (0, self.field_width)],
            [(-self.field_length, -self.field_width), (-self.field_length, self.field_width)],
            [(self.field_length, self.field_width), (-self.field_length, self.field_width)],
            [(self.field_length, -self.field_width), (-self.field_length, -self.field_width)],
            [
                (self.field_length * GOAL_SCALE_X, -self.field_width * GOAL_SCALE_Y),
                (self.field_length * GOAL_SCALE_X, self.field_width * GOAL_SCALE_Y),
            ],
            [
                (self.field_length, -self.field_width * GOAL_SCALE_Y),
                (self.field_length * GOAL_SCALE_X, -self.field_width * GOAL_SCALE_Y),
            ],
            [
                (self.field_length, self.field_width * GOAL_SCALE_Y),
                (self.field_length * GOAL_SCALE_X, self.field_width * GOAL_SCALE_Y),
            ],
            [
                (self.field_length * PENALTY_SCALE_X, -self.field_width * PENALTY_SCALE_Y),
                (self.field_length * PENALTY_SCALE_X, self.field_width * PENALTY_SCALE_Y),
            ],
            [
                (self.field_length, -self.field_width * PENALTY_SCALE_Y),
                (self.field_length * PENALTY_SCALE_X, -self.field_width * PENALTY_SCALE_Y),
            ],
            [
                (self.field_length, self.field_width * PENALTY_SCALE_Y),
                (self.field_length * PENALTY_SCALE_X, self.field_width * PENALTY_SCALE_Y),
            ],
            [
                (-self.field_length * GOAL_SCALE_X, -self.field_width * GOAL_SCALE_Y),
                (-self.field_length * GOAL_SCALE_X, self.field_width * GOAL_SCALE_Y),
            ],
            [
                (-self.field_length, -self.field_width * GOAL_SCALE_Y),
                (-self.field_length * GOAL_SCALE_X, -self.field_width * GOAL_SCALE_Y),
            ],
            [
                (-self.field_length, self.field_width * GOAL_SCALE_Y),
                (-self.field_length * GOAL_SCALE_X, self.field_width * GOAL_SCALE_Y),
            ],
            [
                (-self.field_length * PENALTY_SCALE_X, -self.field_width * PENALTY_SCALE_Y),
                (-self.field_length * PENALTY_SCALE_X, self.field_width * PENALTY_SCALE_Y),
            ],
            [
                (-self.field_length, -self.field_width * PENALTY_SCALE_Y),
                (-self.field_length * PENALTY_SCALE_X, -self.field_width * PENALTY_SCALE_Y),
            ],
            [
                (-self.field_length, self.field_width * PENALTY_SCALE_Y),
                (-self.field_length * PENALTY_SCALE_X, self.field_width * PENALTY_SCALE_Y),
            ],
        ]

        for line in linesB:
            self.plot_widget.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], pen=penB)
        for line in linesR:
            self.plot_widget.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], pen=penR)

        for line in linesW:
            self.plot_widget.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], pen=penW)

        radius = self.field_length * CIRCLE_SCALE

        circle = pg.QtGui.QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)  # x, y, width, height
        circle.setPen(penW)
        self.plot_widget.addItem(circle)


def main():
    wks_logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "wks_logs")
    window = TrajAnalyser(wks_logs_dir)
    window.showFullScreen()

    sys.exit(app.exec_())


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    main()
