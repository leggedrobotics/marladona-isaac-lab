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

RESOLUTION = 80
NORMALIZE_EVALUATOR = False

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib import collections as mc
from matplotlib import transforms

DEFAULT_FIGURE_SIZE = (15.9, 13)
GOAL_SCALE_Y = 22 / 60
GOAL_SCALE_X = 39 / 45

CIRCLE_SCALE = 75 / 450

PENALTY_SCALE_Y = 4 / 6
PENALTY_SCALE_X = 285 / 450
SCALING = 0.6


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == "TkAgg":
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == "WXAgg":
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


class ValuePlotter:
    def __init__(self, num_agents_per_team, num_of_players_to_visualize, plot_value_func_list):
        self.num_agents_per_team = num_agents_per_team
        self.num_players_to_visualize = min(num_of_players_to_visualize, num_agents_per_team)

        self.num_value_funcs = len(plot_value_func_list)
        self.plot_value_func_list = plot_value_func_list

        self.default_sub_plot_title = ["Agent Position", "Ball Position"]
        self.border_offset = 0.5
        self.field_length = 4.5 * SCALING
        self.field_width = 3 * SCALING
        self.goal_width = 1.2 * SCALING
        self.goal_depth = 0.1
        if NORMALIZE_EVALUATOR:
            y, x = np.meshgrid(np.linspace(-1, 1, RESOLUTION), np.linspace(-1, 1, RESOLUTION))
        else:
            y, x = np.meshgrid(
                np.linspace(-(3 + self.border_offset) * SCALING, (3 + self.border_offset) * SCALING, RESOLUTION),
                np.linspace(-(4.5 + self.border_offset) * SCALING, (4.5 + self.border_offset) * SCALING, RESOLUTION),
            )
        self.x = x
        self.y = y
        self.z = x * 0.0 + 0.5

        self.pcolormeshes = []
        self.scatters = []
        self.scatterBs = []
        self.scatterRs = []
        self.scatterOthers = []
        self.lines = []
        self.fieldcircles = []
        self.fieldLines = []
        # self.axbackgrounds = []
        self.last_time_stamp = None

        self.default_plot_kwargs = {
            "path_effects": [
                pe.Stroke(
                    linewidth=3.8,
                    foreground="g",
                ),
                pe.Normal(),
            ],
        }

    def init_plot(self):
        self.figure = plt.figure(figsize=DEFAULT_FIGURE_SIZE)
        self.sub_figures = self.figure.subfigures(nrows=self.num_value_funcs, ncols=1)
        plt.subplots_adjust(bottom=0.1, top=0.85)
        if not isinstance(self.sub_figures, np.ndarray):
            self.sub_figures = [self.sub_figures]
        self.axs_list = []
        for row, sub_fig in enumerate(self.sub_figures):
            sub_fig.suptitle(self.default_sub_plot_title[row], fontsize=30)
            axs = sub_fig.subplots(nrows=1, ncols=self.num_players_to_visualize)
            if not isinstance(axs, np.ndarray):
                axs = [axs]
            self.axs_list.append(axs)
            for ax in axs:
                ax.set_aspect("equal")
        move_figure(self.figure, 1150, 0)

        self.figure.canvas.draw()

    def reset_plot(self, x, y, color_bar=True):
        self.pcolormeshes.clear()
        self.scatters.clear()
        self.scatterBs.clear()
        self.scatterRs.clear()
        self.scatterOthers.clear()
        self.lines.clear()
        self.fieldcircles.clear()
        self.fieldLines.clear()

        for j in range(self.num_value_funcs):
            for i in range(self.num_players_to_visualize):
                ax = self.axs_list[j][i]
                ax.set_title("For Blue Agent " + str(i + 1), fontsize=20)
                ax.tick_params(axis="both", which="major", labelsize=20)

                rot = transforms.Affine2D().rotate_deg(90) + ax.transData
                pcolormesh = ax.pcolormesh(x, y, self.z, cmap="RdBu", vmin=0, vmax=1, transform=rot)  # inferno or RdBu
                scatter = ax.scatter([], [], color="w", s=80, edgecolor="k", transform=rot)
                scatterOther = ax.scatter([], [], color="k", s=500, edgecolor="w", transform=rot)
                scatterR = ax.scatter([], [], color="r", s=150, edgecolor="w", transform=rot)
                scatterB = ax.scatter([], [], color="b", s=150, edgecolor="w", transform=rot)
                lineCollection = ax.add_collection(
                    mc.LineCollection([], linewidths=3, colors="w", linestyles="solid", transform=rot)
                )
                fieldcircle = ax.add_patch(
                    plt.Circle(
                        (0.0, 0.0),
                        self.field_length * CIRCLE_SCALE,
                        color="w",
                        fill=False,
                        linewidth=3,
                        **self.default_plot_kwargs
                    )
                )
                fieldLineCollection = ax.add_collection(
                    mc.LineCollection(
                        [
                            [
                                (self.field_length + self.goal_depth, -self.goal_width),
                                (self.field_length + self.goal_depth, self.goal_width),
                            ],
                            [
                                (self.field_length, -self.goal_width),
                                (self.field_length + self.goal_depth, -self.goal_width),
                            ],
                            [
                                (self.field_length, self.goal_width),
                                (self.field_length + self.goal_depth, self.goal_width),
                            ],
                            [
                                (-self.field_length - self.goal_depth, -self.goal_width),
                                (-self.field_length - self.goal_depth, self.goal_width),
                            ],
                            [
                                (-self.field_length, -self.goal_width),
                                (-self.field_length - self.goal_depth, -self.goal_width),
                            ],
                            [
                                (-self.field_length, self.goal_width),
                                (-self.field_length - self.goal_depth, self.goal_width),
                            ],
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
                        ],
                        colors=["b" for _ in range(3)] + ["r" for _ in range(3)] + ["w" for _ in range(17)],
                        linewidths=3,
                        transform=rot,
                        linestyles="solid",
                        **self.default_plot_kwargs
                    )
                )

                if color_bar:
                    cbar = self.figure.colorbar(pcolormesh, ax=ax)
                    cbar.ax.tick_params(labelsize=20)
                self.pcolormeshes.append(pcolormesh)
                self.scatters.append(scatter)
                self.scatterBs.append(scatterB)
                self.scatterRs.append(scatterR)
                self.scatterOthers.append(scatterOther)
                self.lines.append(lineCollection)
                self.fieldcircles.append(fieldcircle)
                self.fieldLines.append(fieldLineCollection)

    def plot_value_fun(self, ego_world_state, values):
        num_cols = self.num_value_funcs
        red_pos_w = ego_world_state[3 * self.num_agents_per_team : -2].view(self.num_agents_per_team, 3).numpy()
        blue_pos_w = ego_world_state[: 3 * self.num_agents_per_team].view(self.num_agents_per_team, 3).numpy()
        ball_pos_w = ego_world_state[-2:]
        for j in range(num_cols):
            for k in range(self.num_players_to_visualize):
                markers = blue_pos_w[k, :2] * 1.0
                i = k + j * self.num_players_to_visualize
                ax = self.axs_list[j][k]

                self.pcolormeshes[i].set_array(values[i].ravel())
                self.scatters[i].set_offsets(ball_pos_w[:2])
                self.scatterOthers[i].set_offsets(markers[:2])
                self.scatterBs[i].set_offsets(blue_pos_w[:, :2])
                self.scatterRs[i].set_offsets(red_pos_w[:, :2])
                agent_pose = np.concatenate((blue_pos_w, red_pos_w), axis=0)
                B = np.concatenate(
                    (
                        np.expand_dims(np.cos(agent_pose[:, 2]), axis=1),
                        np.expand_dims(np.sin(agent_pose[:, 2]), axis=1),
                    ),
                    axis=1,
                )
                line_segments = np.concatenate(
                    (np.expand_dims(agent_pose[:, :2], axis=1), np.expand_dims(agent_pose[:, :2] + B * 0.1, axis=1)),
                    axis=1,
                )
                self.lines[i].set_segments(line_segments)
                for art in [
                    self.pcolormeshes[i],
                    self.fieldLines[i],
                    self.fieldcircles[i],
                    self.scatterOthers[i],
                    self.scatters[i],
                    self.scatterBs[i],
                    self.scatterRs[i],
                    self.lines[i],
                ]:
                    ax.draw_artist(art)
                self.figure.canvas.blit(ax.bbox)
        self.figure.canvas.flush_events()

    def terminate(self):
        plt.close("all")

    def call_back(self):
        self.last_time_stamp = time.time()
        last_item = None
        while not self.queue.empty():
            item = self.queue.get()
            if item is None:
                self.terminate()
                break
            last_item = item
        if last_item is not None:
            ego_world_state, values = last_item
            self.plot_value_fun(ego_world_state, values)
        return True

    def __call__(self, queue):
        print("starting plotter...")
        self.queue = queue
        self.init_plot()
        self.last_time_stamp = time.time()
        timer = self.figure.canvas.new_timer(interval=25)
        timer.add_callback(self.call_back)
        timer.start()

        self.reset_plot(self.x, self.y)
        plt.show()
