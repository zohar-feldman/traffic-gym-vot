from gym import Env
from gym import error, spaces, utils
from gym.error import Error
from gym.utils import seeding
import traci
import traci.constants as tc
from scipy.misc import imread
from gym import spaces
from string import Template
import os, sys
import numpy as np
import math
import random
import time

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


class TrafficEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, lights, netfile, routefile, guifile, addfile, loops=[], lanes=[], inc_lanes=[], exitloops=[],
                 tmpfile="tmp.rou.xml",
                 pngfile="tmp.png", mode="gui", simulation_end=3600, sleep_between_restart=1):
        self.simulation_end = simulation_end
        self.sleep_between_restart = sleep_between_restart
        self.mode = mode
        args = ["--net-file", netfile, "--route-files", tmpfile, "--additional-files", addfile]
        if mode == "gui":
            binary = "sumo-gui"
            args += ["-S", "-Q", "--gui-settings-file", guifile]
        else:
            binary = "sumo"
            args += ["--no-step-log"]

        with open(routefile) as f:
            self.route = f.read()
        self.tmpfile = tmpfile
        self.pngfile = pngfile
        self.sumo_cmd = [binary] + args
        self.sumo_running = False
        self._seed()
        self.simulation_end =  simulation_end
        self.sleep_between_restart = sleep_between_restart
        self.lanes = lanes
        self.inc_lanes = inc_lanes
        self.loops = loops
        self.exit_loops = exitloops
        self.lights = lights
        self.start_sumo()
        self.viewer = None

    def relative_path(self, *paths):
        os.path.join(os.path.dirname(__file__), *paths)

    def write_routes(self):
        self.route_info = self.route_sample()
        with open(self.tmpfile, 'w') as f:
            f.write(Template(self.route).substitute(self.route_info))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def start_sumo(self):
        if not self.sumo_running:
            self.write_routes()
            traci.start(self.sumo_cmd)
            self.sumo_step = 0
            self.sumo_running = True
            self.screenshot()

    def stop_sumo(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False



    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if self.mode == "gui":
            img = imread(self.pngfile, mode="RGB")
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                from gym.envs.classic_control import rendering
                if self.viewer is None:
                    self.viewer = rendering.SimpleImageViewer()
                self.viewer.imshow(img)
        else:
            pass #raise NotImplementedError("Only rendering in GUI mode is supported. Please use Traffic-...-gui-v0.")


    def screenshot(self):
        if self.mode == "gui":
            traci.gui.screenshot("View #0", self.pngfile)

    def close(self):
        traci.close()