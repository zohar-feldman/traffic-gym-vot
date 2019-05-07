from gym_traffic_vot.envs.traffic_vot import TrafficVotEnv
from traffic_lights import TrafficLightTwoWay
import os


class SimpleTrafficEnv(TrafficVotEnv):
    def __init__(self, mode="gui", vots=None):
        lights = [TrafficLightTwoWay(id="0", yield_time=5)]
        loops = ["loop{}".format(i) for i in range(12)]
        lanes = ["n_0_0", "s_0_0", "e_0_0", "w_0_0", "0_n_0", "0_s_0", "0_e_0", "0_w_0"]
        junc_lanes = []
        inc_lanes = ["n_0_0", "s_0_0", "e_0_0", "w_0_0"]
        basepath = os.path.dirname(__file__)
        netfile = os.path.join(basepath, "traffic.net.xml")
        routefile = os.path.join(basepath, "traffic.rou.xml")
        guifile = os.path.join(basepath, "view.settings.xml")
        addfile = os.path.join(basepath, "traffic.add.xml")
        exitloops = ["loop4", "loop5", "loop6", "loop7"]
        super(SimpleTrafficEnv, self).__init__(mode=mode, lights=lights, netfile=netfile, routefile=routefile,
                                               guifile=guifile, loops=loops, addfile=addfile, simulation_end=1000,
                                               lanes=lanes, inc_lanes=inc_lanes, exitloops=exitloops, vots=vots)

    def route_sample(self):
        if self.np_random.uniform(0, 1) > 0.5:
            ew = 0.01
            ns = 0.05
        else:
            ns = 0.01
            ew = 0.05
        return {"ns": ns,
                "sn": ns,
                "ew": ew,
                "we": ew
                }