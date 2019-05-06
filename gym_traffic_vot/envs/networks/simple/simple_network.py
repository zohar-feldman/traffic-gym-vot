from gym_traffic_vot.envs.networks.traffic_network import TrafficNetwork
from gym_traffic_vot.envs.networks.traffic_lights import TrafficLightTwoWay
import os

class SimpleTrafficNetwork(TrafficNetwork):
    def __init__(self, arrival_rate, scales):
        lights = [TrafficLightTwoWay(id="0", yield_time=5)]
        loops = ["loop{}".format(i) for i in range(12)]
        lanes = ["n_0_0", "s_0_0", "e_0_0", "w_0_0", "0_n_0", "0_s_0", "0_e_0", "0_w_0"]
        junc_lanes = []
        self.arrival_rate = arrival_rate
        self.scales = scales
        inc_lanes = ["n_0_0", "s_0_0", "e_0_0", "w_0_0"]
        basepath = os.path.dirname(__file__)
        netfile = os.path.join(basepath, "traffic.net.xml")
        routefile = os.path.join(basepath, "traffic.rou.xml")
        guifile = os.path.join(basepath, "view.settings.xml")
        addfile = os.path.join(basepath, "traffic.add.xml")
        exitloops = ["loop4", "loop5", "loop6", "loop7"]
        super(SimpleTrafficNetwork, self).__init__(lights, netfile, routefile,
                                                   guifile, loops, addfile, lanes,
                                                   inc_lanes, exitloops)


    def route_sample(self, random):
        a = self.arrival_rate / (max(self.scales['n_0_0'], self.scales['s_0_0']) + max(self.scales['e_0_0'], self.scales['w_0_0']))
        arrs = {"ns": a * self.scales['n_0_0'],
                "sn": a * self.scales['s_0_0'],
                "ew": a * self.scales['e_0_0'],
                "we": a * self.scales['w_0_0']
                }

        print ("lane arrivals probs: {}".format(arrs))
        return arrs
