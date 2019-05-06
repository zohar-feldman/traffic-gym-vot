from string import Template


class TrafficNetwork(object):
    def __init__(self, lights, netfile, routefile,
                       guifile, loops, addfile, lanes,
                       inc_lanes, exitloops):
        self.lights = lights
        self.netfile = netfile
        self.routefile = routefile
        self.guifile = guifile
        self.loops = loops
        self.addfile = addfile
        self.lanes = lanes
        self.inc_lanes = inc_lanes
        self.exitloops = exitloops


    def route_sample(self, random):
        pass

    def max_cross_time(self):
        return 4