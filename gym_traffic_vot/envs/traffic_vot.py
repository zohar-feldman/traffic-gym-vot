from gym_traffic_vot.envs.traffic_env import TrafficEnv
import traci
import math
from gym import error, spaces, utils
from scipy.stats import uniform, truncnorm, lognorm
import numpy
from traci import constants as tc
import time

option = 2

class TrafficVot(TrafficEnv):
    def __init__(self, lights, netfile, routefile, guifile, addfile, loops=[], lanes=[], inc_lanes=[], exitloops=[],
                 tmpfile="tmp.rou.xml",
                 pngfile="tmp.png", mode="gui", simulation_end=3600, sleep_between_restart=1):
        super(TrafficVot, self).__init__(mode=mode, lights=lights, netfile=netfile, routefile=routefile,
                                               guifile=guifile, loops=loops, addfile=addfile, simulation_end=1000,
                                               lanes=lanes, inc_lanes=inc_lanes, exitloops=exitloops)

        self.cell_size = 4.5
        self.lanes_size = {id:math.ceil(traci.lane.getLength(id) / self.cell_size) for id in self.inc_lanes}
        self.lanes_max_speed = {id:traci.lane.getMaxSpeed(id) for id in self.inc_lanes}

        if len(self.lights) == 1:
            self.action_space = spaces.Discrete(len(self.lights[0].actions))
        else:
            self.action_space = spaces.MultiDiscrete([len(l.actions) for l in self.lights])

        vot_space = spaces.Box(low=float('-inf'), high=float('inf'),
                                  shape=(sum(self.lanes_size.values()),))
        light_spaces = [spaces.Discrete(len(light.actions)) for light in self.lights]
        lane_vel_space = spaces.Box(low=float(0), high=float('inf'),
                                  shape=(len(self.inc_lanes),))

        self.observation_space = spaces.Tuple([vot_space] + [lane_vel_space] + light_spaces)
        self.vehicles_vot = dict()
        self.vehicles = list()

    def start_sumo(self):
        super(TrafficVot, self).start_sumo()
        # option 2: subs
        if option == 2:
            for lane in self.inc_lanes:
                traci.lane.subscribe(lane, [tc.LAST_STEP_VEHICLE_ID_LIST])

    def assign_vot(self, vehicles):
        self.vehicles_vot.update({v:lognorm.rvs(size=1, s=3) for v in vehicles}) # TODO: different distributions for different lanes

    def step(self, action):
        action = [action]
        self.sumo_step += 1
        assert (len(action) == len(self.lights))
        for act, light in zip(action, self.lights):
            signal = light.act(act)
            traci.trafficlights.setRedYellowGreenState(light.id, signal)
        traci.simulationStep()

        vehicles = traci.simulation.getDepartedIDList()
        self.vehicles = traci.vehicle.getIDList() if option == 1 else ()
        self.assign_vot(vehicles)
        # option 1: subscribe vehicles
        for veh_id in vehicles:
            vars = [tc.VAR_LANEPOSITION, tc.VAR_SPEED]
            if option == 1:
                vars.append(tc.VAR_LANE_ID)
            traci.vehicle.subscribe(veh_id, vars)

        # TODO: check if it is worthwhile to unsubscribe removed vehicles
        observation = self.observation()
        reward = self.reward()
        done = self.sumo_step > self.simulation_end
        self.screenshot()
        if done:
            self.stop_sumo()
        return observation, reward, done, self.route_info

    def observation(self):
        vot_state = {lane:numpy.zeros(shape=(self.lanes_size[lane],)) for lane in self.inc_lanes}
        lane_speeds = dict()
        velobs = list()
        if option == 1:
            for veh in self.vehicles:
                stats = traci.vehicle.getSubscriptionResults(veh)
                lane_id = stats[tc.VAR_LANE_ID]
                lane_stat = vot_state.get(lane_id)
                if lane_stat is not None:
                    pos = stats[tc.VAR_LANEPOSITION]
                    cell = math.floor(pos / self.cell_size)
                    lane_stat[cell] += self.vehicles_vot[veh]
                    speed = stats[tc.VAR_SPEED]
                    tmp_speed = lane_speeds.get(lane_id)
                    if tmp_speed is None or tmp_speed[0] < pos:
                        lane_speeds[lane_id] = [pos, speed]
        else:
            for lane_id in self.inc_lanes:
                lane_vehicles = traci.lane.getSubscriptionResults(lane_id)[tc.LAST_STEP_VEHICLE_ID_LIST]
                # print([traci.vehicle.getSubscriptionResults(veh)[tc.VAR_LANEPOSITION] for veh in lane_vehicles])
                lane_stat = vot_state.get(lane_id)
                for veh in lane_vehicles:
                    stats = traci.vehicle.getSubscriptionResults(veh)
                    # if lane_stat is not None:
                    pos = stats[tc.VAR_LANEPOSITION]
                    cell = math.floor(pos / self.cell_size)
                    lane_stat[cell] += self.vehicles_vot[veh]
                    if veh == lane_vehicles[-1]:
                        speed = stats[tc.VAR_SPEED]
                        velobs.append(speed)
                        # tmp_speed = lane_speeds.get(lane_id)
                        # if tmp_speed is None or tmp_speed[0] < pos:
                        #     lane_speeds[lane_id] = [pos, speed]
        if option == 1:
            velobs = [lane_speeds[lane][1] if lane in lane_speeds else self.lanes_max_speed[lane] for lane in self.inc_lanes]
        stack = [vot_state[lane] for lane in self.inc_lanes]
        trafficobs = numpy.hstack(stack) # TODO: check if safe for order
        lightobs = [light.state for light in self.lights]

        return (trafficobs, velobs, lightobs)

    def reward(self):
        in_tracffic_vehIDs = self.vehicles
        if option == 1:
            reward = float(-numpy.sum([self.vehicles_vot[v] for v in in_tracffic_vehIDs if traci.vehicle.getSubscriptionResults(v)[tc.VAR_LANE_ID] in self.inc_lanes]))
        else:
            reward = 0.
            for lane in self.inc_lanes:
                reward += float(-numpy.sum([self.vehicles_vot[v] for v in traci.lane.getSubscriptionResults(lane)[tc.LAST_STEP_VEHICLE_ID_LIST]]))
        return reward

    def reset(self):
        self.vehicles_vot.clear()
        self.vehicles = ()
        # self.lanes_size.clear()
        self.stop_sumo()
        # sleep required on some systems
        if self.sleep_between_restart > 0:
            time.sleep(self.sleep_between_restart)
        self.start_sumo()
        observation = self.observation()
        return observation
