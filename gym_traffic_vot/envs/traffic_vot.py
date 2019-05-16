from gym_traffic_vot.envs.traffic_env import TrafficEnv
import traci
import math
from gym import error, spaces, utils
from scipy.stats import uniform, truncnorm, lognorm
import numpy
from traci import constants as tc
import time
from gym_traffic_vot.envs.networks.simple.simple_network import SimpleTrafficNetwork
option = 2

def one_hot(ind, depth):
    res = [0] * depth
    res[ind] = 1
    return res

def sample():
    values = numpy.array([28.12, 40.89, 53.62, 73.70, 104.56, 145.37]) / 60
    weights = [0.1, 0.15, 0.25, 0.25, 0.15, 0.1]
    return numpy.random.choice(values, 1, p=weights)

class TrafficVotEnv(TrafficEnv):
    def __init__(self, network="simple", arrival_rate=None, scale=None, mode="gui", simulation_end=1000, sleep_between_restart=0, vots = True): #TODO: kqargs
        self.scales = {}
        self.scale = scale if scale is not None else 1
        if network == "simple":
            if scale is None:
                self.scale = 1
            if arrival_rate is None:
                arrival_rate = 0.2
            self.scales.update({'n_0_0':1, 's_0_0':1, 'w_0_0':scale, 'e_0_0':scale})
            network = SimpleTrafficNetwork(arrival_rate, self.scales)
        super(TrafficVotEnv, self).__init__(mode=mode, network=network, simulation_end=simulation_end)

        self.cell_size = 4.5
        self.lanes_size = {id:math.ceil(traci.lane.getLength(id) / self.cell_size) for id in self.inc_lanes}
        self.lanes_max_speed = {id:50 for id in self.inc_lanes} # TODO: replace with traci.lane.getMaxSpeed(id)
        if len(self.lights) == 1:
            self.action_space = spaces.Discrete(len(self.lights[0].actions))
        else:
            self.action_space = spaces.MultiDiscrete([len(l.actions) for l in self.lights])
        self.vots = vots
        vot_size = sum(self.lanes_size.values())
        low = [float('-inf')] * vot_size
        high = [0] * vot_size
        vel_size = len(self.inc_lanes)
        low.extend([0] * vel_size)
        high.extend([float('inf')] * vel_size)
        light_size = sum([len(light.actions) for light in self.lights])
        low.extend([0] * light_size)
        high.extend([1] * light_size)
        light_len = len(self.lights)
        low.extend([0] * light_len)
        high.extend([float('inf')] * light_len)

        self.observation_space = spaces.Box(low=numpy.array(low), high=numpy.array(high))
        self.vehicles_vot = dict()
        self.vehicles = list()

    def start_sumo(self):
        super(TrafficVotEnv, self).start_sumo()
        # option 2: subs
        if option == 2:
            for lane in self.inc_lanes:
                traci.lane.subscribe(lane, [tc.LAST_STEP_VEHICLE_ID_LIST])

    def assign_vot(self, vehicles):
        if self.vots:
            self.vehicles_vot.update({v:sample() * self.scale / self.scales[traci.vehicle.getLaneID(v)] for v in vehicles})
        else:
            self.vehicles_vot.update({v:1 for v in vehicles}) # TODO: different distributions for different lanes

    def step(self, action):
        action = [action]
        self.sumo_step += 1
        assert (len(action) == len(self.lights))
        for act, light in zip(action, self.lights):
            signal = light.act(act)
            traci.trafficlights.setRedYellowGreenState(light.id, signal)
        traci.simulationStep()

        self.last_step_loaded_vehicles = traci.simulation.getDepartedIDList() #VAR_LOADED_VEHICLES_NUMBER
        self.vehicles = traci.vehicle.getIDList() if option == 1 else ()
        self.assign_vot(self.last_step_loaded_vehicles)
        # option 1: subscribe vehicles
        for veh_id in self.last_step_loaded_vehicles:
            vars = [tc.VAR_LANEPOSITION, tc.VAR_SPEED]
            if option == 1:
                vars.append(tc.VAR_LANE_ID)
            traci.vehicle.subscribe(veh_id, vars)

        # TODO: check if it is worthwhile to unsubscribe removed vehicles
        self.curr_observation = self.observation()
        reward = self.reward()
        done = self.sumo_step > self.simulation_end
        self.screenshot()
        if done:
            self.stop_sumo()
        return self.curr_observation, reward, done, self.route_info

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
                if len(lane_vehicles) == 0:
                    velobs.append(self.lanes_max_speed[lane_id])
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
        lightobs = numpy.hstack([one_hot(light.state, len(light.actions)) for light in self.lights])
        lightstepobs = [light.step for light in self.lights]
        obs = numpy.hstack([trafficobs, velobs, lightobs, lightstepobs])
        return obs

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
