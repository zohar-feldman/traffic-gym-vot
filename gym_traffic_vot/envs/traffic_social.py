from gym_traffic_vot.envs.traffic_vot import TrafficVotEnv, option
from baselines.deepq.models import build_q_func
import tensorflow as tf
from baselines.deepq.utils import ObservationInput
import baselines.common.tf_util as U
import os
import traci
from traci import constants as tc
import numpy
import math
from gym_traffic_vot.envs.networks.simple.simple_network import SimpleTrafficNetwork

class TrafficSocialEnv(TrafficVotEnv):
    def __init__(self, load_path, network=SimpleTrafficNetwork(None, None), mode="gui", simulation_end=1000, sleep_between_restart=1, vots = None):
        super(TrafficSocialEnv, self).__init__(mode=mode, network=network, simulation_end=simulation_end, vots=vots)
        self.sess = tf.Session()
        q_func = build_q_func('mlp')
        with tf.variable_scope('deepq_play', reuse=tf.AUTO_REUSE):
            self.obs_t_input = ObservationInput(self.observation_space, name="obs_t")
            self.q_t = q_func(self.obs_t_input.get(), self.action_space.n, scope="q_func")
            self.v_t = tf.squeeze(tf.reduce_max(self.q_t, axis=1))
            self.q_action = tf.squeeze(tf.argmax(self.q_t, axis=1))
            tf.initialize_all_variables().run(session=self.sess)
            self.q_values = U.function([self.obs_t_input], self.q_t)
            self.load_variables(load_path, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="deepq_play"))

    def load_variables(self, load_path, variables=None):
        import joblib
        # sess = tf.get_default_session()
        variables = variables or tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        loaded_params = joblib.load(os.path.expanduser(load_path))
        restores = []
        if isinstance(loaded_params, list):
            assert len(loaded_params) == len(variables), 'number of variables loaded mismatches len(variables)'
            for d, v in zip(loaded_params, variables):
                restores.append(v.assign(d))
        else:
            for v in variables:
                restores.append(v.assign(loaded_params[v.name.replace("_play", "")]))
        self.sess.run(restores)

    def reward(self):
        sum_vot = super(TrafficSocialEnv, self).reward()
        sum_payments = 0
        v_curr = self.sess.run([self.v_t], {self.obs_t_input.get(): [self.curr_observation]})[0]
        for vehicle in self.last_step_loaded_vehicles:
            delta = numpy.zeros_like(self.curr_observation)
            stats = traci.vehicle.getSubscriptionResults(vehicle)
            lane_id = stats[tc.VAR_LANE_ID] if option == 1 else traci.vehicle.getLaneID(vehicle)
            pos = stats[tc.VAR_LANEPOSITION]
            cell = math.floor(pos / self.cell_size)
            index = 0
            for l in self.inc_lanes:
                if l == lane_id:
                    break
                index += self.lanes_size[lane_id]
            index += cell
            delta[index] += self.vehicles_vot[vehicle]
            v_minus = self.sess.run([self.v_t], {self.obs_t_input.get(): [self.curr_observation - delta]})[0]#self.q_values(self.state - delta)
            payment = v_curr - v_minus
            sum_payments += payment

        return sum_payments - sum_vot