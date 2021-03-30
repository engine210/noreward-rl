from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy, StateActionPredictor, StatePredictor
import six.moves.queue as queue
import scipy.signal
import threading
import distutils.version
from constants import constants

Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])

def discount(x, gamma):
    """
        x = [r1, r2, r3, ..., rN]
        returns [r1 + r2*gamma + r3*gamma^2 + ...,
                r2 + r3*gamma + r4*gamma^2 + ...,
                r3 + r4*gamma + r5*gamma^2 + ...,
                ..., ..., rN]
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def process_rollout(rollout, gamma, lambda_=1.0, clip=False):
    """
    Given a rollout, compute its returns and the advantage.
    """
    # collecting transitions
    if rollout.unsup:
        batch_si = np.asarray(rollout.states + [rollout.end_state])
    else:
        batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)

    # collecting target for value network
    # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])  # bootstrapping
    if rollout.unsup:
        rewards_plus_v += np.asarray(rollout.bonuses + [0])
    if clip:
        rewards_plus_v[:-1] = np.clip(rewards_plus_v[:-1], -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    batch_r = discount(rewards_plus_v, gamma)[:-1]  # value network target

    # collecting target for policy network
    rewards = np.asarray(rollout.rewards)
    if rollout.unsup:
        rewards += np.asarray(rollout.bonuses)
    if clip:
        rewards = np.clip(rewards, -constants['REWARD_CLIP'], constants['REWARD_CLIP'])
    vpred_t = np.asarray(rollout.values + [rollout.r])
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    # Eq (10): delta_t = Rt + gamma*V_{t+1} - V_t
    # Eq (16): batch_adv_t = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + ...
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)

class PartialRollout(object):
    """
    A piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self, unsup=False):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.unsup = unsup
        if self.unsup:
            self.bonuses = []
            self.end_state = None


    def add(self, state, action, reward, value, terminal, features, bonus=None, end_state=None):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        if self.unsup:
            self.bonuses += [bonus]
            self.end_state = end_state

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        if self.unsup:
            self.bonuses.extend(other.bonuses)
            self.end_state = other.end_state

class A2C:
    def __init__(self, unsupType, envWrap=True, designHead='universe', noReward=False):
        self.unsup = unsupType is not None
        self.envWrap = envWrap
        self.designHead = designHead
        self.numaction = 3
        self.obs_shape = [42, 42, 4]
        self.local_steps = 0
        self._build_net()

    def _build_net(self):
        self.network = pi = LSTMPolicy(self.obs_shape, self.numaction, self.designHead)

        with tf.variable_scope("predictor"):
            self.ap_network = predictor = StateActionPredictor(self.obs_shape, self.numaction, self.designHead)
        
        self.ac = tf.placeholder(tf.float32, [None, self.numaction], name="ac")
        self.adv = tf.placeholder(tf.float32, [None], name="adv")
        self.r = tf.placeholder(tf.float32, [None], name="r")
        log_prob_tf = tf.nn.log_softmax(pi.logits)
        prob_tf = tf.nn.softmax(pi.logits)
        pi_loss = - tf.reduce_mean(tf.reduce_sum(log_prob_tf * self.ac, 1) * self.adv)
        vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.vf - self.r))
        entropy = - tf.reduce_mean(tf.reduce_sum(prob_tf * log_prob_tf, 1))
        self.loss = pi_loss + 0.5 * vf_loss - entropy * constants['ENTROPY_BETA']

        # compute gradients
        grads = tf.gradients(self.loss, pi.var_list)

        # computing predictor loss
        self.predloss = constants['PREDICTION_LR_SCALE'] * (predictor.invloss * (1-constants['FORWARD_LOSS_WT']) + predictor.forwardloss * constants['FORWARD_LOSS_WT'])
        predgrads = tf.gradients(self.predloss, predictor.var_list)

        # clip gradients
        grads, _ = tf.clip_by_global_norm(grads, constants['GRAD_NORM_CLIP'])
        grads_and_vars = list(zip(grads, self.network.var_list))

        predgrads, _ = tf.clip_by_global_norm(predgrads, constants['GRAD_NORM_CLIP'])
        pred_grads_and_vars = list(zip(predgrads, self.ap_network.var_list))
        grads_and_vars = grads_and_vars + pred_grads_and_vars

        opt = tf.train.AdamOptimizer(constants['LEARNING_RATE'])
        self.train_op = tf.group(opt.apply_gradients(grads_and_vars))

    def process(self, sess, rollout):
        """
        Process grabs a rollout that's been produced by the thread runner,
        and updates the parameters.  The update is then sent to the parameter
        server.
        """
        batch = process_rollout(rollout, gamma=constants['GAMMA'], lambda_=constants['LAMBDA'], clip=self.envWrap)

        _, loss, pred_loss = sess.run([self.train_op, self.loss, self.predloss], feed_dict={
            self.network.x: batch.si,
            self.ac: batch.a,
            self.adv: batch.adv,
            self.r: batch.r,
            self.network.state_in[0]: batch.features[0],
            self.network.state_in[1]: batch.features[1],
            self.network.x: batch.si[:-1],
            self.ap_network.s1: batch.si[:-1],
            self.ap_network.s2: batch.si[1:],
            self.ap_network.asample: batch.a
        })
        self.local_steps += 1

        return loss, pred_loss