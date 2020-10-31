import time
from contextlib import contextmanager
from collections import deque

import gym
from mpi4py import MPI
import tensorflow as tf
import numpy as np

import stable_baselines.common.tf_util as tf_util
from stable_baselines.common import explained_variance, zipsame, dataset, fmt_row, colorize, ActorCriticRLModel, \
    SetVerbosity, TensorboardWriter
from stable_baselines import logger
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.cg import conjugate_gradient
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.mdpo_on.utils import traj_segment_generator, add_vtarg_and_adv, flatten_lists
from stable_baselines.sac_trpo.tf_tsallis_statistics import *

class MDPO(ActorCriticRLModel):
    """
    Mirror Descent Policy Optimization (On-policy)

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount value
    :param timesteps_per_batch: (int) the number of timesteps to run per batch (horizon)
    :param max_kl: (float) the Kullback-Leibler loss threshold
    :param cg_iters: (int) the number of iterations for the conjugate gradient calculation
    :param lam: (float) GAE factor
    :param entcoeff: (float) the weight for the entropy loss
    :param cg_damping: (float) the compute gradient dampening factor
    :param vf_stepsize: (float) the value function stepsize
    :param vf_iters: (int) the value function's number iterations for learning
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    """

    def __init__(self, policy, env, gamma=0.99, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, lam=0.98,
                 entcoeff=0.0, cg_damping=1e-2, vf_stepsize=3e-4, vf_iters=3, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=0, sgd_steps=5, 
                 klcoeff=0.1, method="multistep-SGD", tsallis_q=1.0):
        super(MDPO, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False,
                                   _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs)

        self.using_gail = False
        self.timesteps_per_batch = timesteps_per_batch
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.gamma = gamma
        self.lam = lam
        self.max_kl = max_kl
        self.vf_iters = vf_iters
        self.vf_stepsize = vf_stepsize
        self.entcoeff = entcoeff
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        # GAIL Params
        self.hidden_size_adversary = 100
        self.adversary_entcoeff = 1e-3
        self.expert_dataset = None
        self.g_step = 1
        self.d_step = 1
        self.d_stepsize = 3e-4

        self.graph = None
        self.sess = None
        self.policy_pi = None
        self.loss_names = None
        self.assign_old_eq_new = None
        self.compute_losses = None
        self.compute_lossandgrad = None
        self.compute_fvp = None
        self.compute_vflossandgrad = None
        self.d_adam = None
        self.vfadam = None
        self.get_flat = None
        self.set_from_flat = None
        self.timed = None
        self.allmean = None
        self.nworkers = None
        self.rank = None
        self.reward_giver = None
        self.step = None
        self.proba_step = None
        self.initial_state = None
        self.params = None
        self.summary = None
        self.episode_reward = None
        self.seed = seed
        self.sgd_steps = sgd_steps
        self.klcoeff = klcoeff
        self.cliprange_vf = 0.2
        self.method = method
        self.tsallis_q = tsallis_q

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_pi
        action_ph = policy.pdtype.sample_placeholder([None])
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.obs_ph, action_ph, policy.policy
        return policy.obs_ph, action_ph, policy.deterministic_action

    def setup_model(self):
        # prevent import loops
        from stable_baselines.gail.adversary import TransitionClassifier

        with SetVerbosity(self.verbose):

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the MDPO model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            self.nworkers = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
            np.set_printoptions(precision=3)

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.single_threaded_session(graph=self.graph)
                self._setup_learn(self.seed)

                if self.using_gail:
                    self.reward_giver = TransitionClassifier(self.observation_space, self.action_space,
                                                             self.hidden_size_adversary,
                                                             entcoeff=self.adversary_entcoeff)

                # Construct network for new policy
                self.policy_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                # Network for old policy
                with tf.variable_scope("oldpi", reuse=False):
                    self.old_policy = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                # Network for fitting closed form
                with tf.variable_scope("closedpi", reuse=False):
                    self.closed_policy = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                             None, reuse=False, **self.policy_kwargs)

                with tf.variable_scope("loss", reuse=False):
                    self.atarg = tf.placeholder(dtype=tf.float32, shape=[None])  # Target advantage function (if applicable)
                    self.vtarg = tf.placeholder(dtype=tf.float32, shape=[None])
                    self.ret = tf.placeholder(dtype=tf.float32, shape=[None])  # Empirical return
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.outer_learning_rate_ph = tf.placeholder(tf.float32, [], name="outer_learning_rate_ph")
                    self.old_vpred_ph = tf.placeholder(tf.float32, [None], name="old_vpred_ph")
                    self.clip_range_vf_ph = tf.placeholder(tf.float32, [], name="clip_range_ph")

                    observation = self.policy_pi.obs_ph
                    self.action = self.policy_pi.pdtype.sample_placeholder([None])

                    if self.tsallis_q == 1.0:
                        #kloldnew = self.old_policy.proba_distribution.kl(self.policy_pi.proba_distribution)
                        kloldnew = self.policy_pi.proba_distribution.kl(self.old_policy.proba_distribution)
                        ent = self.policy_pi.proba_distribution.entropy()
                        meankl = tf.reduce_mean(kloldnew)

                    else:
                        logp_pi = self.policy_pi.proba_distribution.logp(self.action)
                        logp_pi_old =  self.old_policy.proba_distribution.logp(self.action)
                        ent = self.policy_pi.proba_distribution.entropy()
                        #kloldnew = self.policy_pi.proba_distribution.kl_tsallis(self.old_policy.proba_distribution, self.tsallis_q)
                        tsallis_q = 2.0 - self.tsallis_q
                        meankl = tf.reduce_mean(tf_log_q(tf.exp(logp_pi), tsallis_q) - tf_log_q(tf.exp(logp_pi_old), tsallis_q)) #tf.reduce_mean(kloldnew)

                    meanent = tf.reduce_mean(ent)
                    entbonus = self.entcoeff * meanent

                    if self.cliprange_vf is None:
                        vpred_clipped = self.policy_pi.value_flat
                    else:
                        vpred_clipped = self.old_vpred_ph + \
                            tf.clip_by_value(self.policy_pi.value_flat - self.old_vpred_ph,
                                             - self.clip_range_vf_ph, self.clip_range_vf_ph)

                    vf_losses1 = tf.square(self.policy_pi.value_flat - self.ret)
                    vf_losses2 = tf.square(vpred_clipped - self.ret)
                    vferr = tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

                    # advantage * pnew / pold
                    ratio = tf.exp(self.policy_pi.proba_distribution.logp(self.action) -
                                   self.old_policy.proba_distribution.logp(self.action))

                    if self.method == "multistep-SGD":
                        surrgain = tf.reduce_mean(ratio * self.atarg) - meankl / self.learning_rate_ph 
                    elif self.method == "closedreverse-KL":    
                        surrgain = tf.reduce_mean(tf.exp(self.atarg) * self.policy_pi.proba_distribution.logp(self.action))
                    else:
                        policygain = tf.reduce_mean(tf.exp(self.atarg) * tf.log(self.closed_policy.proba_distribution.mean)) 
                        surrgain = tf.reduce_mean(ratio * self.atarg) - tf.reduce_mean(self.learning_rate_ph * ratio * self.policy_pi.proba_distribution.logp(self.action))

                    optimgain = surrgain #+ entbonus - self.learning_rate_ph * meankl
                    losses = [optimgain, meankl, entbonus, surrgain, meanent]
                    self.loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

                    dist = meankl

                    all_var_list = tf_util.get_trainable_vars("model")
                    var_list = [v for v in all_var_list if "/vf" not in v.name and "/q/" not in v.name]
                    vf_var_list = [v for v in all_var_list if "/pi" not in v.name and "/logstd" not in v.name]
                    print("policy vars", var_list)

                    all_closed_var_list = tf_util.get_trainable_vars("closedpi")
                    closed_var_list = [v for v in all_closed_var_list if "/vf" not in v.name and "/q" not in v.name]

                    self.get_flat = tf_util.GetFlat(var_list, sess=self.sess)
                    self.set_from_flat = tf_util.SetFromFlat(var_list, sess=self.sess)

                    klgrads = tf.gradients(dist, var_list)
                    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
                    shapes = [var.get_shape().as_list() for var in var_list]
                    start = 0
                    tangents = []
                    for shape in shapes:
                        var_size = tf_util.intprod(shape)
                        tangents.append(tf.reshape(flat_tangent[start: start + var_size], shape))
                        start += var_size
                    gvp = tf.add_n([tf.reduce_sum(grad * tangent)
                                    for (grad, tangent) in zipsame(klgrads, tangents)])  # pylint: disable=E1111
                    fvp = tf_util.flatgrad(gvp, var_list)

                    tf.summary.scalar('entropy_loss', meanent)
                    tf.summary.scalar('policy_gradient_loss', optimgain)
                    tf.summary.scalar('value_function_loss', surrgain)
                    tf.summary.scalar('approximate_kullback-leibler', meankl)
                    tf.summary.scalar('loss', optimgain + meankl + entbonus + surrgain + meanent)

                    self.assign_old_eq_new = \
                        tf_util.function([], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                                          zipsame(tf_util.get_globals_vars("oldpi"),
                                                                  tf_util.get_globals_vars("model"))])
                    self.compute_losses = tf_util.function([observation, self.old_policy.obs_ph, self.action, self.atarg, self.learning_rate_ph, self.vtarg], losses)
                    self.compute_fvp = tf_util.function([flat_tangent, observation, self.old_policy.obs_ph, self.action, self.atarg],
                                                        fvp)
                    self.compute_vflossandgrad = tf_util.function([observation, self.old_policy.obs_ph, self.ret, self.old_vpred_ph, self.clip_range_vf_ph],
                                                                  tf_util.flatgrad(vferr, vf_var_list))

                    grads = tf.gradients(-optimgain, var_list)
                    grads, _grad_norm = tf.clip_by_global_norm(grads, 0.5)
                    trainer = tf.train.AdamOptimizer(learning_rate=self.outer_learning_rate_ph, epsilon=1e-5)
                    #trainer = tf.train.AdamOptimizer(learning_rate=3e-4, epsilon=1e-5)
                    grads = list(zip(grads, var_list))
                    self._train = trainer.apply_gradients(grads)

                    @contextmanager
                    def timed(msg):
                        if self.rank == 0 and self.verbose >= 1:
                            print(colorize(msg, color='magenta'))
                            start_time = time.time()
                            yield
                            print(colorize("done in {:.3f} seconds".format((time.time() - start_time)),
                                           color='magenta'))
                        else:
                            yield

                    def allmean(arr):
                        assert isinstance(arr, np.ndarray)
                        out = np.empty_like(arr)
                        MPI.COMM_WORLD.Allreduce(arr, out, op=MPI.SUM)
                        out /= self.nworkers
                        return out

                    tf_util.initialize(sess=self.sess)

                    th_init = self.get_flat()
                    MPI.COMM_WORLD.Bcast(th_init, root=0)
                    self.set_from_flat(th_init)

                with tf.variable_scope("Adam_mpi", reuse=False):
                    self.vfadam = MpiAdam(vf_var_list, sess=self.sess)
                    if self.using_gail:
                        self.d_adam = MpiAdam(self.reward_giver.get_trainable_variables(), sess=self.sess)
                        self.d_adam.sync()
                    self.vfadam.sync()

                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(self.ret))
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.vf_stepsize))
                    tf.summary.scalar('advantage', tf.reduce_mean(self.atarg))
                    tf.summary.scalar('kl_clip_range', tf.reduce_mean(self.max_kl))

                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.ret)
                        tf.summary.histogram('learning_rate', self.vf_stepsize)
                        tf.summary.histogram('advantage', self.atarg)
                        tf.summary.histogram('kl_clip_range', self.max_kl)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', observation)
                        else:
                            tf.summary.histogram('observation', observation)

                self.timed = timed
                self.allmean = allmean

                self.step = self.policy_pi.step
                self.proba_step = self.policy_pi.proba_step
                self.initial_state = self.policy_pi.initial_state

                self.params = tf_util.get_trainable_vars("model") + tf_util.get_trainable_vars("oldpi")
                if self.using_gail:
                    self.params.extend(self.reward_giver.get_trainable_variables())

                self.summary = tf.summary.merge_all()

                self.compute_lossandgrad = \
                    tf_util.function([observation, self.old_policy.obs_ph, self.action, self.atarg, self.ret, self.learning_rate_ph, self.vtarg, self.closed_policy.obs_ph],
                                     [self.summary, tf_util.flatgrad(optimgain, var_list)] + losses)

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="MDPO",
              reset_num_timesteps=True):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        print("got seed, sgd_steps", seed, self.sgd_steps)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            with self.sess.as_default():
                seg_gen = traj_segment_generator(self.old_policy, self.env, self.timesteps_per_batch,
                                                 reward_giver=self.reward_giver, gail=self.using_gail, entcoeff=self.entcoeff)

                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()
                len_buffer = deque(maxlen=40)  # rolling buffer for episode lengths
                reward_buffer = deque(maxlen=40)  # rolling buffer for episode rewards
                self.episode_reward = np.zeros((self.n_envs,))
                self.outer_learning_rate = get_schedule_fn(3e-4)
                self.cliprange_vf = get_schedule_fn(0.2)

                true_reward_buffer = None
                if self.using_gail:
                    true_reward_buffer = deque(maxlen=40)

                    # Initialize dataloader
                    batchsize = self.timesteps_per_batch // self.d_step
                    self.expert_dataset.init_dataloader(batchsize)

                    #  Stats not used for now
                    # TODO: replace with normal tb logging
                    #  g_loss_stats = Stats(loss_names)
                    #  d_loss_stats = Stats(reward_giver.loss_name)
                    #  ep_stats = Stats(["True_rewards", "Rewards", "Episode_length"])

                while True:
                    if callback is not None:
                        # Only stop training if return value is False, not when it is None. This is for backwards
                        # compatibility with callbacks that have no return statement.
                        if callback(locals(), globals()) is False:
                            break
                    if total_timesteps and timesteps_so_far >= total_timesteps:
                        break

                    logger.log("********** Iteration %i ************" % iters_so_far)

                    #def fisher_vector_product(vec):
                    #    return self.allmean(self.compute_fvp(vec, *fvpargs, sess=self.sess)) + self.cg_damping * vec

                    # ------------------ Update G ------------------
                    logger.log("Optimizing Policy...")
                    # g_step = 1 when not using GAIL
                    mean_losses = None
                    vpredbefore = None
                    tdlamret = None
                    observation = None
                    action = None
                    seg = None
                    for k in range(self.g_step):
                        with self.timed("sampling"):
                            seg = seg_gen.__next__()
                        add_vtarg_and_adv(seg, self.gamma, self.lam)
                        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                        observation, action = seg["observations"], seg["actions"]
                        atarg, tdlamret = seg["adv"], seg["tdlamret"]


                        vpredbefore = seg["vpred"]  # predicted value function before update
                        atarg = (atarg - atarg.mean()) / atarg.std()  # standardized advantage function estimate

                        # true_rew is the reward without discount
                        if writer is not None:
                            self.episode_reward = total_episode_reward_logger(self.episode_reward,
                                                                              seg["true_rewards"].reshape(
                                                                                  (self.n_envs, -1)),
                                                                              seg["dones"].reshape((self.n_envs, -1)),
                                                                              writer, self.num_timesteps)

                        n_updates = int(total_timesteps / self.timesteps_per_batch)
                        lr_now = 1.0 - (iters_so_far - 1.0) / n_updates
                        outer_lr_now = self.outer_learning_rate(1.0 - (iters_so_far - 1.0) / n_updates)
                        clip_now = self.cliprange_vf(1.0 - (iters_so_far - 1.0) / n_updates)
                        args = seg["observations"], seg["observations"], seg["actions"], atarg
                        # Subsampling: see p40-42 of John Schulman thesis
                        # http://joschu.net/docs/thesis.pdf
                        #fvpargs = [arr[::5] for arr in args]

                        with self.timed("computegrad"):
                            steps = self.num_timesteps + (k + 1) * (seg["total_timestep"] / self.g_step)
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata() if self.full_tensorboard_log else None
                            # run loss backprop with summary, and save the metadata (memory, compute time, ...)
                            if writer is not None:
                                summary, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, sess=self.sess,
                                                                                      options=run_options,
                                                                                      run_metadata=run_metadata)
                                if self.full_tensorboard_log:
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *lossbefore = self.compute_lossandgrad(*args, tdlamret, lr_now, seg["vpred"], seg["observations"], sess=self.sess,
                                                                                options=run_options,
                                                                                run_metadata=run_metadata)
                                td_map = {self.policy_pi.obs_ph: seg["observations"], self.old_policy.obs_ph: seg["observations"], self.closed_policy.obs_ph: seg["observations"],
                                            self.action: seg["actions"], self.atarg: atarg, self.ret: tdlamret,
                                            self.learning_rate_ph: lr_now, self.outer_learning_rate_ph: outer_lr_now, self.vtarg: seg["vpred"]}
                                for _ in range(int(self.sgd_steps)):
                                    _ = self.sess.run(self._train, td_map)
                                    #if self.method == "closed-KL":
                                    #    _ = self.sess.run(self._train_policy, td_map)

                        if np.allclose(grad, 0):
                            logger.log("Got zero gradient. not updating")
                        else:
                            for _ in range(1):
                                mean_losses = surr, kl_loss, *_ = self.allmean(
                                    np.array(self.compute_losses(*args, lr_now, seg["vpred"], sess=self.sess)))

                        with self.timed("vf"):
                            for _ in range(self.vf_iters):
                                # NOTE: for recurrent policies, use shuffle=False?
                                for (mbob, mbret, mbval) in dataset.iterbatches((seg["observations"], seg["tdlamret"], seg["vpred"]),
                                                                         include_final_partial_batch=False,
                                                                         batch_size=128,
                                                                         shuffle=True):
                                    grad = self.allmean(self.compute_vflossandgrad(mbob, mbob, mbret, mbval, clip_now, sess=self.sess))
                                    self.vfadam.update(grad, outer_lr_now) #self.vf_stepsize)

                        if iters_so_far % 1 == 0:
                            print("updating theta now")
                            self.assign_old_eq_new(sess=self.sess)

                    for (loss_name, loss_val) in zip(self.loss_names, mean_losses):
                        logger.record_tabular(loss_name, loss_val)

                    logger.record_tabular("explained_variance_tdlam_before",
                                          explained_variance(vpredbefore, tdlamret))

                    if self.using_gail:
                        # ------------------ Update D ------------------
                        logger.log("Optimizing Discriminator...")
                        logger.log(fmt_row(13, self.reward_giver.loss_name))
                        assert len(observation) == self.timesteps_per_batch
                        batch_size = self.timesteps_per_batch // self.d_step

                        # NOTE: uses only the last g step for observation
                        d_losses = []  # list of tuples, each of which gives the loss for a minibatch
                        # NOTE: for recurrent policies, use shuffle=False?
                        for ob_batch, ac_batch in dataset.iterbatches((observation, action),
                                                                      include_final_partial_batch=False,
                                                                      batch_size=batch_size,
                                                                      shuffle=True):
                            ob_expert, ac_expert = self.expert_dataset.get_next_batch()
                            # update running mean/std for reward_giver
                            if self.reward_giver.normalize:
                                self.reward_giver.obs_rms.update(np.concatenate((ob_batch, ob_expert), 0))

                            # Reshape actions if needed when using discrete actions
                            if isinstance(self.action_space, gym.spaces.Discrete):
                                if len(ac_batch.shape) == 2:
                                    ac_batch = ac_batch[:, 0]
                                if len(ac_expert.shape) == 2:
                                    ac_expert = ac_expert[:, 0]
                            *newlosses, grad = self.reward_giver.lossandgrad(ob_batch, ac_batch, ob_expert, ac_expert)
                            self.d_adam.update(self.allmean(grad), self.d_stepsize)
                            d_losses.append(newlosses)
                        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

                        # lr: lengths and rewards
                        lr_local = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"])  # local values
                        list_lr_pairs = MPI.COMM_WORLD.allgather(lr_local)  # list of tuples
                        lens, rews, true_rets = map(flatten_lists, zip(*list_lr_pairs))
                        true_reward_buffer.extend(true_rets)
                    else:
                        # lr: lengths and rewards
                        lr_local = (seg["ep_lens"], seg["ep_rets"])  # local values
                        list_lr_pairs = MPI.COMM_WORLD.allgather(lr_local)  # list of tuples
                        lens, rews = map(flatten_lists, zip(*list_lr_pairs))
                    len_buffer.extend(lens)
                    reward_buffer.extend(rews)

                    if len(len_buffer) > 0:
                        logger.record_tabular("EpLenMean", np.mean(len_buffer))
                        logger.record_tabular("EpRewMean", np.mean(reward_buffer))
                    if self.using_gail:
                        logger.record_tabular("EpTrueRewMean", np.mean(true_reward_buffer))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    timesteps_so_far += current_it_timesteps
                    self.num_timesteps += current_it_timesteps
                    iters_so_far += 1

                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", self.num_timesteps)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)
                    logger.record_tabular("Tsallis-q", self.tsallis_q)

                    if self.verbose >= 1 and self.rank == 0:
                        logger.dump_tabular()

        return self

    def save(self, save_path):
        if self.using_gail and self.expert_dataset is not None:
            # Exit processes to pickle the dataset
            self.expert_dataset.prepare_pickling()
        data = {
            "gamma": self.gamma,
            "timesteps_per_batch": self.timesteps_per_batch,
            "max_kl": self.max_kl,
            "cg_iters": self.cg_iters,
            "lam": self.lam,
            "entcoeff": self.entcoeff,
            "cg_damping": self.cg_damping,
            "vf_stepsize": self.vf_stepsize,
            "vf_iters": self.vf_iters,
            "hidden_size_adversary": self.hidden_size_adversary,
            "adversary_entcoeff": self.adversary_entcoeff,
            "expert_dataset": self.expert_dataset,
            "g_step": self.g_step,
            "d_step": self.d_step,
            "d_stepsize": self.d_stepsize,
            "using_gail": self.using_gail,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save)

def get_schedule_fn(value_schedule):
    """
    Transform (if needed) learning rate and clip range
    to callable.

    :param value_schedule: (callable or float)
    :return: (function)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, (float, int)):
        # Cast to float to avoid errors
        value_schedule = constfn(float(value_schedule))
    else:
        assert callable(value_schedule)
    return value_schedule

def constfn(val):
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    :param val: (float)
    :return: (function)
    """

    def func(_):
        return val

    return func