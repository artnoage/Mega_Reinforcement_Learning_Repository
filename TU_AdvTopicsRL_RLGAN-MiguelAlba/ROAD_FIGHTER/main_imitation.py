import logging
import glob
import os
import time

import numpy as np
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from ROAD_FIGHTER.arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from ROAD_FIGHTER.envs import make_env, _obs, set_gan_off
from ROAD_FIGHTER.model import Policy
from ROAD_FIGHTER.storage import RolloutStorage
import random

import ROAD_FIGHTER.agent as algo


def update_current_obs(obs, shape_dim0, current_obs):
    """
    update the current batch of observations
    :param obs: a new observation
    :param shape_dim0: the shape of the observation
    :param current_obs: the current observations batch
    """
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:].clone()
    current_obs[:, -shape_dim0:] = obs


def write_logging_info(update, total_num_steps, end, start, final_rewards, dist_entropy, value_loss, action_loss):
    template = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, " \
               "entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}\n "
    template.format(update, total_num_steps,
                    int(total_num_steps / (end - start)),
                    final_rewards.mean(),
                    final_rewards.median(),
                    final_rewards.min(),
                    final_rewards.max(), dist_entropy,
                    value_loss, action_loss)
    return template


class ImitationLearning(object):
    def __init__(self, args):
        self.gan_file = args.gan_imitation_file
        self.num_processes = 1
        self.envs, self.obs_shape = self.create_environment()

        # new a2c for level x, pre trained 12c in level 1, agent
        self.actor_critic, self.actor_critic_gan, self.agent = self.create_agent()
        # load pre-trained A2C model - the model is only used for the the trajectory collection phase
        if args.load:
            self.load_pretrained_a2c()

        # rollouts to start trajectory recollection
        self.rollouts, self.current_obs, self.current_obs_orig = self.init_rollouts(self.actor_critic_gan, True)

    def create_environment(self):
        envs = [make_env(i, args, True, self.gan_file) for i in range(self.num_processes)]
        envs = DummyVecEnv(envs)
        if len(envs.observation_space.shape) == 1:
            envs = VecNormalize(envs, gamma=args.gamma)

        obs_shape = envs.observation_space.shape
        obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
        return envs, obs_shape

    def create_agent(self):
        # create A2C model and agent
        actor_critic = Policy(self.obs_shape, self.envs.action_space, args.recurrent_policy)
        if args.cuda:
            actor_critic.cuda()

        agent = algo.A2C(actor_critic, args.value_loss_coef,
                         args.entropy_coef, lr=args.lr,
                         eps=args.eps, alpha=args.alpha,
                         max_grad_norm=args.max_grad_norm)
        actor_critic_gan = Policy(self.obs_shape, self.envs.action_space, args.recurrent_policy)
        if args.cuda:
            actor_critic_gan.cuda()

        return actor_critic, actor_critic_gan, agent

    def load_pretrained_a2c(self):
        if args.load_dir is None:
            fname = os.path.join(args.save_dir, args.env_name, '.pth.tar')
        else:
            fname = 'trained_models/' + args.env_name + '.pth.tar'
        print('Loading pretrain agent from : {}'.format(fname))
        logging.info(fname)
        if os.path.isfile(fname):
            if args.cuda:
                checkpoint = torch.load(fname)
            else:
                checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
            self.actor_critic_gan.load_state_dict(checkpoint['state_dict'])
            for param in self.actor_critic_gan.parameters():
                param.requires_grad = False
            logging.info("model loaded")

    def init_rollouts(self, ac_model, use_gan=False, last_obs=None):
        """
        initialize rollouts and observations
        :param ac_model: actor-critic model
        :param use_gan: True if using gans to generate, False otherwise
        :param last_obs: last observations received from the game
        """
        if use_gan:
            rollouts = RolloutStorage(args.num_traj_steps, self.num_processes, self.obs_shape, self.envs.action_space,
                                      ac_model.state_size, True)
            current_obs_orig = torch.zeros(self.num_processes, *self.obs_shape)
        else:
            rollouts = RolloutStorage(args.num_traj_steps, self.num_processes, self.obs_shape, self.envs.action_space,
                                      ac_model.state_size, False)
            current_obs_orig = None

        shape_dim0 = self.envs.observation_space.shape[0]
        current_obs = torch.zeros(self.num_processes, *self.obs_shape)

        if use_gan:
            obs = self.envs.reset()
        else:
            obs = last_obs
        if use_gan:
            obs_orig = _obs()
            obs_orig = np.expand_dims(obs_orig, axis=0)
            update_current_obs(obs_orig, shape_dim0, current_obs_orig)
            rollouts.observations_orig[0].copy_(current_obs_orig)

        update_current_obs(obs, shape_dim0, current_obs)
        rollouts.observations[0].copy_(current_obs)

        if args.cuda:
            rollouts.cuda()
            current_obs = current_obs.cuda()
            if use_gan:
                current_obs_orig = current_obs_orig.cuda()
        return rollouts, current_obs, current_obs_orig

    def run_game(self, current_obs, current_obs_orig, collect=False):
        """
        run a full game until its finished
        :param current_obs: the current observations used by the agent
        :param current_obs_orig: the original observations of the target task
        :param collect: True if collecting trajectories, False otherwise
        """
        step = 0
        done = False
        episode_reward = 0.0
        shape_dim0 = self.envs.observation_space.shape[0]

        while not done:
            with torch.no_grad():
                if collect:
                    value, action, action_log_prob, states, _ = self.actor_critic_gan.act(
                        self.rollouts.observations[step],
                        self.rollouts.states[step],
                        self.rollouts.masks[step],
                        deterministic=False)
                else:
                    value, action, action_log_prob, states, _ = self.actor_critic.act(
                        self.rollouts.observations[step],
                        self.rollouts.states[step],
                        self.rollouts.masks[step],
                        deterministic=False)
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = self.envs.step(cpu_actions)
            if collect:
                obs_orig = _obs()
                obs_orig = np.expand_dims(obs_orig, axis=0)

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_reward += reward
            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks
            if collect:
                current_obs_orig *= masks.unsqueeze(2).unsqueeze(2)

            update_current_obs(obs, shape_dim0, current_obs)
            if collect:
                update_current_obs(obs_orig, shape_dim0, current_obs_orig)
                self.rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks, current_obs_orig)
            else:
                self.rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

            if not done:
                step += 1

        return episode_reward, obs

    def collect_trajectories(self):
        # collecting trajectories with source task policy
        logging.info("#STEP 1: Collecting Trajectories")
        t = 0
        while t < args.traj_num:
            ("  #trajectory {}: starting".format(t))
            episode_reward, obs = self.run_game(self.current_obs, self.current_obs_orig, True)
            logging.info("  #trajectory {}: finished".format(t))
            logging.info("  #trajectory {} reward: {}".format(t, episode_reward))
            if episode_reward >= args.det_score * args.traj_coef:
                self.rollouts.compute_returns(0, args.gamma, True)
                self.rollouts.copy_to_db()
                t += 1
            else:
                self.rollouts.step = 0
            self.rollouts.after_update()
        set_gan_off()

        shape_dim0 = self.envs.observation_space.shape[0]
        current_obs = torch.zeros(self.num_processes, *self.obs_shape)

        update_current_obs(obs, shape_dim0, current_obs)
        self.rollouts.observations[0].copy_(current_obs)

        if args.cuda:
            self.rollouts.cuda()
            current_obs = current_obs.cuda()

        self.rollouts.collect_traj = False

        # train a new agent using supervised training
        logging.info("#STEP 2: Supervised Training")
        db_size = self.rollouts.size_db
        for iter in range(args.max_super_iter):
            value_losses = []
            policy_losses = []
            for i in range(0, db_size - args.db_batch_size, args.db_batch_size):
                observation, real_action, returns = self.rollouts.get_item_from_db(i, args.db_batch_size)
                if args.cuda:
                    observation = observation.cuda()
                    real_action = real_action.cuda()
                    returns = returns.cuda()
                value, action, action_log_prob, _, dist_probs = self.actor_critic.act(
                    observation,
                    None,
                    None)
                value_loss, policy_loss = self.agent.supervised_updates(dist_probs, value, real_action, returns)
                value_losses.append(value_loss)
                policy_losses.append(policy_loss)
            logging.info(
                "Iteration: {} Policy loss: {} Value loss: {}".format(iter, np.mean(policy_losses),
                                                                      np.mean(value_losses)))
            self.rollouts.after_update()
        return self.agent, self.actor_critic, self.rollouts


def main():
    torch.set_num_threads(1)

    imitation = ImitationLearning(args)
    agent, actor_critic, rollouts = imitation.collect_trajectories()

    logging.info("#STEP 3: A2C Training")

    envs = [make_env(i, args)
            for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, gamma=args.gamma)

    shape_dim0 = envs.observation_space.shape[0]

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    rollouts.__init__(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    obs = envs.reset()
    update_current_obs(obs, shape_dim0, current_obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()

    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states, _ = actor_critic.act(
                    rollouts.observations[step],
                    rollouts.states[step],
                    rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs, shape_dim0, current_obs)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if args.super_during_rl and j % args.off_policy_interval == 0 and final_rewards.mean() < args.det_score * args.off_policy_coef:
            db_size = rollouts.size_db
            for i in range(db_size - args.db_batch_size):
                observation, real_action, returns = rollouts.get_item_from_db(i, args.db_batch_size)
                if args.cuda:
                    observation = observation.cuda()
                    real_action = real_action.cuda()
                    returns = returns.cuda()
                value, action, action_log_prob, _, dist_probs = actor_critic.act(
                    observation,
                    None,
                    None)
                value_loss, policy_loss = agent.supervised_updates(dist_probs, value, real_action, returns)

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            logging.info(write_logging_info(j, total_num_steps, end, start,
                                            final_rewards, dist_entropy, value_loss, action_loss))
            with open("log_{}.txt".format(args.gan_dir), 'a+') as f:
                f.write(write_logging_info(j, total_num_steps, end, start,
                                           final_rewards, dist_entropy, value_loss, action_loss))


if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(format='%(message)s', filename='trained_models/' + args.log_name, level=logging.INFO)
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.log_dir is not None:
        try:
            os.makedirs(args.log_dir)
        except OSError:
            files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)
    main()
