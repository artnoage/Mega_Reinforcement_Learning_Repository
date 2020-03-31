import copy
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from ROAD_FIGHTER.arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from ROAD_FIGHTER.envs import make_env, get_img_counter, set_generator
from ROAD_FIGHTER.model import Policy
from ROAD_FIGHTER.storage import RolloutStorage

import ROAD_FIGHTER.agent as algo

args = get_args()

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def print_gan_log(j, final_rewards, gan_file):
    end = time.time()
    total_num_steps = (j + 1) * args.num_processes * args.num_steps
    template = "GAN iter {}, Updates {}, num timesteps {}, reward {:.1f}"
    print(template.format(gan_file, j, total_num_steps, final_rewards.max()))
    with open("roadfighter_a2c/log_{}.txt".format(args.gan_dir), 'a+') as f:
        f.write("GAN iter {}, Updates {}, num timesteps {}, reward {:.1f}\n".format(gan_file, j, total_num_steps,
                                                                                    final_rewards.max()))


def save_checkpoint(state, filename):
    torch.save(state, filename)


def mdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        print("Directory ", path, " already exists")


def plot_reward_log(logfile, type=['maximum', 'mean']):
    log_data = pd.read_csv(logfile, sep=" ", header=None)
    metrics = {
        'timesteps': [int(m.split(',')[0]) for m in list(log_data[4])],
        'mean': [float(m.split('/')[0]) for m in list(log_data[9])],
        'median': [float(m.split('/')[1].split(',')[0]) for m in list(log_data[9])],
        'max': [float(m.split('/')[1].split(',')[0]) for m in list(log_data[12])],
        'min': [float(m.split('/')[0]) for m in list(log_data[12])]
    }
    plt.style.use('ggplot')
    for t in type:
        plt.plot(metrics['timesteps'], metrics[t])

    plt.title('{}/{} Score in 100M timesteps'.format(type[0], type[1]))
    plt.ylabel('Score')
    plt.xlabel('Time steps')
    plt.show()


def main():
    print("#######")
    print("WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot "
          "to get true rewards")
    print("#######")

    torch.set_num_threads(1)

    if args.test_gan:
        gan_path = args.gan_models_path + args.gan_dir + '/checkpoints'
        files = [os.path.join(gan_path, f).split('_')[1].split('.')[0] for f in os.listdir(gan_path) if
                 os.path.isfile(os.path.join(gan_path, f)) and f.startswith('gen')]
        gan_file = files.pop(0)

        envs = [make_env(i, args, True, gan_file)
                for i in range(args.num_processes)]
    else:
        envs = [make_env(i, args)
                for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs, gamma=args.gamma)

    obs_shape = envs.observation_space.shape
    # observations are taken by 4 RGB images, (12, 63, 64)
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    # this policy in the case of road fighter does not include a recurrent policy, action space is 9
    actor_critic = Policy(obs_shape, envs.action_space, args.recurrent_policy)

    action_shape = 1 if envs.action_space.__class__.__name__ == "Discrete" else envs.action_space.shape[0]

    if args.cuda:
        actor_critic.cuda()

    # Initialize the agent
    agent = algo.A2C(actor_critic, args.value_loss_coef,
                     args.entropy_coef, lr=args.lr,
                     eps=args.eps, alpha=args.alpha,
                     max_grad_norm=args.max_grad_norm)

    if args.load:
        if args.load_dir is None:
            fname = os.path.join(args.save_dir, args.env_name, '.pth.tar')
        else:
            fname = 'trained_models/' + args.env_name + '.pth.tar'
        print(fname)
        if os.path.isfile(fname):
            checkpoint = torch.load(fname, map_location=lambda storage, loc: storage)
            actor_critic.load_state_dict(checkpoint['state_dict'])
            for param in actor_critic.parameters():
                param.requires_grad = True
            print("model loaded")

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            clone_obs = current_obs[:, shape_dim0:].clone()
            current_obs[:, :-shape_dim0] = clone_obs
        current_obs[:, -shape_dim0:] = obs
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    _obs = obs[0] if isinstance(obs, list) else obs
    update_current_obs(_obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])
    total_rewards = torch.zeros([args.num_processes, 1])
    reward = 0.0

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()

    # maximum 100M time steps
    for j in range(num_updates):
        # one entire update is over 4 images in a tensor, 20 number of steps by general update
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states, _ = actor_critic.act(
                    rollouts.observations[step],
                    rollouts.states[step],
                    rollouts.masks[step],
                    deterministic=args.test_gan)
            cpu_actions = action.squeeze(1).cpu().numpy()
            total_rewards += reward

            # Obs reward and next obs
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

            _obs = obs[0] if isinstance(obs, list) else obs
            update_current_obs(_obs)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

            if args.test_gan:
                if done:
                    print_gan_log(j, final_rewards, gan_file)
                    gan_file = files.pop(0)
                    set_generator(gan_file)
                    j = 0

        if not args.test_gan:
            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1],
                                                    rollouts.states[-1],
                                                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.gamma)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if args.save and j % args.save_interval == 0 and args.save_dir != "":
            if j == 0:
                print("Saving model")
                mdir(args.save_dir)

            save_path = args.save_dir
            save_checkpoint({'state_dict': actor_critic.state_dict()}, os.path.join(save_path,
                                                                                    args.env_name + ".pth.tar"))
            save_model = copy.deepcopy(actor_critic).cpu() if args.cuda else actor_critic
            save_model = [save_model, hasattr(envs, 'ob_rms') and envs.ob_rms or None]
            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if args.collect_images:
            if get_img_counter() > args.num_collected_imgs:
                break

        if not args.test_gan and j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            template = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {" \
                       ":.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f} "
            print(template.format(j, total_num_steps, int(total_num_steps / (end - start)),
                                  final_rewards.mean(),
                                  final_rewards.median(),
                                  final_rewards.min(),
                                  final_rewards.max(),
                                  dist_entropy,
                                  value_loss,
                                  action_loss))
            if args.log and args.save:
                with open(os.path.join(args.save_dir, "log_lvl{}.txt".args.level), "a+") as f:
                    template = "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max " \
                               "reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}\n "
                    f.write(template.format(j, total_num_steps,
                                            int(total_num_steps / (end - start)),
                                            final_rewards.mean(),
                                            final_rewards.median(),
                                            final_rewards.min(),
                                            final_rewards.max(), dist_entropy,
                                            value_loss, action_loss))


if __name__ == "__main__":
    main()
