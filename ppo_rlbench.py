# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# RLBench specific imports
import rlbench.gym
from rlbench.tasks import ReachTarget

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "RLBench"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    task_name: str = "ReachTarget"
    """the name of the RLBench task"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 4
    """the number of parallel evaluation environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 200
    """the number of steps to run in each evaluation environment during evaluation"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

class RLBenchWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "rgb": env.observation_space["wrist_rgb"],
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        })

    def observation(self, obs):
        state = np.concatenate([
            obs["joint_velocities"],
            obs["joint_positions"],
            obs["gripper_open"],
            obs["gripper_pose"],
            obs["gripper_joint_positions"],
            obs["task_low_dim_state"]
        ])
        return {"rgb": obs["wrist_rgb"], "state": state}

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class NatureCNN(nn.Module):
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels=sample_obs["rgb"].shape[-1]
        image_size=(sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])


        # here we use a NatureCNN architecture to process images, but any architecture is permissble here
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # to easily figure out the dimensions after flattening, we pass a test tensor
        with torch.no_grad():
            n_flatten = cnn(torch.as_tensor(sample_obs["rgb"]).float().permute(0,3,1,2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # for state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0,3,1,2)
                obs = obs / 255.0
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    def __init__(self, envs, sample_obs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features
        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.critic(x)

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.task_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    def make_env(task_class, seed):
        def thunk():
            env = rlbench.gym.RLBenchEnv(task_class=task_class, observation_mode='vision')
            env = RLBenchWrapper(env)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env
        return thunk

    task_class = getattr(rlbench.tasks, args.task_name)
    envs = gym.vector.SyncVectorEnv(
        [make_env(task_class, args.seed + i) for i in range(args.num_envs)]
    )
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env(task_class, args.seed + i) for i in range(args.num_eval_envs)]
    )

    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        eval_envs = gym.wrappers.RecordVideo(eval_envs, eval_output_dir, episode_trigger=lambda x: True)

    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=vars(args),
                name=run_name,
                save_code=True,
                group=args.wandb_group,
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    # ALGO Logic: Storage setup
    obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    
    sample_obs = {k: torch.from_numpy(v[0]) for k,v in next_obs.items()}
    agent = Agent(envs, sample_obs=sample_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_action = agent.get_action(eval_obs, deterministic=True)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action)
                    
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v[mask])
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.cat(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(torch.from_numpy(terminations), torch.from_numpy(truncations)).to(device, dtype=torch.float32)
            rewards[step] = torch.from_numpy(reward).to(device).view(-1)

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: logger.close()
