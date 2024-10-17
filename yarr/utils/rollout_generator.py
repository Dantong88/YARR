import os
from multiprocessing import Value
from PIL import Image
import numpy as np
import torch
from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition


class RolloutGenerator(object):

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int,
                  eval: bool, eval_demo_seed: int = 0,
                  record_enabled: bool = False):

        if eval:
            obs = env.reset_to_demo(eval_demo_seed)
        else:
            obs = env.reset()

        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        obs_history['gripper_open'] = []
        obs_history['gripper_pos'] = []

        episode_length = 10
        for step in range(episode_length):
            print(step)
            debug = True
            if debug:
                image = np.transpose(obs_history['front_rgb'][-1], (1, 2, 0))
                image = Image.fromarray(image)
                save_path = '/home/niudt/peract/Processing_data/keyframe_version/annotations_new/keyframes_rlbench_val/{}/{}.png'.format(eval_demo_seed, step)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                image.save(save_path)
            gripper_pos = env._task._scene.get_observation().gripper_pose
            gripper_open = env._task._scene.get_observation().gripper_open
            obs_history['gripper_pos'].append(gripper_pos)
            obs_history['gripper_open'].append(gripper_open)

            # prepped_data = obs_history
            # prepped_data = {k:torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}

            act_result = agent.act(eval_demo_seed, step, obs_history,
                                   deterministic=eval)

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)

            for k in obs_history.keys():
                if k == 'gripper_pos' or k == 'gripper_open':
                    obs_history[k].pop(0)
                else:
                    obs_history[k].append(transition.observation[k])
                    obs_history[k].pop(0)

            transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                print('i am here*')
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            if record_enabled and transition.terminal or timeout or step == episode_length - 1:
                print('i am here****')
                env.env._action_mode.arm_action_mode.record_end(env.env._scene,
                                                                steps=60, step_scene=True)

            obs = dict(transition.observation)
            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
