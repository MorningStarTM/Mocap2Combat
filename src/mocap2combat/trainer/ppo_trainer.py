import os
import torch
import json
import torch.nn as nn
import numpy as np
from collections import deque
from datetime import datetime
from src.mocap2combat.agent.ppo import PPO
from src.mocap2combat.utils.logger import logger
import csv





class PPOTrainer:
    def __init__(self, env, config):
        self.env = env
        self.env_name = "ToriBash"
        self.config = config

        # agent
        self.agent = PPO(
            state_dim=config["state_dim"],
            action_dim=config["action_dim"],
            config=config
        )

        self.best_score = 0.0
        self.score_history = []
        self.episode_rewards = []   # total reward per episode
        self.step_rewards = []      # reward per step
        self.best_ep_reward = -float("inf")
        self.no_improve_epochs = 0

        self.early_stop_patience = config.get("early_stop_patience", 50)  # N episodes
        self.early_stop_min_delta = config.get("early_stop_min_delta", 0.0)

        self.lr_decay_gamma = config.get("lr_decay_gamma", 0.5)  # multiply lr by this
        self.min_lr = config.get("min_lr", 1e-6)

        self.stop_training = False

        # -----------------------------
        # Common root folder for everything
        # -----------------------------
        self.root_dir = os.path.join("runs", self.env_name)
        os.makedirs(self.root_dir, exist_ok=True)

        # Make run folders: run_000, run_001, ...
        run_num = self._next_run_number(self.root_dir)
        self.run_name = f"run_{run_num:03d}"

        self.run_dir = os.path.join(self.root_dir, self.run_name)
        self.models_dir = os.path.join(self.run_dir, "models")
        self.logs_dir = os.path.join(self.run_dir, "model_logs")
        self.rewards_dir = os.path.join(self.run_dir, "rewards")

        for d in (self.run_dir, self.models_dir, self.logs_dir, self.rewards_dir):
            os.makedirs(d, exist_ok=True)

        # log file path (inside model_logs/)
        self.log_f_name = os.path.join(
            self.logs_dir, f"PPO_{self.env_name}_log_{run_num:03d}.csv"
        )

        # reward file path (inside rewards/)
        logger.info(f"Run: {self.run_name}")
        logger.info(f"Run folder: {self.run_dir}")
        logger.info(f"Logging at: {self.log_f_name}")

        # (optional) write CSV headers once
        self._init_csv(self.log_f_name, header=["episode", "timestep", "reward", "done"])

    
    def _init_monitoring(self):
        self.mon_window = int(self.config.get("monitor_window", 100))
        self._mon = {
            "ep_reward": deque(maxlen=self.mon_window),
            "win": deque(maxlen=self.mon_window),
            "smoothness": deque(maxlen=self.mon_window),
            # damage_ratio kept only if we can infer damage from obs later
            "damage_ratio": deque(maxlen=self.mon_window),
        }

        self.metrics_f_name = os.path.join(self.rewards_dir, f"ppo_{self.env_name}_metrics.csv")
        self.metrics_jsonl_name = os.path.join(self.rewards_dir, f"ppo_{self.env_name}_metrics.jsonl")

        # per-episode
        self._prev_action_for_smooth = None
        self._ep_smooth_sum = 0.0
        self._ep_smooth_steps = 0

        # placeholders for damage proxy (optional)
        self._damage_proxy_enabled = False
        self._ep_damage_dealt = 0.0
        self._ep_damage_taken = 0.0
        self._prev_damage_proxy = None  # whatever we decide is "health/injury/score" from obs


    def _open_metrics_files(self):
        self._metrics_csv = open(self.metrics_f_name, "w+", buffering=1)
        self._metrics_csv.write(
            "episode,timestep,ep_reward,win,win_rate_roll,action_smoothness,smooth_roll,"
            "damage_dealt,damage_taken,damage_ratio,damage_ratio_roll\n"
        )
        self._metrics_jsonl = open(self.metrics_jsonl_name, "w+", buffering=1)


    def _close_metrics_files(self):
        try:
            self._metrics_csv.close()
        except Exception:
            pass
        try:
            self._metrics_jsonl.close()
        except Exception:
            pass


    def _reset_episode_monitoring(self):
        self._prev_action_for_smooth = None
        self._ep_smooth_sum = 0.0
        self._ep_smooth_steps = 0

        self._ep_damage_dealt = 0.0
        self._ep_damage_taken = 0.0
        self._prev_damage_proxy = None


    def _as_np_action(self, action):
        try:
            import torch
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
        except Exception:
            pass

        if isinstance(action, (int, float, np.integer, np.floating)):
            return np.asarray([action], dtype=np.float32)
        arr = np.asarray(action, dtype=np.float32)
        return arr.reshape(-1)


    def _update_action_smoothness(self, action):
        a = self._as_np_action(action)
        if self._prev_action_for_smooth is not None:
            da = a - self._prev_action_for_smooth
            self._ep_smooth_sum += float(np.linalg.norm(da))
            self._ep_smooth_steps += 1
        self._prev_action_for_smooth = a


    def _safe_ratio(self, num, den, eps=1e-8):
        return float(num) / float(den + eps)


    def _infer_win_from_rewards(self, current_ep_reward):
        """
        Since info has no outcome, define win using reward.
        Default: win if episode return > 0.
        If your env uses a different scheme, change the threshold ONLY (monitoring only).
        """
        return 1 if float(current_ep_reward) > 0.0 else 0


    def _maybe_enable_damage_proxy_from_obs(self, obs):
        """
        If obs contains a recognizable 'health/injury/score' structure,
        you can enable damage_ratio monitoring. Otherwise it stays disabled.
        This function is conservative: it won't guess wrong automatically.
        """
        # If obs is dict-like, look for obvious keys:
        if isinstance(obs, dict):
            for k in ["health", "hp", "injury", "score", "player_hp", "opponent_hp"]:
                if k in obs:
                    self._damage_proxy_enabled = True
                    return

        # If obs is a flat vector, we can't safely guess indices without your mapping.
        # Keep disabled by default.
        self._damage_proxy_enabled = False


    def _update_damage_proxy(self, obs):
        """
        OPTIONAL: update a damage proxy if obs supports it.
        By default, does nothing unless _damage_proxy_enabled and you define extraction.
        """
        if not self._damage_proxy_enabled:
            return

        # Example for dict observations (edit if your env actually uses these keys)
        if isinstance(obs, dict):
            # lower health / higher injury -> taking damage
            # You MUST map this to your env’s actual obs format if available.
            player = obs.get("player_hp", None)
            opp = obs.get("opponent_hp", None)

            if player is None or opp is None:
                return

            player = float(player)
            opp = float(opp)

            if self._prev_damage_proxy is None:
                self._prev_damage_proxy = (player, opp)
                return

            prev_player, prev_opp = self._prev_damage_proxy
            # if player decreased -> damage taken; if opp decreased -> damage dealt
            self._ep_damage_taken += max(0.0, prev_player - player)
            self._ep_damage_dealt += max(0.0, prev_opp - opp)

            self._prev_damage_proxy = (player, opp)


    def _log_episode_metrics(self, i_episode, time_step, ep_reward, ep_win):
        smooth = self._safe_ratio(self._ep_smooth_sum, max(self._ep_smooth_steps, 1))

        # damage
        dmg_ratio = ""
        dmg_ratio_roll = ""
        dmg_dealt = ""
        dmg_taken = ""

        if self._damage_proxy_enabled:
            dmg_dealt = float(self._ep_damage_dealt)
            dmg_taken = float(self._ep_damage_taken)
            dmg_ratio = self._safe_ratio(self._ep_damage_dealt, self._ep_damage_taken)
            self._mon["damage_ratio"].append(float(dmg_ratio))
            dmg_ratio_roll = float(np.mean(self._mon["damage_ratio"])) if len(self._mon["damage_ratio"]) else float(dmg_ratio)

        self._mon["ep_reward"].append(float(ep_reward))
        self._mon["win"].append(int(ep_win))
        self._mon["smoothness"].append(float(smooth))

        win_rate_roll = float(np.mean(self._mon["win"])) if len(self._mon["win"]) else float(ep_win)
        smooth_roll = float(np.mean(self._mon["smoothness"])) if len(self._mon["smoothness"]) else float(smooth)

        # CSV (empty fields for damage if not enabled)
        self._metrics_csv.write(
            f"{i_episode},{time_step},{float(ep_reward)},{int(ep_win)},{win_rate_roll},"
            f"{smooth},{smooth_roll},{dmg_dealt},{dmg_taken},{dmg_ratio},{dmg_ratio_roll}\n"
        )

        row = {
            "episode": int(i_episode),
            "timestep": int(time_step),
            "ep_reward": float(ep_reward),
            "win": int(ep_win),
            "win_rate_roll": win_rate_roll,
            "action_smoothness": float(smooth),
            "smooth_roll": smooth_roll,
            "damage_proxy_enabled": bool(self._damage_proxy_enabled),
        }
        if self._damage_proxy_enabled:
            row.update({
                "damage_dealt": float(self._ep_damage_dealt),
                "damage_taken": float(self._ep_damage_taken),
                "damage_ratio": float(dmg_ratio),
                "damage_ratio_roll": float(dmg_ratio_roll),
            })

        self._metrics_jsonl.write(json.dumps(row) + "\n")


    @staticmethod
    def _next_run_number(root_dir: str) -> int:
        """
        Counts existing run_* folders and returns the next index.
        """
        if not os.path.isdir(root_dir):
            return 0
        existing = [
            name for name in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, name)) and name.startswith("run_")
        ]
        nums = []
        for name in existing:
            try:
                nums.append(int(name.split("_")[1]))
            except Exception:
                pass
        return (max(nums) + 1) if nums else 0

    @staticmethod
    def _init_csv(path: str, header):
        """
        Create CSV with header if file doesn't exist or is empty.
        """
        need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        if need_header:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)



    def train(self):
        start_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)

        logger.info("============================================================================================")

        # logging file
        log_f = open(self.log_f_name, "a+", buffering=1)
        if not hasattr(self, "_mon"):
            self._init_monitoring()
        self._open_metrics_files()
        

        # printing and logging variables
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0
        
        while time_step <= self.config['max_training_timesteps']:

            state = self.env.reset()
            self._reset_episode_monitoring()
            self._maybe_enable_damage_proxy_from_obs(state)
            current_ep_reward = 0

            for t in range(1, self.config['max_ep_len']+1):

                # select action with policy

                action, *_ = self.agent.select_action(state)
                state, reward, done, info  = self.env.step(action)
                self._update_action_smoothness(action)
                self._update_damage_proxy(state)
                self.step_rewards.append(reward)

                # saving reward and is_terminals
                self.agent.buffer.rewards.append(reward)
                self.agent.buffer.is_terminals.append(done)

                time_step +=1
                current_ep_reward += reward

                # update PPO agent
                if time_step % self.config['update_timestep'] == 0:
                    self.agent.update()
                    self.agent.buffer.clear()


                # log in logging file
                if time_step % self.config['log_freq'] == 0:

                    if log_running_episodes > 0:
                        log_avg_reward = log_running_reward / log_running_episodes
                        log_avg_reward = round(log_avg_reward, 4)
                        log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                        log_f.flush()

                        log_running_reward = 0
                        log_running_episodes = 0

                # printing average reward
                if time_step % self.config['print_freq'] == 0:

                    if print_running_episodes > 0:
                        print_avg_reward = print_running_reward / print_running_episodes
                        print_avg_reward = round(print_avg_reward, 2)

                        logger.info("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                            i_episode, time_step, print_avg_reward
                        ))

                        print_running_reward = 0
                        print_running_episodes = 0

                # save model weights
                if time_step % self.config['save_model_freq'] == 0:
                    logger.info("--------------------------------------------------------------------------------------------")
                    logger.info("saving model at : " + self.models_dir)
                    self.agent.save_safetensors(self.models_dir)
                    logger.info("model saved")
                    logger.info("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    logger.info("--------------------------------------------------------------------------------------------")

                # break; if the episode is over
                if done:
                    break
            
            ep_win = self._infer_win_from_rewards(current_ep_reward)
            self._log_episode_metrics(i_episode=i_episode, time_step=time_step, ep_reward=current_ep_reward, ep_win=ep_win)

            self.episode_rewards.append(current_ep_reward)  
            if current_ep_reward > (self.best_ep_reward + self.early_stop_min_delta):
                self.best_ep_reward = current_ep_reward
                self.no_improve_epochs = 0
            else:
                self.no_improve_epochs += 1

            if self.no_improve_epochs >= self.early_stop_patience:
                lrs = self.agent.decay_lr(self.lr_decay_gamma, self.min_lr)

                logger.info(
                    f"[CALLBACK] No improvement for {self.no_improve_epochs} eps. "
                    f"Decayed LR -> {lrs}"
                )

                # reset counter after decaying (so it can decay again later)
                self.no_improve_epochs = 0

                # optional hard-stop when LR hits min (stop after decay can’t reduce anymore)
                lr_vals = list(lrs.values()) if lrs else []
                if lr_vals and all(lr <= self.min_lr + 1e-12 for lr in lr_vals):
                    logger.info("[CALLBACK] LR reached min_lr. Stopping training.")
                    self.stop_training = True

        
            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

            i_episode += 1
            if self.stop_training:
                break

        log_f.close()
        self._close_metrics_files()
        self.env.close()

        # print total training time
        logger.info("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        logger.info("Started training at (GMT) : ", start_time)
        logger.info("Finished training at (GMT) : ", end_time)
        logger.info("Total training time  : ", end_time - start_time)
        logger.info("============================================================================================")

        np.save(os.path.join(self.rewards_dir, f"ppo_{self.env_name}_step_rewards.npy"), np.array(self.step_rewards))
        np.save(os.path.join(self.rewards_dir, f"ppo_{self.env_name}_episode_rewards.npy"), np.array(self.episode_rewards))
        logger.info(f"Saved step_rewards and episode_rewards to {self.rewards_dir}")