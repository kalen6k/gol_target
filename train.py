# train.py
import os
import argparse
import importlib
import torch
import warnings
import gymnasium as gym
from pathlib import Path
import time
import traceback

warnings.filterwarnings("ignore", "torch.compile")
warnings.filterwarnings("ignore", category=UserWarning, message=".*As shared layers in the mlp_extractor*")


torch.backends.cuda.matmul.allow_tf32 = True

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CallbackList


try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WandbCallback = None
    WANDB_AVAILABLE = False

# Project imports
from agent_model import GOLKeyAgent
from word_env import WordTargetWrapper
from callbacks import LengthCurriculumCallback, SuccessRateCallback, PromptHintCallback
from huggingface_hub import snapshot_download

# --- Default Config ---
DEFAULT_CONFIG = dict(
    # Training Params
    LR          = 3e-4,
    TOTAL_STEPS = 100_000,
    N_ENVS      = 32,
    N_STEPS     = 128,   # Rollout buffer size per env
    BATCH       = 128,   # PPO train batch size
    N_EPOCHS    = 8,     # PPO epochs per update
    GAMMA       = 0.99,
    GAE_LAMBDA  = 0.95,
    CLIP_RANGE  = 0.2,
    ENT_COEF    = 0.0,
    VF_COEF     = 0.5,
    MAX_GRAD_NORM = 0.5,
    # Env/Model Params
    MODEL_DIR   = "Qwen/Qwen2.5-VL-3B-Instruct",
    ENV_MAX_STEPS = 512,
    GRID_SHAPE   = (28, 28),
    WORD_FILE   = "train_words.txt",
    MIN_WORD_LEN= 3,
    MAX_WORD_LEN= 3,
    RESAVE_LIGHT_FROM = None,
    # System/Logging Params
    SAVE_EVERY  = 25_000,
    SEED        = 42,
    DEVICE      = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"),
    TENSORBOARD_LOG = "runs",
    # Curriculum Params
    CURRICULUM_TARGET_SUCCESS = 0.5,
    CURRICULUM_WINDOW = 1000,
    MAX_LEN_LIMIT = 8,
    PROMPT_HINT_TARGET_SUCCESS = 0.75,
)

# --- Configuration Loading ---
def load_config(config_path_override=None):
    """Loads config from defaults, optional file, and CLI args."""
    parser = argparse.ArgumentParser(description="Train GOLKey Agent", add_help=False)
    parser.add_argument("--config", default=config_path_override,
                       help="Python file with UPPER-CASE config overrides.")
    parser.add_argument("--wandb_off", action="store_true", default=False,
                       help="Disable Weights-&-Biases logging.")
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Override TOTAL_STEPS config.")
    parser.add_argument("--resave_light_from", type=str, default=None,
                       help="Path to an existing model.zip to load and resave as inference-only (skips training).")
    # Add other CLI overrides here if needed

    args, _ = parser.parse_known_args()

    cfg_file_vars = {}
    config_to_load = args.config

    if config_to_load and os.path.exists(config_to_load):
        print(f"Loading config from file: {config_to_load}")
        try:
            module_name = config_to_load.replace(".py", "").replace(os.sep, ".")
            if module_name.startswith("."): module_name = module_name[1:]
            mod = importlib.import_module(module_name)
            cfg_file_vars = {k: v for k, v in mod.__dict__.items() if k.isupper()}
        except Exception as e:
            print(f"Warning: Error loading config module '{config_to_load}': {e}")
    elif config_to_load:
        print(f"Warning: Config file '{config_to_load}' not found.")
    else: print("Using default config.")

    final_config = DEFAULT_CONFIG.copy()
    final_config.update(cfg_file_vars)
    if args.total_steps is not None:
        final_config['TOTAL_STEPS'] = args.total_steps
    if args.resave_light_from is not None:
        final_config['RESAVE_LIGHT_FROM'] = args.resave_light_from
    
    class ConfigObject:
        def __init__(self, dictionary):
            self.__dict__.update(dictionary)
        def __getattr__(self, name):
             return self.__dict__.get(name, None)

    config_obj = ConfigObject(final_config)
    config_obj.wandb_off = args.wandb_off
    config_obj.config_file_path = config_to_load

    essential_keys = ['N_ENVS', 'N_STEPS', 'DEVICE', 'TOTAL_STEPS', 'MODEL_DIR', 'LR', 'BATCH', 'MAX_LEN_LIMIT']
    missing = [k for k in essential_keys if getattr(config_obj, k, None) is None]
    if missing:
         raise ValueError(f"Missing essential config keys: {missing}")

    return config_obj

# ---  Feature Extractor ---
class VLMExtractor(BaseFeaturesExtractor):
    """ Custom feature extractor using the GOLKeyAgent's embed method. """
    def __init__(self, observation_space, agent: "GOLKeyAgent", vlm_internal_batch_size: int):
        super().__init__(observation_space, features_dim=agent.intermediate_state)
        self.agent = agent
        self.proj = agent.proj
        self.vlm_internal_batch_size = vlm_internal_batch_size
        print(f"VLMExtractor Initialized: Features Dim = {agent.intermediate_state}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # SB3 handles device placement. Pass obs directly to agent's embed.
        # Assuming obs is (B, C, H, W) due to VecTransposeImage
        #print(f"    VLMExtractor: Entering forward...")
        with torch.no_grad():
            #print(f"    VLMExtractor: Calling agent.embed...")
            raw_features = self.agent.embed(obs, max_batch=self.vlm_internal_batch_size)
            #print(f"    VLMExtractor: Returned from agent.embed.")
        #print(f"    VLMExtractor: Calling self.proj...")
        features = self.proj(raw_features.to(torch.float32))
        #print(f"    VLMExtractor: Exiting forward.")
        return features

# --- Setup Function ---
def setup_model_and_env(config):
    """Initializes VLM agent, environment, and PPO model."""
    print(f"--- Running Setup (Device: {config.DEVICE}) ---")

    model_path = Path(config.MODEL_DIR)
    if not model_path.exists():
        print(f"Downloading {config.MODEL_DIR}...")
        snapshot_download(repo_id=config.MODEL_DIR, local_dir=config.MODEL_DIR,
                          local_dir_use_symlinks=False, resume_download=True)
    else:
         print(f"Model found locally at {config.MODEL_DIR}")

    # VLM Agent
    print("Initializing GOLKeyAgent...")
    vlm_agent = GOLKeyAgent(model_dir=config.MODEL_DIR)
    print(f"GOLKeyAgent initialized on device: {vlm_agent.device}")

    # Create Env
    print(f"Creating {config.N_ENVS} parallel environments...")
    vec_env = make_vec_env(
        lambda: WordTargetWrapper(
            config.WORD_FILE,
            min_len=config.MIN_WORD_LEN,
            max_len=config.MAX_WORD_LEN,
            shape=config.GRID_SHAPE,
            max_steps=config.ENV_MAX_STEPS,
            agent_instance=vlm_agent
        ),
        n_envs = config.N_ENVS,
        seed = config.SEED,
    )
    vec_env = VecTransposeImage(vec_env)
    print(f"Observation space: {vec_env.observation_space.shape}")

    # Policy Kwargs
    vlm_internal_batch = 4
    policy_kwargs = dict(
        features_extractor_class=VLMExtractor,
        features_extractor_kwargs=dict(agent=vlm_agent, vlm_internal_batch_size=vlm_internal_batch),
        net_arch=[dict(pi=[1024, 256, 128, 32],
                       vf=[1024, 256, 128, 32])],
        ortho_init=(config.DEVICE != "cpu"),
    )
    if config.RESAVE_LIGHT_FROM:
        print(f"Loading existing PPO model from: {config.RESAVE_LIGHT_FROM}")
        def get_constant_fn(value: float):
            """Helper function to create a constant schedule."""
            def constant_fn(_progress_remaining: float) -> float:
                return value
            return constant_fn

        lr_schedule_fn = get_constant_fn(config.LR)
        clip_range_fn = get_constant_fn(config.CLIP_RANGE)

        clip_range_vf_fn = get_constant_fn(config.CLIP_RANGE) if config.VF_COEF > 0 else None

        print(f"DEBUG: Using LR={config.LR} and CLIP_RANGE={config.CLIP_RANGE} for potential schedule reconstruction.")

        ppo_model = PPO.load(
            config.RESAVE_LIGHT_FROM,
            env=vec_env,
            device=config.DEVICE,
            custom_objects={
                 "learning_rate": config.LR,
                 "lr_schedule": lambda _: config.LR,
                 "clip_range": lambda _: config.CLIP_RANGE,
                 "lr_schedule": lr_schedule_fn,
                 "clip_range": clip_range_fn,
                 "clip_range_vf": clip_range_vf_fn,
            }
        )
        print("PPO model loaded.")
        print(f"  DEBUG: Loaded ppo_model.lr_schedule: {ppo_model.lr_schedule}")
        if callable(ppo_model.lr_schedule): print(f"    DEBUG: ppo_model.lr_schedule(0.5) evaluation: {ppo_model.lr_schedule(0.5)}")
        else: print(f"    DEBUG: ppo_model.lr_schedule IS NOT CALLABLE.")

        print(f"  DEBUG: Loaded ppo_model.clip_range: {ppo_model.clip_range}")
        if callable(ppo_model.clip_range): print(f"    DEBUG: ppo_model.clip_range(0.5) evaluation: {ppo_model.clip_range(0.5)}")
        else: print(f"    DEBUG: ppo_model.clip_range IS NOT CALLABLE.")

        if hasattr(ppo_model, "clip_range_vf"):
            print(f"  DEBUG: Loaded ppo_model.clip_range_vf: {ppo_model.clip_range_vf}")
            if ppo_model.clip_range_vf is not None and callable(ppo_model.clip_range_vf): print(f"    DEBUG: ppo_model.clip_range_vf(0.5) evaluation: {ppo_model.clip_range_vf(0.5)}")
            elif ppo_model.clip_range_vf is not None: print(f"    DEBUG: ppo_model.clip_range_vf IS NOT CALLABLE (but not None).")
            else: print(f"    DEBUG: ppo_model.clip_range_vf IS None.")
        else: print("  DEBUG: ppo_model does not have attribute clip_range_vf.")
        if ppo_model.lr_schedule is None or not callable(ppo_model.lr_schedule):
            print(f"WARNING: ppo_model.lr_schedule is {ppo_model.lr_schedule}. This might be an issue.")
        if ppo_model.clip_range is None or not callable(ppo_model.clip_range):
            print(f"WARNING: ppo_model.clip_range is {ppo_model.clip_range}. This might be an issue.")
    else:
        # PPO Model
        print("Initializing PPO model...")
        ppo_model = PPO(
            "CnnPolicy",
            vec_env,
            learning_rate = config.LR,
            n_steps       = config.N_STEPS,
            batch_size    = config.BATCH,
            n_epochs      = config.N_EPOCHS,
            gamma         = config.GAMMA,
            gae_lambda    = config.GAE_LAMBDA,
            clip_range    = config.CLIP_RANGE,
            ent_coef      = config.ENT_COEF,
            vf_coef       = config.VF_COEF,
            max_grad_norm = config.MAX_GRAD_NORM,
            policy_kwargs = policy_kwargs,
            device        = config.DEVICE,
            verbose       = 1,
            tensorboard_log= config.TENSORBOARD_LOG,
            seed          = config.SEED
        )
        print("PPO model initialized.")
    print("--- Setup Complete ---")
    return vlm_agent, vec_env, ppo_model

# --- Main Execution Block ---
if __name__ == "__main__":
    C = load_config()

    print(f"--- Starting Training Run ---")
    print(f"Config File: {C.config_file_path or 'Defaults'}")
    print(f"Device: {C.DEVICE}, Seed: {C.SEED}")
    print(f"Total Steps: {C.TOTAL_STEPS:,}")
    print(f"Env: N={C.N_ENVS}, Steps/Rollout={C.N_STEPS}, Buffer={C.N_ENVS * C.N_STEPS:,}")
    print(f"PPO: Batch={C.BATCH:,}, LR={C.LR}")
    print(f"Wandb Logging: {'Disabled' if C.wandb_off or not WANDB_AVAILABLE else 'Enabled'}")
    print(f"-----------------------------")

    # Setup Wandb
    run = None
    use_wandb = WANDB_AVAILABLE and not C.wandb_off
    if use_wandb:
        try:
            run_name = f"ppo_curric_{Path(C.config_file_path).stem if C.config_file_path else 'defaults'}_{int(time.time())}"
            run = wandb.init(
                project="golkey-qwen2_5_vl_3B-stage_gpu_1.0",
                name=run_name,
                config=C.__dict__, # Log the final effective config
                sync_tensorboard=True,
                monitor_gym=False,
                save_code=True,        # Save main script to wandb
            )
            print(f"Wandb initialized. Run Name: {run.name}, ID: {run.id}")
        except Exception as e:
            print(f"Wandb initialization failed: {e}. Disabling Wandb for this run.")
            use_wandb = False
    elif not WANDB_AVAILABLE:
         print("Wandb not installed, skipping.")
    else:
         print("Wandb disabled via --wandb_off flag.")

    # Setup agent, env, model
    vlm_agent, vec_env, model = setup_model_and_env(C)

    # Setup callbacks
    callbacks = []
    if not C.RESAVE_LIGHT_FROM:
        if use_wandb and run is not None:
            save_path = f"models/{run.name}"
            os.makedirs(save_path, exist_ok=True)
            print(f"Adding WandbCallback (Save Freq: {C.SAVE_EVERY}, Path: {save_path})")
            callbacks.append(WandbCallback(model_save_freq=C.SAVE_EVERY,
                                            model_save_path=save_path,
                                            verbose=1))
            pass
        pass

        print("Adding LengthCurriculumCallback.")
        callbacks.append(LengthCurriculumCallback(target_success=C.CURRICULUM_TARGET_SUCCESS,
                                                window=C.CURRICULUM_WINDOW,
                                                max_len_limit=C.MAX_LEN_LIMIT,
                                                verbose=1))
        
        print("Adding SuccessRateCallback.")
        callbacks.append(SuccessRateCallback(window_size=24, verbose=1))

        print("Adding PromptHintCallback.")
        callbacks.append(PromptHintCallback(target_success=C.PROMPT_HINT_TARGET_SUCCESS,
                                            window=C.CURRICULUM_WINDOW,
                                            max_len_limit=C.MAX_LEN_LIMIT,
                                            verbose=1))

    # --- Start training ---
    start_time = time.time()
    if C.RESAVE_LIGHT_FROM:
        print(f"\n--- Resaving model from {C.RESAVE_LIGHT_FROM} ---")
        print("Skipping training.")
        final_model_name = Path(C.RESAVE_LIGHT_FROM).stem
    else:
        print(f"\nStarting training for {C.TOTAL_STEPS:,} total timesteps...")
        final_model_name = "stage_gpu_1.0"
        try:
            model.learn(
                total_timesteps=C.TOTAL_STEPS,
                callback=CallbackList(callbacks) if callbacks else None,
                log_interval=1
            )
            print("\n--- Training finished successfully ---")
        except torch.cuda.OutOfMemoryError:
            print("!!! CAUGHT CUDA OOM ERROR !!!")
            traceback.print_exc()
            final_model_name = "final_model_oom"
        except RuntimeError as e:
            print(f"!!! CAUGHT RUNTIME ERROR: {e} !!!")
            if "CUDA" in str(e):
                print("Looks like a CUDA Runtime error.")
            traceback.print_exc()
            final_model_name = "final_model_runtime_error"
        except KeyboardInterrupt:
            print("\n--- Training Interrupted by User ---")
            final_model_name = "final_model_interrupted"
        except Exception as e:
            print(f"\n--- Training Error: {type(e).__name__}: {e} ---")
            traceback.print_exc()
            final_model_name = "final_model_error"
        except Exception as e:
            print(f"\n--- Training Error: {type(e).__name__}: {e} ---")
            traceback.print_exc()
            final_model_name = "final_model_error"
    try:
        if C.RESAVE_LIGHT_FROM:
            output_path_stem = Path(C.RESAVE_LIGHT_FROM).stem
            output_dir = Path(C.RESAVE_LIGHT_FROM).parent
            resaved_inference_path = output_dir / f"{output_path_stem}_inference_resaved.zip"
            print(f"\nResaving inference-only model to: {resaved_inference_path}")
            model.save(resaved_inference_path, exclude=["rollout_buffer"])
            print("Inference-only model resaved.")
        else:
            save_dir = Path(model.tensorboard_log)
            base_filename_stem = f"{final_model_name}_{C.TOTAL_STEPS}steps"

            full_model_path = save_dir / f"{base_filename_stem}_full.zip"
            print(f"\nSaving full final model (with rollout buffer) to: {full_model_path}")
            model.save(full_model_path)
            print("Full final model saved.")

            inference_model_path = save_dir / f"{base_filename_stem}_inference.zip"
            print(f"\nSaving inference-only final model (without rollout buffer) to: {inference_model_path}")
            model.save(inference_model_path, exclude=["rollout_buffer"])
            print("Inference-only final model saved.")
    except Exception as save_e:
        print(f"Error during model saving/resaving: {save_e}")
        traceback.print_exc()
    
    finally:
        end_time = time.time()
        if not C.RESAVE_LIGHT_FROM:
            print(f"Total training time: {end_time - start_time:.2f} seconds")

        print("Closing environment...")
        try:
            vec_env.close()
        except Exception as close_e:
             print(f"Error closing environment: {close_e}")

        # Finish wandb run
        if use_wandb and run is not None:
             print("Finishing wandb run...")
             wandb.finish()
             print("Wandb run finished.")

    print("--- Run Script Complete ---")