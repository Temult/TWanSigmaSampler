# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
import gc
# Assuming these imports are correct relative to your custom node structure
# Make sure utils.py is in the same directory
from .utils import log, print_memory, apply_lora
import numpy as np
import math
from tqdm import tqdm

# Make sure wanvideo folder is in the same directory
from .wanvideo.modules.clip import CLIPModel
from .wanvideo.modules.model import WanModel, rope_params
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

# Make sure enhance_a_video folder is in the same directory
from .enhance_a_video.globals import enable_enhance, disable_enhance, set_enhance_weight, set_num_frames
# Make sure taehv.py is in the same directory
from .taehv import TAEHV

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.model_base
from comfy.cli_args import args, LatentPreviewMethod
# Make sure context.py is in the same directory
from .context import get_context_scheduler
# Make sure latent_preview.py is in the same directory (or use comfy's)
prepare_callback = None # Initialize prepare_callback to None
try:
    if args.preview_method in [LatentPreviewMethod.Auto, LatentPreviewMethod.Latent2RGB]:
        from latent_preview import prepare_callback
    else:
        from .latent_preview import prepare_callback # custom for tiny VAE previews
except ImportError:
    log.warning("Could not import latent_preview, previews disabled.")
    # prepare_callback remains None


# Helper function from original code (ensure it's available)
def optimized_scale(positive_flat, negative_flat):
    # Calculate dot production
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    # Squared norm of uncondition
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm
    return st_star

# --- WindowTracker Class (ensure it's defined or imported) ---
class WindowTracker:
    def __init__(self, verbose=False):
        self.window_map = {}  # Maps frame sequence tuple to persistent ID
        self.next_id = 0
        # Stores teacache state ([cond_state, uncond_state]) per window ID
        self.teacache_states = {}
        self.verbose = verbose

    def get_window_id(self, frames):
        # Use tuple of frames directly as key assumes order matters for context
        key = tuple(frames)
        if key not in self.window_map:
            self.window_map[key] = self.next_id
            if self.verbose:
                log.info(f"New window pattern {key} -> ID {self.next_id}")
            self.next_id += 1
        return self.window_map[key]

    def get_teacache(self, window_id, base_state):
        if window_id not in self.teacache_states:
            if self.verbose:
                log.info(f"Initializing persistent teacache for window {window_id}")
            # Important: Create a *copy* of the base state, not a reference
            self.teacache_states[window_id] = list(base_state) if isinstance(base_state, list) else base_state # Shallow copy for list
        return self.teacache_states[window_id]
# --- End WindowTracker ---


# region Sampler
class TWanVideoSigmaSampler:
    @classmethod
    def INPUT_TYPES(s):
        # --- No changes needed here ---
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": (
                "BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (["unipc", "dpm++", "dpm++_sde", "euler", "euler/beta"],
                              {
                                  "default": 'unipc'
                              }),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                                              "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),

            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feta_args": ("FETAARGS",),
                "context_options": ("WANVIDCONTEXT",),
                "teacache_args": ("TEACACHEARGS",),
                "flowedit_args": ("FLOWEDITARGS",),
                "batched_cfg": ("BOOLEAN", {"default": False,
                                            "tooltip": "Batc cond and uncond for faster sampling, possibly faster on some hardware, uses more memory"}),
                "slg_args": ("SLGARGS",),
                "rope_function": (["default", "comfy"], {"default": "comfy",
                                                         "tooltip": "Comfy's RoPE implementation doesn't use complex numbers and can thus be compiled, that should be a lot faster when using torch.compile"}),
                "loop_args": ("LOOPARGS",),
                "experimental_args": ("EXPERIMENTALARGS",),
                # --- ADD SIGMAS INPUT ---
                "sigmas": ("SIGMAS", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    # Make sure CATEGORY matches where you want it in ComfyUI
    CATEGORY = "TWanVideoWrapper" # Or keep "WanVideoWrapper" if preferred

    def process(self, model, text_embeds, image_embeds, shift, steps, cfg, seed, scheduler, riflex_freq_index,
                force_offload=True, samples=None, feta_args=None, denoise_strength=1.0, context_options=None,
                teacache_args=None, flowedit_args=None, batched_cfg=False, slg_args=None, rope_function="default",
                loop_args=None, experimental_args=None, sigmas=None): # Added sigmas=None
        # assert not (context_options and teacache_args), "Context options cannot currently be used together with teacache."
        patcher = model
        model = model.model
        transformer = model.diffusion_model

        control_lora = model["control_lora"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        # --- Initialize potentially problematic variables ---
        sample_scheduler = None
        block_swap_args = None
        context_frames = 0
        context_stride = 0
        context_overlap = 0
        context_vae = None
        context = None # Function/generator
        create_window_mask = None # Function
        mask = None # Tensor or None
        original_image = None # Tensor or None
        skip_steps = 0
        latent_shift_start_percent = 0.0
        latent_shift_end_percent = 1.0
        shift_idx = 0
        latent_skip = 1 # Default value for latent_skip if loop_args is None
        feta_start_percent = 0.0
        feta_end_percent = 1.0
        x_init = None
        drift_steps = 0
        drift_cfg = [] # Use list for CFG schedules
        source_cfg = [] # Use list for CFG schedules
        x_tgt = None
        section_size = 0.0
        num_prompts = 1
        source_embeds = None
        source_image_cond = None
        source_clip_fea = None
        image_index = 0 # Default index
        control_start_percent = 0.0 # Default for control latents if used
        control_end_percent = 1.0 # Default for control latents if used
        # --- End Initialization ---

        # --- Sigma/Scheduler Logic ---
        if sigmas is not None:
            log.info("Using provided sigmas tensor.")
            sigmas = sigmas.to(device=device, dtype=torch.float32)
            steps = len(sigmas) -1 # Number of steps is len(sigmas) - 1
            if steps < 0: raise ValueError("Sigmas tensor must contain at least 2 values.")
            # Derive timesteps (common convention, might need adjustment for specific models)
            # Ensure sigmas are clamped to avoid issues with log/sqrt
            clamped_sigmas = torch.clamp(sigmas, min=1e-4) # Adjust min value if needed
            timesteps = (clamped_sigmas * 1000.0).round().long() # Example scaling

            # We still need a scheduler instance for its .step() method's formula.
            # Let's default to EulerDiscrete as it's simple.
            # Its internal schedule won't be used.
            scheduler_args = {
                "num_train_timesteps": 1000,  # Keep this, might be needed internally
                # "beta_start": 0.00085,       # REMOVE
                # "beta_end": 0.012,           # REMOVE
                # "beta_schedule": "scaled_linear" # REMOVE
            }
            # It might even work with no arguments if num_train_timesteps isn't strictly needed
            # when sigmas are manually set later. If the above fails, try:
            # sample_scheduler = FlowMatchEulerDiscreteScheduler()
            sample_scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_args)
            # Manually set the sigmas and timesteps on the instance
            sample_scheduler.sigmas = sigmas
            sample_scheduler.timesteps = timesteps
            # Set the number of inference steps
            sample_scheduler.set_timesteps(steps, device=device) # This recalculates internal sigmas/timesteps, but we override below
            sample_scheduler.sigmas = sigmas # Override again after set_timesteps
            sample_scheduler.timesteps = timesteps # Override again
            sample_scheduler.num_inference_steps = steps

            # Handle denoise strength with sigmas
            if denoise_strength < 1.0:
                 t_start = steps - int(round(steps * denoise_strength))
                 # Ensure t_start is valid index for slicing (needs at least one step)
                 t_start = max(0, min(t_start, steps)) # Clamp between 0 and steps
                 sigmas = sigmas[t_start:]
                 timesteps = timesteps[t_start:]
                 steps = len(sigmas) - 1 # Update steps count
                 if steps < 0: raise ValueError("Denoise strength resulted in zero steps.")
                 # Re-assign to scheduler
                 sample_scheduler.sigmas = sigmas
                 sample_scheduler.timesteps = timesteps
                 sample_scheduler.num_inference_steps = steps
                 log.info(f"Denoising using {steps+1} sigmas ({steps} steps) from provided schedule (Denoise: {denoise_strength})")

        else:
            # --- Existing scheduler logic ---
            log.info("Generating sigmas internally using scheduler settings.")
            # Calculate original total steps before denoise slicing
            original_total_steps = steps
            # Apply denoise strength early to get the number of steps for schedule generation
            steps_for_schedule = int(original_total_steps / denoise_strength) if denoise_strength > 0 else 0
            if steps_for_schedule < 1: steps_for_schedule = 1 # Ensure at least 1 step

            scheduler_args = {
                "num_train_timesteps": 1000,
                "shift": shift,
                "use_dynamic_shifting": False,
            }

            timesteps = None # Reset timesteps here
            if scheduler == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(**scheduler_args)
                # Use steps_for_schedule here
                sample_scheduler.set_timesteps(steps_for_schedule, device=device, shift=shift)
            elif scheduler in ['euler/beta', 'euler']:
                sample_scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_args,
                                                                   use_beta_sigmas=(scheduler == 'euler/beta'))
                if flowedit_args:  # seems to work better
                    # Use steps_for_schedule here
                    timesteps, _ = retrieve_timesteps(sample_scheduler, device=device,
                                                      sigmas=get_sampling_sigmas(steps_for_schedule, shift))
                else:
                     # Use steps_for_schedule here
                    sample_scheduler.set_timesteps(steps_for_schedule, device=device, mu=1)
            elif 'dpm++' in scheduler:
                if scheduler == 'dpm++_sde':
                    algorithm_type = "sde-dpmsolver++"
                else:
                    algorithm_type = "dpmsolver++"
                sample_scheduler = FlowDPMSolverMultistepScheduler(**scheduler_args, algorithm_type=algorithm_type)
                 # Use steps_for_schedule here
                sample_scheduler.set_timesteps(steps_for_schedule, device=device, mu=1)
            else:
                 raise ValueError(f"Unsupported scheduler: {scheduler}")

            if timesteps is None:
                timesteps = sample_scheduler.timesteps
                # Also get sigmas if not already calculated (needed for step function)
                if not hasattr(sample_scheduler, 'sigmas') or sample_scheduler.sigmas is None or len(sample_scheduler.sigmas) != len(timesteps):
                     # Attempt to get sigmas from scheduler if possible, else derive
                     if hasattr(sample_scheduler, 'sigmas') and sample_scheduler.sigmas is not None:
                         sigmas = sample_scheduler.sigmas
                     else:
                         sigmas = sample_scheduler.timesteps / 1000.0 # Approximation if not present
                         sample_scheduler.sigmas = sigmas # Store derived sigmas

            # Now apply denoise strength slicing
            if denoise_strength < 1.0:
                effective_steps = int(round(original_total_steps * denoise_strength))
                if effective_steps < 1: effective_steps = 1
                # Calculate start index for slicing (keep `effective_steps` steps + 1 for sigmas/timesteps)
                start_index = len(timesteps) - (effective_steps + 1)
                if start_index < 0: start_index = 0

                timesteps = timesteps[start_index:]
                # Slice sigmas too if they exist on the scheduler and match length
                if hasattr(sample_scheduler, 'sigmas') and sample_scheduler.sigmas is not None and len(sample_scheduler.sigmas) == len(timesteps) + (len(sample_scheduler.sigmas) - len(timesteps)): # Check if sigmas length matches original timesteps
                    sigmas = sample_scheduler.sigmas[start_index:]
                    sample_scheduler.sigmas = sigmas # Update scheduler instance
                else:
                    # If sigmas weren't explicitly generated or length mismatch, derive from sliced timesteps
                    sigmas = timesteps / 1000.0 # Approximation
                    sample_scheduler.sigmas = sigmas # Store on scheduler

                steps = len(timesteps) - 1 # Update actual steps count
                if steps < 0: raise ValueError("Denoise strength resulted in zero steps.")
                sample_scheduler.num_inference_steps = steps # Update scheduler instance
                sample_scheduler.timesteps = timesteps # Update scheduler instance
                log.info(f"Denoising using {steps+1} sigmas ({steps} steps) (Denoise: {denoise_strength})")
            else:
                steps = len(timesteps) - 1 # Full steps
                if steps < 0: raise ValueError("Schedule resulted in zero steps.")
                if hasattr(sample_scheduler, 'sigmas') and sample_scheduler.sigmas is not None and len(sample_scheduler.sigmas) == len(timesteps):
                    sigmas = sample_scheduler.sigmas
                else:
                    sigmas = timesteps / 1000.0
                    sample_scheduler.sigmas = sigmas
                log.info(f"Denoising using {steps+1} sigmas ({steps} steps) (Full Denoise)")
        # --- End Sigma/Scheduler Logic ---


        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)

        # Initialize these based on image_embeds type
        control_latents, clip_fea, clip_fea_neg, end_image = None, None, None, None
        vace_data, vace_context, vace_scale = None, None, None
        fun_model, has_ref, drop_last = False, False, False

        image_cond = image_embeds.get("image_embeds", None)

        # --- Noise Generation & Shape Determination ---
        if image_cond is not None: # I2V or ControlNet case
            end_image = image_embeds.get("end_image", None)
            lat_h = image_embeds.get("lat_h", None)
            lat_w = image_embeds.get("lat_w", None)
            if lat_h is None or lat_w is None:
                raise ValueError("Image embeds must provide lat_h and lat_w for I2V/ControlNet")
            fun_model = image_embeds.get("fun_model", False)
            num_latent_frames = (image_embeds["num_frames"] - 1) // 4 + (2 if end_image is not None and not fun_model else 1)

            noise_shape = (16, num_latent_frames, lat_h, lat_w)
            noise = torch.randn(noise_shape, dtype=torch.float32, generator=seed_g, device=torch.device("cpu"))

            seq_len = image_embeds["max_seq_len"]
            clip_fea = image_embeds.get("clip_context", None)
            clip_fea_neg = image_embeds.get("negative_clip_context", None)

            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                # Assuming control_embeds structure is correct
                control_latents = control_embeds["control_images"].to(device)
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
                # Validate dimensions if needed
                if transformer.in_dim != 48 and not control_lora: # Allow control lora with different dims
                    log.warning("Control signal might not work correctly with non Fun-Control model unless using Control-LoRA")
            drop_last = image_embeds.get("drop_last", False)
            has_ref = image_embeds.get("has_ref", False)

        else: # T2V case
            target_shape = image_embeds.get("target_shape", None)
            if target_shape is None:
                raise ValueError("Empty image embeds must be provided for T2V (Text to Video)")

            has_ref = image_embeds.get("has_ref", False)
            vace_context = image_embeds.get("vace_context", None)
            vace_scale = image_embeds.get("vace_scale", None)
            vace_start_percent = image_embeds.get("vace_start_percent", 0.0)
            vace_end_percent = image_embeds.get("vace_end_percent", 1.0)

            vace_additional_embeds = image_embeds.get("additional_vace_inputs", [])
            if vace_context is not None:
                vace_data = [
                    {"context": vace_context, "scale": vace_scale, "start": vace_start_percent, "end": vace_end_percent}
                ]
                # Simplified loop
                vace_data.extend(vace_additional_embeds)

            noise_shape = (
                    target_shape[0],
                    target_shape[1] + 1 if has_ref else target_shape[1],
                    target_shape[2],
                    target_shape[3],
            )
            noise = torch.randn(noise_shape, dtype=torch.float32, device=torch.device("cpu"), generator=seed_g)

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])

            control_embeds = image_embeds.get("control_embeds", None)
            if control_embeds is not None:
                control_latents = control_embeds["control_images"].to(device)
                control_start_percent = control_embeds.get("start_percent", 0.0)
                control_end_percent = control_embeds.get("end_percent", 1.0)
                if control_lora:
                    image_cond = control_latents.to(device) # Use control latents as image_cond for LoRA
                    # LoRA loading logic might need adjustment depending on how patcher handles it
                else:
                    if transformer.in_dim != 48:
                         log.warning("Control signal might not work correctly with non Fun-Control model")
                    # For non-LoRA control, image_cond might be zeros or specific input
                    image_cond = torch.zeros_like(control_latents).to(device) # Example for Fun-Control
                    clip_fea = None # Typically no separate clip features for T2V control

            elif transformer.in_dim == 36: # fun inp case for T2V without control
                mask_latents = torch.tile(torch.zeros_like(noise[:1]), [4, 1, 1, 1])
                masked_video_latents_input = torch.zeros_like(noise)
                image_cond = torch.cat([mask_latents, masked_video_latents_input], dim=0).to(device)
        # --- End Noise Generation ---

        latent_video_length = noise.shape[1]

        is_looped = False
        # Define create_window_mask here if needed by flowedit_args even when context_options is None
        if context_options is not None or flowedit_args is not None: # Define if either needs it
            def create_window_mask(noise_pred_context, c, latent_video_length, context_overlap, looped=False):
                window_mask = torch.ones_like(noise_pred_context)
                # Apply left-side blending
                if min(c) > 0 or (looped and max(c) == latent_video_length - 1):
                    ramp_up = torch.linspace(0, 1, context_overlap, device=noise_pred_context.device).view(1, -1, 1, 1)
                    window_mask[:, :context_overlap] = ramp_up
                # Apply right-side blending
                if max(c) < latent_video_length - 1 or (looped and min(c) == 0):
                    ramp_down = torch.linspace(1, 0, context_overlap, device=noise_pred_context.device).view(1, -1, 1, 1)
                    window_mask[:, -context_overlap:] = ramp_down
                return window_mask

        if context_options is not None:
            context_schedule = context_options["context_schedule"]
            context_frames = (context_options["context_frames"] - 1) // 4 + 1
            context_stride = context_options["context_stride"] // 4
            context_overlap = context_options["context_overlap"] // 4
            context_vae = context_options.get("vae", None)
            if context_vae is not None:
                context_vae.to(device)

            # Ensure WindowTracker is defined or imported correctly
            self.window_tracker = WindowTracker(verbose=context_options["verbose"])

            num_prompts = len(text_embeds["prompt_embeds"])
            log.info(f"Number of prompts: {num_prompts}")
            section_size = latent_video_length / num_prompts if num_prompts > 0 else latent_video_length
            log.info(f"Section size: {section_size}")
            is_looped = context_schedule == "uniform_looped"

            # Adjust seq_len for context windowing
            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * context_frames)

            if context_options["freenoise"]:
                log.info("Applying FreeNoise")
                # ... (FreeNoise logic as before) ...
                delta = context_frames - context_overlap
                if delta > 0: # Avoid division by zero or infinite loop if overlap >= frames
                    for start_idx in range(0, latent_video_length - context_frames, delta):
                        # ... (rest of FreeNoise logic)
                        place_idx = start_idx + context_frames
                        if place_idx >= latent_video_length:
                            break
                        end_idx = place_idx - 1

                        if end_idx + delta >= latent_video_length:
                            final_delta = latent_video_length - place_idx
                            if final_delta > 0:
                                list_idx = torch.tensor(list(range(start_idx, start_idx + final_delta)), device=torch.device("cpu"), dtype=torch.long)
                                list_idx = list_idx[torch.randperm(final_delta, generator=seed_g)]
                                noise[:, place_idx:place_idx + final_delta, :, :] = noise[:, list_idx, :, :]
                            break
                        else:
                             if delta > 0:
                                list_idx = torch.tensor(list(range(start_idx, start_idx + delta)), device=torch.device("cpu"), dtype=torch.long)
                                list_idx = list_idx[torch.randperm(delta, generator=seed_g)]
                                noise[:, place_idx:place_idx + delta, :, :] = noise[:, list_idx, :, :]

            log.info(f"Context schedule enabled: {context_frames} frames, {context_stride} stride, {context_overlap} overlap")
            # Ensure context module is imported correctly
            # from .context import get_context_scheduler # Already imported above
            context = get_context_scheduler(context_schedule)
        # --- End context_options setup ---

        # --- Initial Latent Setup ---
        if samples is not None: # V2V or FlowEdit init
            original_image = samples["samples"].clone().squeeze(0).to(device) # Used for V2V denoise and diff-diff
            mask = samples.get("mask", None) # Used for diff-diff

            if denoise_strength < 1.0:
                # Noise up the initial image based on the *first* timestep of the *final* schedule
                noise_timestep_index = 0 # Index corresponding to the highest noise level we start at
                if noise_timestep_index < len(timesteps):
                    latent_timestep = timesteps[noise_timestep_index:noise_timestep_index+1].to(noise.device, noise.dtype)
                    # A more common approach using scheduler's add_noise:
                    noise_for_init = torch.randn_like(original_image, device=noise.device) # Use noise device
                    init_latent = sample_scheduler.add_noise(original_image, noise_for_init, timesteps[noise_timestep_index:noise_timestep_index+1])
                    latent = init_latent.to(device) # Ensure it's on the main device
                    log.info(f"Applied V2V noise for denoise strength {denoise_strength} at timestep {latent_timestep.item()}")
                else:
                    log.warning("Could not apply V2V noise: schedule too short.")
                    latent = noise.to(device) # Fallback to pure noise
            else:
                 # If denoise is 1.0, but samples provided, maybe start from pure noise?
                 # Or should it ignore samples? Let's assume start from noise.
                 log.info("Denoise is 1.0, starting from noise even though initial samples were provided.")
                 latent = noise.to(device)
        else: # T2V
            latent = noise.to(device)
        # --- End Initial Latent Setup ---


        # --- RoPE Setup ---
        freqs = None
        # Ensure rope_embedder exists and has expected attributes
        if hasattr(transformer, 'rope_embedder'):
            transformer.rope_embedder.k = None
            transformer.rope_embedder.num_frames = None
            if rope_function == "comfy":
                transformer.rope_embedder.k = riflex_freq_index
                transformer.rope_embedder.num_frames = latent_video_length
            else: # Default RoPE calculation
                # Check necessary attributes exist
                if hasattr(transformer, 'dim') and hasattr(transformer, 'num_heads'):
                    d = transformer.dim // transformer.num_heads
                    # Ensure rope_params is imported/defined
                    try:
                        freqs = torch.cat([
                            rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index),
                            rope_params(1024, 2 * (d // 6)),
                            rope_params(1024, 2 * (d // 6))
                        ], dim=1)
                    except NameError:
                        log.error("rope_params function not found for RoPE calculation.")
                        freqs = None
                else:
                    log.warning("Transformer dimensions not found for default RoPE calculation.")
        else:
            log.warning("Transformer does not have rope_embedder attribute.")
        # --- End RoPE Setup ---

        # --- CFG Schedule Setup ---
        # Store the original float cfg value
        base_cfg_value = cfg
        # Create the schedule list
        cfg_schedule = [base_cfg_value] * (steps + 1) # Use actual steps count
        # --- End CFG Schedule Setup ---

        print("Seq len:", seq_len)
        pbar = ProgressBar(steps)

        # --- Callback Setup ---
        callback = None
        # FIX: Check if prepare_callback is callable before calling it
        if prepare_callback is not None and callable(prepare_callback):
            try:
                callback = prepare_callback(patcher, steps)
            except Exception as e:
                log.warning(f"Failed to prepare callback: {e}. Preview disabled.")
        else:
            log.warning("prepare_callback not available or not callable. Preview disabled.")
        # --- End Callback Setup ---

        # --- Block Swap Init ---
        transformer_options = patcher.model_options.get("transformer_options", {}) # Use {} default
        block_swap_args = transformer_options.get("block_swap_args", None) # Get block_swap_args again safely

        if block_swap_args is not None:
            # ... (block swap logic as before) ...
            transformer.use_non_blocking = block_swap_args.get("use_non_blocking", True)
            offload_txt = block_swap_args.get("offload_txt_emb", False)
            offload_img = block_swap_args.get("offload_img_emb", False)
            blocks_to_swap_count = block_swap_args.get("blocks_to_swap", 0)
            vace_blocks_to_swap_count = block_swap_args.get("vace_blocks_to_swap", None)

            # Offload parameters before swapping if specified
            for name, param in transformer.named_parameters():
                if "block" not in name: # Keep non-block params on main device initially
                    param.data = param.data.to(device)
                elif offload_txt and "txt_emb" in name:
                    param.data = param.data.to(offload_device, non_blocking=transformer.use_non_blocking)
                elif offload_img and "img_emb" in name:
                    param.data = param.data.to(offload_device, non_blocking=transformer.use_non_blocking)
                # Other block params remain on their current device (likely main) for now

            # Perform the swap (assuming block_swap method exists and handles device placement)
            if hasattr(transformer, 'block_swap') and blocks_to_swap_count > 0:
                transformer.block_swap(
                    blocks_to_swap_count - 1, # Adjust index if needed
                    offload_txt,
                    offload_img,
                    vace_blocks_to_swap = vace_blocks_to_swap_count,
                )
            else:
                log.warning("Transformer has no block_swap method or blocks_to_swap is 0.")

        elif patcher.model_options.get("auto_cpu_offload", False):  # Check key safely using patcher.model_options
            log.info("Using auto_cpu_offload.")
            for module in transformer.modules():
                if hasattr(module, "offload"): module.offload()
                if hasattr(module, "onload"): module.onload()
        elif patcher.model_options.get("manual_offloading",
                                       True):  # Check key safely using patcher.model_options, default to True if not present
            log.info("Using manual_offloading (default). Moving transformer to main device.")
            transformer.to(device)
        # --- End Block Swap Init ---

        # --- FETA Setup ---
        if feta_args is not None and latent_video_length > 1:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"] # Assign here
            feta_end_percent = feta_args["end_percent"]   # Assign here
            num_frames_for_feta = context_frames if context_options is not None else latent_video_length
            set_num_frames(num_frames_for_feta)
            # enable_enhance() # Enable/disable is handled inside the loop
        else:
            # Ensure feta_args is None if not used, disable_enhance called once
            feta_args = None
            disable_enhance()
        # --- End FETA Setup ---

        # --- TeaCache Setup ---
        if teacache_args is not None:
            transformer.enable_teacache = True
            transformer.rel_l1_thresh = teacache_args["rel_l1_thresh"]
            transformer.teacache_start_step = teacache_args["start_step"]
            transformer.teacache_cache_device = teacache_args["cache_device"]
            # Calculate end step based on the *current* schedule length
            transformer.teacache_end_step = steps if teacache_args["end_step"] == -1 else teacache_args["end_step"]
            transformer.teacache_use_coefficients = teacache_args["use_coefficients"]
            transformer.teacache_mode = teacache_args["mode"]
            if hasattr(transformer, 'teacache_state') and hasattr(transformer.teacache_state, 'clear_all'):
                transformer.teacache_state.clear_all()
            else:
                log.warning("Transformer does not have teacache_state or clear_all method.")
        else:
            transformer.enable_teacache = False
        # --- End TeaCache Setup ---

        # --- SLG Setup ---
        if slg_args is not None:
            if batched_cfg:
                 log.warning("Batched CFG is not supported with SLG. Disabling SLG.")
                 slg_args = None # Disable SLG
                 transformer.slg_blocks = None
            else:
                transformer.slg_blocks = slg_args["blocks"]
                transformer.slg_start_percent = slg_args["start_percent"]
                transformer.slg_end_percent = slg_args["end_percent"]
        else:
            transformer.slg_blocks = None
        # --- End SLG Setup ---

        # Initialize TeaCache state tracking variables
        self.teacache_state = [None, None]
        self.teacache_state_source = [None, None]
        # self.teacache_states_context = [] # This wasn't used, can be removed

        # --- FlowEdit Setup ---
        if flowedit_args is not None:
            if samples is None:
                raise ValueError("FlowEdit requires initial 'samples' (latents) input.")
            source_embeds = flowedit_args["source_embeds"]
            source_image_embeds = flowedit_args.get("source_image_embeds", image_embeds) # Use target embeds as default source
            source_image_cond = source_image_embeds.get("image_embeds", None)
            source_clip_fea = source_image_embeds.get("clip_fea", clip_fea) # Use target clip_fea as default
            skip_steps = flowedit_args["skip_steps"]
            drift_steps = flowedit_args["drift_steps"]

            # Setup CFG schedules for source and drift phases
            source_cfg_value = flowedit_args["source_cfg"]
            drift_cfg_value = flowedit_args["drift_cfg"]
            source_cfg = [source_cfg_value] * (steps + 1) # Use actual steps count
            drift_cfg = [drift_cfg_value] * (steps + 1)   # Use actual steps count

            x_init = original_image.clone() # Use the initial latent provided
            x_tgt = original_image.clone()  # Target starts as the initial latent

            drift_flow_shift = flowedit_args.get("drift_flow_shift", 3.0) # Default if not provided
            # Commented out drift timestep recalculation - assuming main loop sigma handles it
        # --- End FlowEdit Setup ---

        # --- Experimental Args Setup ---
        use_cfg_zero_star = False
        use_zero_init = True # Default from original code
        zero_star_steps = 0
        if experimental_args is not None:
            video_attention_split_steps_str = experimental_args.get("video_attention_split_steps", "")
            if video_attention_split_steps_str:
                try:
                    transformer.video_attention_split_steps = [int(x.strip()) for x in video_attention_split_steps_str.split(",")]
                except ValueError:
                    log.warning(f"Invalid format for video_attention_split_steps: '{video_attention_split_steps_str}'. Ignoring.")
                    transformer.video_attention_split_steps = []
            else:
                transformer.video_attention_split_steps = []
            use_zero_init = experimental_args.get("use_zero_init", True)
            use_cfg_zero_star = experimental_args.get("cfg_zero_star", False)
            zero_star_steps = experimental_args.get("zero_star_steps", 0)
        # --- End Experimental Args Setup ---

        # region model pred
        # This function definition seems okay, relies on variables defined in the outer scope.
        def predict_with_cfg(z, cfg_scale, positive_embeds, negative_embeds, timestep, idx, image_cond=None,
                             clip_fea=None, control_latents=None, vace_data=None, teacache_state=None):
            # Use torch.inference_mode() for efficiency if gradients aren't needed
            with torch.inference_mode(mode=not transformer.training): # More robust than autocast alone
                with torch.autocast(device_type=mm.get_autocast_device(device), dtype=model["dtype"], enabled=True):

                    # Access latent_model_input from outer scope (ensure it's defined before loop)
                    if use_cfg_zero_star and (idx <= zero_star_steps) and use_zero_init:
                        # Need latent_model_input shape. Define it before the loop.
                        if 'latent_model_input' in locals() or 'latent_model_input' in globals():
                             return latent_model_input * 0, None
                        else:
                             # Fallback: create zeros based on input z shape if latent_model_input not ready
                             log.warning("latent_model_input not available for zero_star init, using z shape.")
                             return torch.zeros_like(z), None


                    nonlocal patcher # Ensure patcher from outer scope is used
                    # Use len(sigmas) instead of len(timesteps) if sigmas are primary
                    schedule_length = len(sigmas) if sigmas is not None else len(timesteps)
                    # Avoid division by zero if schedule_length is 0 (e.g., 1 sigma value)
                    current_step_percentage = idx / schedule_length if schedule_length > 0 else 0.0


                    control_lora_enabled = False
                    image_cond_input = image_cond # Start with base image_cond

                    # Control Latents / Control LoRA Logic
                    if control_latents is not None:
                        is_control_active = (control_start_percent <= current_step_percentage <= control_end_percent) or \
                                            (control_end_percent > 0 and idx == 0 and current_step_percentage >= control_start_percent)

                        if control_lora:
                            control_lora_enabled = is_control_active
                            if control_lora_enabled:
                                image_cond_input = control_latents.to(device) # LoRA uses control latents directly
                                # Ensure LoRA is patched if needed
                                if not patcher.model.is_patched:
                                    log.info("Loading Control-LoRA...")
                                    # Assuming apply_lora handles patching correctly
                                    patcher = apply_lora(patcher, device, device, low_mem_load=False) # Re-check apply_lora args
                                    patcher.model.is_patched = True
                            else:
                                # Unpatch LoRA if not active
                                if patcher.model.is_patched:
                                    log.info("Unloading Control-LoRA...")
                                    patcher.unpatch_model(device) # Assuming this method exists
                                    patcher.model.is_patched = False
                                # What should image_cond_input be if LoRA is inactive? Maybe None or original image_cond?
                                # Let's assume original image_cond if available, else None
                                image_cond_input = image_cond
                        else: # Non-LoRA control (e.g., Fun-Control)
                            if is_control_active:
                                # Concatenate control signal with base image condition
                                if image_cond is not None:
                                     image_cond_input = torch.cat([control_latents, image_cond], dim=0) # Check dim
                                else:
                                     # If no base image_cond, maybe just use control_latents? Depends on model.
                                     # Or maybe zeros + control? Let's assume zeros + control for Fun-Control like logic
                                     zeros_for_control = torch.zeros_like(control_latents)
                                     image_cond_input = torch.cat([control_latents, zeros_for_control], dim=0) # Check dim
                            else:
                                # Concatenate zeros with base image condition
                                if image_cond is not None:
                                     zeros_for_control = torch.zeros_like(control_latents)
                                     image_cond_input = torch.cat([zeros_for_control, image_cond], dim=0) # Check dim
                                else:
                                     # If no base image_cond and control inactive, maybe None?
                                     image_cond_input = None

                    # Base parameters for transformer call
                    base_params = {
                        'seq_len': seq_len,
                        'device': device,
                        'freqs': freqs, # From outer scope
                        't': timestep, # Current timestep
                        'current_step': idx,
                        'y': [image_cond_input] if image_cond_input is not None else None,
                        'control_lora_enabled': control_lora_enabled,
                        'vace_data': vace_data if vace_data is not None else None, # From outer scope
                    }

                    batch_size = z.shape[0] # Get batch size from input latent

                    # Handle multiple positive prompts for CFG
                    effective_negative_embeds = negative_embeds
                    if not math.isclose(cfg_scale, 1.0) and isinstance(positive_embeds, list) and len(positive_embeds) > 1:
                        # Repeat negative embeds to match the number of positive prompts if doing multi-prompt CFG
                        effective_negative_embeds = negative_embeds.repeat(len(positive_embeds), 1, 1) # Adjust repeat dims if needed

                    # Determine intermediate device (usually same as main device unless offloading)
                    intermediate_device = device # Or offload_device if doing specific offloading here

                    # --- CFG Prediction Logic ---
                    noise_pred = None
                    final_teacache_state = [None, None] # Default

                    if math.isclose(cfg_scale, 1.0):
                        # No CFG, just conditional prediction
                        noise_pred_cond, teacache_state_cond = transformer(
                            [z], context=positive_embeds, clip_fea=clip_fea, is_uncond=False,
                            current_step_percentage=current_step_percentage,
                            pred_id=teacache_state[0] if teacache_state else None,
                            **base_params
                        )
                        noise_pred = noise_pred_cond[0].to(intermediate_device)
                        final_teacache_state = [teacache_state_cond, None]
                    elif not batched_cfg:
                        # --- Separate Cond/Uncond ---
                        # Conditional Prediction
                        noise_pred_cond, teacache_state_cond = transformer(
                            [z], context=positive_embeds, clip_fea=clip_fea, is_uncond=False,
                            current_step_percentage=current_step_percentage,
                            pred_id=teacache_state[0] if teacache_state else None,
                            **base_params
                        )
                        noise_pred_cond = noise_pred_cond[0].to(intermediate_device)

                        # Unconditional Prediction
                        # Use effective_negative_embeds which might be repeated
                        noise_pred_uncond, teacache_state_uncond = transformer(
                            [z], context=effective_negative_embeds, clip_fea=clip_fea_neg if clip_fea_neg is not None else clip_fea,
                            is_uncond=True, current_step_percentage=current_step_percentage,
                            pred_id=teacache_state[1] if teacache_state else None,
                            **base_params
                        )
                        noise_pred_uncond = noise_pred_uncond[0].to(intermediate_device)

                        # Combine using CFG formula
                        if use_cfg_zero_star:
                            alpha = optimized_scale(
                                noise_pred_cond.view(batch_size, -1),
                                noise_pred_uncond.view(batch_size, -1)
                            ).view(batch_size, 1, 1, 1) # Reshape alpha
                            noise_pred = noise_pred_uncond * alpha + cfg_scale * (noise_pred_cond - noise_pred_uncond * alpha)
                        else:
                            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

                        final_teacache_state = [teacache_state_cond, teacache_state_uncond]
                        # --- End Separate Cond/Uncond ---
                    else: # batched_cfg == True
                        # --- Batched Cond/Uncond ---
                        # Prepare batched input
                        batch_z = torch.cat([z] * 2, dim=0) # Repeat latent
                        batch_context = list(positive_embeds) + list(effective_negative_embeds) # Combine contexts
                        # Clip features might need similar batching if they differ per prompt
                        batch_clip_fea = clip_fea # Assuming same clip_fea for cond/uncond for now

                        # Call transformer once with batched input
                        # Note: Transformer needs to support this batching internally
                        [noise_pred_cond_batch, noise_pred_uncond_batch], teacache_state_batch = transformer(
                            [batch_z], context=batch_context, clip_fea=batch_clip_fea, is_uncond=False, # is_uncond might need adjustment in transformer
                            current_step_percentage=current_step_percentage,
                            pred_id=teacache_state[0] if teacache_state else None, # How does TeaCache handle batching? Assuming state[0] is okay.
                            **base_params
                        )
                        # Separate the results
                        noise_pred_cond = noise_pred_cond_batch[0:batch_size].to(intermediate_device)
                        noise_pred_uncond = noise_pred_uncond_batch[batch_size:].to(intermediate_device)

                        # Combine using CFG formula
                        if use_cfg_zero_star:
                             alpha = optimized_scale(
                                noise_pred_cond.view(batch_size, -1),
                                noise_pred_uncond.view(batch_size, -1)
                            ).view(batch_size, 1, 1, 1)
                             noise_pred = noise_pred_uncond * alpha + cfg_scale * (noise_pred_cond - noise_pred_uncond * alpha)
                        else:
                            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

                        # How to handle teacache state in batch mode? Assuming single state returned
                        final_teacache_state = [teacache_state_batch, None] # Or adjust based on transformer output
                        # --- End Batched Cond/Uncond ---

                    return noise_pred, final_teacache_state
        # endregion model pred

        log.info(f"Sampling {(latent_video_length - 1) * 4 + 1} frames at {latent.shape[3] * 8}x{latent.shape[2] * 8} with {steps} steps")

        intermediate_device = device # Default intermediate device

        # --- Diff-Diff Mask Setup ---
        diff_diff_masks = None # Renamed from 'masks' to avoid conflict
        if samples is not None and mask is not None: # Check original mask from samples input
            log.info("Preparing masks for Diff-Diff.")
            diff_diff_mask_base = 1.0 - mask # Invert mask
            # Ensure thresholds match the current schedule length
            thresholds = torch.arange(len(timesteps), dtype=original_image.dtype, device=device) / len(timesteps)
            thresholds = thresholds.view(-1, 1, 1, 1, 1) # Reshape for broadcasting
            # Repeat base mask and compare with thresholds
            diff_diff_masks = diff_diff_mask_base.unsqueeze(0).repeat(len(timesteps), 1, 1, 1, 1).to(device)
            diff_diff_masks = diff_diff_masks > thresholds
        # --- End Diff-Diff Mask Setup ---


        latent_shift_loop = False
        if loop_args is not None:
            latent_shift_loop = True
            is_looped = True # Set is_looped flag
            latent_skip = loop_args["shift_skip"] # Assign here
            latent_shift_start_percent = loop_args["start_percent"] # Assign here
            latent_shift_end_percent = loop_args["end_percent"]   # Assign here
            shift_idx = 0 # Initialize shift index

        # --- Memory Cleanup ---
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
        except Exception as e:
            log.warning(f"Could not reset CUDA stats: {e}")
        # --- End Memory Cleanup ---

        # Define latent_model_input before the loop for zero_star init check
        latent_model_input = latent.to(device)

        # region main loop start
        # Iterate using the final `sigmas` and `timesteps`
        for idx, t in enumerate(tqdm(timesteps[:-1])): # Loop up to the second-to-last timestep/sigma
            current_sigma = sigmas[idx]
            next_sigma = sigmas[idx + 1]
            timestep = torch.tensor([t], device=device) # Use current timestep t

            if flowedit_args is not None:
                if idx < skip_steps:
                    continue

            # --- Diff-Diff Application ---
            if diff_diff_masks is not None and original_image is not None:
                if idx < len(diff_diff_masks): # Ensure index is valid
                    # Use the *next* timestep for scaling the original image
                    noise_timestep_for_dd = timesteps[idx + 1]
                    # Use scheduler's method if available, otherwise manual scaling
                    if hasattr(sample_scheduler, 'scale_noise'):
                         image_latent_dd = sample_scheduler.scale_noise(
                            original_image, torch.tensor([noise_timestep_for_dd], device=device), noise.to(device) # Pass noise tensor
                         )
                    elif hasattr(sample_scheduler, 'add_noise'):
                         # Alternative: add noise corresponding to the *next* step's sigma
                         noise_for_dd = torch.randn_like(original_image)
                         image_latent_dd = sample_scheduler.add_noise(original_image, noise_for_dd, timesteps[idx+1:idx+2])
                    else:
                         # Manual scaling (less reliable)
                         alpha_next = 1.0 - sigmas[idx+1] # Example alpha calculation
                         image_latent_dd = alpha_next * original_image + sigmas[idx+1] * torch.randn_like(original_image)

                    current_mask_dd = diff_diff_masks[idx].to(latent.device, latent.dtype)
                    latent = image_latent_dd * current_mask_dd + latent * (1.0 - current_mask_dd)
                else:
                    log.warning(f"Diff-Diff mask index {idx} out of bounds.")
            # --- End Diff-Diff Application ---

            latent_model_input = latent.to(device) # Ensure latent is on device for this step

            # Use len(sigmas) - 1 for percentage calculation
            schedule_length_for_percent = len(sigmas) -1 if sigmas is not None and len(sigmas)>1 else steps
            # Avoid division by zero if schedule has only 1 step (0 length)
            current_step_percentage = idx / schedule_length_for_percent if schedule_length_for_percent > 0 else 0.0


            # --- Latent Shift ---
            if latent_shift_loop:
                if latent_shift_start_percent <= current_step_percentage <= latent_shift_end_percent:
                    latent_model_input = torch.cat(
                        [latent_model_input[:, shift_idx:]] + [latent_model_input[:, :shift_idx]], dim=1)
            # --- End Latent Shift ---

            # --- FETA Enable/Disable ---
            if feta_args is not None:
                if feta_start_percent <= current_step_percentage <= feta_end_percent:
                    enable_enhance()
                else:
                    disable_enhance()
            # --- End FETA ---

            # --- Main Prediction and Step ---
            noise_pred = None
            x0 = None # Initialize x0 for the step

            # --- FlowEdit Path ---
            if flowedit_args is not None:
                # Use sigmas directly
                sigma_t = current_sigma
                sigma_prev = next_sigma

                # Generate noise for source interpolation
                noise_fe = torch.randn(x_init.shape, generator=seed_g, device=device, dtype=x_init.dtype)

                # Determine current CFG scale based on phase (drift or source)
                current_cfg_list = drift_cfg if idx < steps - drift_steps else source_cfg
                current_cfg_scale = current_cfg_list[idx] if idx < len(current_cfg_list) else base_cfg_value # Fallback

                # Interpolate source latent
                zt_src = (1.0 - sigma_t) * x_init + sigma_t * noise_fe # Check formula if needed

                # Calculate target latent based on source and current target estimate
                # Ensure x_tgt is initialized correctly before the loop if flowedit_args is not None
                if x_tgt is None: x_tgt = x_init.clone() # Initialize x_tgt if first step
                zt_tgt = x_tgt + zt_src - x_init # FlowEdit formula

                vt_src = torch.zeros_like(zt_src) # Initialize source velocity prediction
                # Source Prediction Phase
                if idx < steps - drift_steps:
                    if context_options is not None:
                        # --- Context Windowing for Source ---
                        counter_src = torch.zeros_like(zt_src, device=intermediate_device)
                        vt_src.fill_(0.0) # Ensure vt_src is zeroed
                        context_queue_src = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                        for c in context_queue_src:
                            window_id = self.window_tracker.get_window_id(c)
                            current_teacache = self.window_tracker.get_teacache(window_id, self.teacache_state_source) if teacache_args is not None else None
                            prompt_index = min(int(max(c) / section_size), num_prompts - 1) if section_size > 0 else 0

                            # Use source prompts
                            positive_src = source_embeds["prompt_embeds"][prompt_index] if len(source_embeds["prompt_embeds"]) > 1 else source_embeds["prompt_embeds"]
                            negative_src = source_embeds["negative_prompt_embeds"]

                            partial_img_emb_src = None
                            if source_image_cond is not None:
                                # Slice source image condition
                                partial_img_emb_src = source_image_cond[:, c, :, :]
                                # Handle reference frame if needed (assuming first frame is ref)
                                if c[0] == 0 and source_image_cond.shape[1] > 0:
                                     partial_img_emb_src[:, 0, :, :] = source_image_cond[:, 0, :, :].to(intermediate_device)

                            partial_zt_src = zt_src[:, c, :, :]
                            # Use source CFG scale
                            vt_src_context, new_teacache = predict_with_cfg(
                                partial_zt_src, current_cfg_scale,
                                positive_src, negative_src,
                                timestep, idx, partial_img_emb_src, None, # Control latents for source? Assume None
                                source_clip_fea, current_teacache)

                            if teacache_args is not None and new_teacache is not None:
                                self.window_tracker.teacache_states[window_id] = new_teacache # Store source teacache state?

                            window_mask = create_window_mask(vt_src_context, c, latent_video_length, context_overlap)
                            vt_src[:, c, :, :] += vt_src_context * window_mask
                            counter_src[:, c, :, :] += window_mask
                        # Avoid division by zero
                        vt_src = torch.where(counter_src > 0, vt_src / counter_src, vt_src)
                        # --- End Context Windowing for Source ---
                    else: # No context options for source
                        vt_src, self.teacache_state_source = predict_with_cfg(
                            zt_src, current_cfg_scale,
                            source_embeds["prompt_embeds"], source_embeds["negative_prompt_embeds"],
                            timestep, idx, source_image_cond, source_clip_fea, None, # Control latents for source? Assume None
                            teacache_state=self.teacache_state_source)
                # else: vt_src remains zeros (drift phase)

                # Target Prediction Phase (always happens)
                vt_tgt = torch.zeros_like(zt_tgt) # Initialize target velocity prediction
                if context_options is not None:
                    # --- Context Windowing for Target ---
                    counter_tgt = torch.zeros_like(zt_tgt, device=intermediate_device)
                    vt_tgt.fill_(0.0)
                    context_queue_tgt = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                    for c in context_queue_tgt:
                        window_id = self.window_tracker.get_window_id(c)
                        current_teacache = self.window_tracker.get_teacache(window_id, self.teacache_state) if teacache_args is not None else None
                        prompt_index = min(int(max(c) / section_size), num_prompts - 1) if section_size > 0 else 0

                        # Use target prompts
                        positive_tgt = text_embeds["prompt_embeds"][prompt_index] if len(text_embeds["prompt_embeds"]) > 1 else text_embeds["prompt_embeds"]
                        negative_tgt = text_embeds["negative_prompt_embeds"]

                        # Slice target image condition and control latents
                        partial_img_emb_tgt = None
                        partial_control_latents_tgt = None
                        if image_cond is not None:
                            partial_img_emb_tgt = image_cond[:, c, :, :]
                            if c[0] == 0 and image_cond.shape[1] > 0:
                                partial_img_emb_tgt[:, 0, :, :] = image_cond[:, 0, :, :].to(intermediate_device)
                        if control_latents is not None:
                             partial_control_latents_tgt = control_latents[:, c, :, :] # Slice control latents

                        partial_zt_tgt = zt_tgt[:, c, :, :]
                        # Use target CFG scale (base_cfg_value or from schedule)
                        target_cfg_scale = cfg_schedule[idx] if idx < len(cfg_schedule) else base_cfg_value
                        vt_tgt_context, new_teacache = predict_with_cfg(
                            partial_zt_tgt, target_cfg_scale,
                            positive_tgt, negative_tgt,
                            timestep, idx, partial_img_emb_tgt, clip_fea, partial_control_latents_tgt, # Pass sliced control
                            current_teacache)

                        if teacache_args is not None and new_teacache is not None:
                            self.window_tracker.teacache_states[window_id] = new_teacache # Store target teacache state

                        window_mask = create_window_mask(vt_tgt_context, c, latent_video_length, context_overlap)
                        vt_tgt[:, c, :, :] += vt_tgt_context * window_mask
                        counter_tgt[:, c, :, :] += window_mask
                    # Avoid division by zero
                    vt_tgt = torch.where(counter_tgt > 0, vt_tgt / counter_tgt, vt_tgt)
                    # --- End Context Windowing for Target ---
                else: # No context options for target
                    target_cfg_scale = cfg_schedule[idx] if idx < len(cfg_schedule) else base_cfg_value
                    vt_tgt, self.teacache_state = predict_with_cfg(
                        zt_tgt, target_cfg_scale,
                        text_embeds["prompt_embeds"], text_embeds["negative_prompt_embeds"],
                        timestep, idx, image_cond, clip_fea, control_latents, # Pass full control latents
                        teacache_state=self.teacache_state)

                # Calculate velocity difference and update target latent
                v_delta = vt_tgt - vt_src
                # Ensure types match for the update step
                x_tgt = x_tgt.to(dtype=torch.float32)
                v_delta = v_delta.to(dtype=torch.float32)
                # FIX: Cast sigmas to float for subtraction
                sigma_diff = float(sigma_prev) - float(sigma_t) # Line ~703
                x_tgt = x_tgt + sigma_diff * v_delta

                # Update latent for the *next* iteration
                latent = x_tgt.to(latent.dtype) # Convert back to original latent dtype
                x0 = x_tgt # Store the estimated x0 for potential callback

            # --- Context Windowing Path (No FlowEdit) ---
            elif context_options is not None:
                counter = torch.zeros_like(latent_model_input, device=intermediate_device)
                noise_pred = torch.zeros_like(latent_model_input, device=intermediate_device)
                context_queue = list(context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))

                for c in context_queue:
                    window_id = self.window_tracker.get_window_id(c)
                    current_teacache = self.window_tracker.get_teacache(window_id, self.teacache_state) if teacache_args is not None else None
                    prompt_index = min(int(max(c) / section_size), num_prompts - 1) if section_size > 0 else 0

                    if context_options["verbose"]: log.info(f"Context Window: {c}, Prompt index: {prompt_index}")

                    positive = text_embeds["prompt_embeds"][prompt_index] if len(text_embeds["prompt_embeds"]) > 1 else text_embeds["prompt_embeds"]
                    negative = text_embeds["negative_prompt_embeds"]

                    partial_img_emb = None
                    partial_control_latents = None
                    image_index = 0 # Reset image_index for each window

                    if image_cond is not None:
                        # --- Image Condition Handling within Context ---
                        num_windows_img = context_options.get("image_cond_window_count", 1) # Default to 1 window
                        section_size_img = latent_video_length / num_windows_img if num_windows_img > 0 else latent_video_length
                        image_index = min(int(max(c) / section_size_img), num_windows_img - 1) if section_size_img > 0 else 0

                        partial_img_emb = image_cond[:, c, :, :] # Slice image condition
                        if control_latents is not None:
                            partial_control_latents = control_latents[:, c, :, :] # Slice control latents

                        # Handle reference frame (assuming first frame is ref)
                        if c[0] == 0 and image_cond.shape[1] > 0:
                             partial_image_cond_ref = image_cond[:, 0:1, :, :].to(intermediate_device) # Get ref frame
                             partial_img_emb[:, 0:1, :, :] = partial_image_cond_ref # Replace first frame in slice

                             # --- Experimental VAE Feedback ---
                             # Check if previous prediction exists and conditions met
                             if hasattr(self, "previous_noise_pred_context") and image_index > 0 and \
                                idx >= context_options.get("image_cond_start_step", steps + 1) and \
                                context_vae is not None:
                                try:
                                    log.info(f"Applying VAE feedback for image_index {image_index}")
                                    # Decode last frame of previous prediction
                                    # Ensure previous_noise_pred_context has the right shape/device
                                    to_decode = self.previous_noise_pred_context[:, -1:, :, :].to(context_vae.dtype) # Keep time dim

                                    if isinstance(context_vae, TAEHV):
                                        # TAEHV expects B, T, C, H, W -> permute latent C,T,H,W -> B,C,T,H,W
                                        # Decode expects B, C, T, H, W
                                        image = context_vae.decode_video(to_decode.unsqueeze(0).permute(0, 2, 1, 3, 4))[0] # Add batch, permute C,T
                                        # Encode expects B, T, C, H, W -> permute back
                                        # Repeat the single decoded frame if needed for encoding context
                                        encoded_image = context_vae.encode_video(image.repeat(1, 1, 1, 1, 1)).permute(0, 2, 1, 3, 4).squeeze(0) # Remove batch
                                    else: # Standard VAE
                                        # Decode expects B, C, H, W -> use last frame T=1
                                        image = context_vae.decode(to_decode.squeeze(1), device=device, tiled=False)[0] # Remove T dim
                                        # Encode expects B, C, H, W
                                        encoded_image = context_vae.encode(image.unsqueeze(0).to(context_vae.dtype), device=device, tiled=False)

                                    # Blend or replace the reference frame in partial_img_emb
                                    # Simple replacement for now:
                                    if encoded_image.shape == partial_img_emb[:, 0:1, :, :].shape:
                                         partial_img_emb[:, 0:1, :, :] = encoded_image.to(partial_img_emb.dtype)
                                    else:
                                         log.warning(f"VAE feedback shape mismatch: {encoded_image.shape} vs {partial_img_emb[:, 0:1, :, :].shape}")

                                except Exception as e_vae:
                                    log.error(f"Error during VAE feedback: {e_vae}")
                                # --- End Experimental VAE Feedback ---
                        # --- End Image Condition Handling ---

                    partial_vace_context = None
                    if vace_data is not None:
                        # Assuming vace_data structure and slicing logic is correct
                        # This might need adjustment based on how vace_data relates to context windows
                        # Simple approach: use the first vace_data entry for all windows?
                        current_vace_entry = vace_data[0]
                        if current_vace_entry and "context" in current_vace_entry:
                             vace_ctx_tensor = current_vace_entry["context"]
                             if isinstance(vace_ctx_tensor, list): vace_ctx_tensor = vace_ctx_tensor[0] # Handle potential list
                             # Slice VACE context - needs careful thought on how VACE context maps to windows
                             # Assuming simple slicing for now:
                             if vace_ctx_tensor.shape[1] >= max(c) + 1: # Check time dimension
                                 partial_vace_context = [vace_ctx_tensor[:, c, :, :]] # Slice and wrap in list
                                 if has_ref and c[0] == 0 and vace_ctx_tensor.shape[1] > 0: # Handle VACE ref frame
                                     partial_vace_context[0][:, 0, :, :] = vace_ctx_tensor[:, 0, :, :]
                             else:
                                 log.warning("VACE context tensor too short for window slicing.")


                    partial_latent_model_input = latent_model_input[:, c, :, :]
                    current_cfg_scale = cfg_schedule[idx] if idx < len(cfg_schedule) else base_cfg_value

                    noise_pred_context, new_teacache = predict_with_cfg(
                        partial_latent_model_input,
                        current_cfg_scale, positive, negative,
                        timestep, idx, partial_img_emb, clip_fea, partial_control_latents, partial_vace_context,
                        current_teacache)

                    if teacache_args is not None and new_teacache is not None:
                        self.window_tracker.teacache_states[window_id] = new_teacache

                    # Store prediction for potential VAE feedback *only if VAE feedback is enabled*
                    if image_cond is not None and image_index > 0 and \
                       idx >= context_options.get("image_cond_start_step", steps + 1) and \
                       context_vae is not None:
                        # Detach and clone to prevent graph issues and modification
                        self.previous_noise_pred_context = noise_pred_context.detach().clone()

                    window_mask = create_window_mask(noise_pred_context, c, latent_video_length, context_overlap, looped=is_looped)
                    noise_pred[:, c, :, :] += noise_pred_context * window_mask
                    counter[:, c, :, :] += window_mask

                # Avoid division by zero
                # FIX: Apply where condition correctly
                noise_pred = torch.where(counter > 0, noise_pred / counter, noise_pred) # Line ~914 - Looks correct, checker likely wrong
                # --- End Context Windowing Path ---

            # --- Normal Inference Path (No FlowEdit, No Context) ---
            else:
                current_cfg_scale = cfg_schedule[idx] if idx < len(cfg_schedule) else base_cfg_value
                noise_pred, self.teacache_state = predict_with_cfg(
                    latent_model_input,
                    current_cfg_scale,
                    text_embeds["prompt_embeds"],
                    text_embeds["negative_prompt_embeds"],
                    timestep, idx, image_cond, clip_fea, control_latents, vace_data,
                    teacache_state=self.teacache_state)
            # --- End Normal Inference Path ---


            # --- Reverse Latent Shift ---
            if latent_shift_loop:
                if latent_shift_start_percent <= current_step_percentage <= latent_shift_end_percent:
                    # Apply reverse shift to noise_pred *before* the sampler step
                    if noise_pred is not None:
                         noise_pred = torch.cat([noise_pred[:, latent_video_length - shift_idx:]] + [
                            noise_pred[:, :latent_video_length - shift_idx]], dim=1)
                    # Increment shift_idx for the *next* iteration's forward shift
                    # FIX: Use latent_skip variable initialized earlier
                    shift_idx = (shift_idx + latent_skip) % latent_video_length # Line ~1238
            # --- End Reverse Latent Shift ---


            # --- Sampler Step ---
            if noise_pred is not None and flowedit_args is None: # Only step if not using FlowEdit's update
                latent = latent.to(intermediate_device) # Ensure latent is on correct device for step

                # Use the scheduler's step function
                # Ensure noise_pred has batch dim
                noise_pred_batch = noise_pred.unsqueeze(0) if noise_pred.dim() == 4 else noise_pred
                latent_batch = latent.unsqueeze(0) if latent.dim() == 4 else latent

                # The step function needs the *current* timestep `t`
                step_output = sample_scheduler.step(
                    noise_pred_batch,
                    t, # Current timestep t
                    latent_batch,
                    return_dict=False,
                    generator=seed_g) # Pass generator if needed by scheduler

                # Get the previous sample (output of the step)
                prev_latent = step_output[0] if isinstance(step_output, tuple) else step_output.prev_sample
                latent = prev_latent.squeeze(0) # Remove batch dim

                # Estimate x0 for callback (often needed)
                # Formula depends on scheduler, e.g., x0 = latent - sigma * noise_pred
                # Use scheduler's predict_original_sample if available
                if hasattr(sample_scheduler, 'predict_original_sample'):
                     # Ensure t is the correct shape/type for predict_original_sample
                     t_for_pred = t.to(latent_batch.device, dtype=latent_batch.dtype)
                     x0_pred = sample_scheduler.predict_original_sample(latent_batch, noise_pred_batch, t_for_pred)
                     x0 = x0_pred.squeeze(0).to(device) # Remove batch dim, move to main device
                else:
                    # Fallback approximation
                    # FIX: Cast current_sigma to float
                    x0 = (latent_model_input - noise_pred.to(latent_model_input.device) * float(current_sigma)).to(device) # Line ~940

            elif flowedit_args is not None:
                 # x0 was already estimated as x_tgt during the FlowEdit update
                 x0 = x_tgt.to(device) # Ensure x0 is on the main device
            else:
                 # Handle case where noise_pred might be None (e.g., zero_star init)
                 x0 = latent.to(device) # Use current latent as x0 estimate

            # --- Callback ---
            if callback is not None and x0 is not None: # Ensure x0 exists
                try:
                    # Permute x0 to T, C, H, W for callback if needed
                    # Check x0 shape before permuting
                    if x0.dim() == 4: # Expecting C, T, H, W or similar
                        callback_latent = x0.permute(1, 0, 2, 3) # Example: C,T,H,W -> T,C,H,W
                    else:
                        log.warning(f"Unexpected x0 shape for callback: {x0.shape}")
                        callback_latent = x0 # Pass as is if shape is wrong

                    callback(idx, callback_latent, None, steps) # Call the callback
                except Exception as e_cb:
                    log.warning(f"Callback failed at step {idx}: {e_cb}")
            elif idx < steps: # Only update pbar if not calling callback
                pbar.update(1)
            # --- End Callback ---

            # Cleanup per-step tensors (optional, helps if memory is tight)
            del latent_model_input, timestep, noise_pred
            if 'noise_pred_batch' in locals(): del noise_pred_batch
            if 'latent_batch' in locals(): del latent_batch
            if 'prev_latent' in locals(): del prev_latent
            if 'x0_pred' in locals(): del x0_pred
            if 'noise_pred_context' in locals(): del noise_pred_context
            if 'vt_src' in locals(): del vt_src
            if 'vt_tgt' in locals(): del vt_tgt
            if 'v_delta' in locals(): del v_delta
            if torch.cuda.is_available(): torch.cuda.empty_cache() # Aggressive cleanup

        # --- End Main Loop ---

        # Final latent is the result of the last step
        final_latent = latent.to(device) # Ensure final latent is on main device

        # --- TeaCache Logging ---
        if teacache_args is not None and hasattr(transformer, 'teacache_state'):
            try: # Add try-except for safety
                states = transformer.teacache_state.states
                state_names = {0: "conditional", 1: "unconditional"}
                for pred_id, state in states.items():
                    name = state_names.get(pred_id, f"prediction_{pred_id}")
                    if 'skipped_steps' in state and state['skipped_steps']:
                        log.info(f"TeaCache skipped: {len(state['skipped_steps'])} {name} steps.") #{state['skipped_steps']}") # Keep log concise
                transformer.teacache_state.clear_all()
            except Exception as e_tea:
                log.warning(f"Error during TeaCache logging/cleanup: {e_tea}")
        # --- End TeaCache Logging ---

        # --- Final Offloading ---
        if force_offload:
            # Access manual_offloading option from the original model's options via patcher
            if patcher.model_options.get("manual_offloading", True):  # Check key safely using patcher.model_options
                log.info(f"Force offload: Moving transformer to {offload_device}")
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()
        # --- End Final Offloading ---

        # --- Final Memory Stats ---
        try:
            print_memory(device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
        except Exception as e:
            log.warning(f"Could not print/reset memory stats: {e}")
        # --- End Final Memory Stats ---

        # Return final latent, adding batch dimension and moving to CPU
        return ({
                    "samples": final_latent.unsqueeze(0).cpu(),
                    "looped": is_looped,
                    "end_image": end_image if not fun_model else None, # Pass end_image if relevant
                    "has_ref": has_ref,
                    "drop_last": drop_last,
                },)

# Note: WindowTracker class definition should be here or imported correctly.
# Assuming it's defined as provided previously.


# --- WindowTracker Class (ensure it's defined or imported) ---
class WindowTracker:
    def __init__(self, verbose=False):
        self.window_map = {}  # Maps frame sequence tuple to persistent ID
        self.next_id = 0
        # Stores teacache state ([cond_state, uncond_state]) per window ID
        self.teacache_states = {}
        self.verbose = verbose

    def get_window_id(self, frames):
        # Use tuple of frames directly as key assumes order matters for context
        key = tuple(frames)
        if key not in self.window_map:
            self.window_map[key] = self.next_id
            if self.verbose:
                log.info(f"New window pattern {key} -> ID {self.next_id}")
            self.next_id += 1
        return self.window_map[key]

    def get_teacache(self, window_id, base_state):
        if window_id not in self.teacache_states:
            if self.verbose:
                log.info(f"Initializing persistent teacache for window {window_id}")
            # Important: Create a *copy* of the base state, not a reference
            self.teacache_states[window_id] = list(base_state) if isinstance(base_state, list) else base_state # Shallow copy for list
        return self.teacache_states[window_id]

# --- End WindowTracker ---
