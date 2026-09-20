import argparse
import inspect
import math
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from peft import PeftModel


CONDITION_IMAGE_SIZE = 384
VAE_IMAGE_SIZE = 1024
USE_ONLY_TEXT_HIDDEN_STATES = True


DEFAULT_PROMPT = (
    "Task: You are a top 3D UV artist. From the input portrait, write a concise positive prompt that guides a "
    "diffusion model to generate a high-precision facial UV texture map. Avoid hair/eyeball description.\n\n"
    "Describe in order:\n"
    "1) Pose & expression: head yaw/pitch/roll angles; FACS AUs and intensity.\n"
    "2) Global geometry: brow ridge, cheekbones, nasal bridge, jawline sharpness; symmetry cues (left/right "
    "differences if any).\n"
    "3) Skin/feature details (relative levels/angles only, no absolute sizes):\n"
    "   - Forehead: height (low/med/high), tilt, smooth vs wrinkled.\n"
    "   - Eyes region (no eyeballs): shape, inner/outer canthus angles (°), eyelid crease type/direction, "
    "under-eye bag depth.\n"
    "   - Nose: root height, bridge straightness, tip shape, nostril wing width/shape, philtrum angle.\n"
    "   - Cheeks: zygomatic expansion, malar volume, smile-line depth.\n"
    "   - Lips/mouth corners: cupid’s bow prominence, philtrum clarity, mouth-corner tilt (°), teeth visibility.\n"
    "   - Jaw/chin: mandibular angle (°), chin projection, mental crease depth, submental fullness.\n"
    "   - Ears: helix/antihelix form, earlobe type, concha depth, transition to skull.\n"
    "   - Neck: laryngeal prominence, anterior neck folds count (few/med/many).\n"
    "4) Lighting (brief, geometry-supporting only): key light azimuth/elevation, intensity (low/med/high), "
    "tone (cool/neutral/warm), fill/rim presence.\n"
    "5) Material cues (geometry-supporting only): wrinkle/spot/scar influence (none/light/med/heavy), fine-line "
    "orientation.\n\n"
    "Output format:\n"
    "Please generate a UV unwrapped texture map of the person's head based on the following feature description, "
    "where each pixel corresponds to a position on the head surface in 3D space.\n"
    "<one-line text assembled from the above features, geometry-focused, no camera/lens/style terms>"
)


NEGATIVE_PROMPT = (
    "Task: Write a concise prompt that describes skin defects, mouth-corner shadow artifacts, "
    "and excessive lip highlights from the image. Avoid hair/eyeball constraints.\n\n"
    "Focus on: freckles, dark spots, acne, blemishes, hyperpigmentation, "
    "deep nasolabial folds, mouth-corner shadows, dark perioral creases, "
    "harsh perioral contrast, uneven lighting around lips, "
    "specular highlights on lips, glossy lips, overexposed lip highlights, bright lip reflections, "
    "asymmetry, distorted face, bad proportions.\n\n"
    "Output format:\n"
    "<one-line constraint text>"
)


def load_pipeline(
    model_path: str,
    adapter_path: Optional[str],
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> QwenImageEditPlusPipeline:
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_path)
    if adapter_path:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, adapter_path)
    pipe.transformer.to(dtype)
    pipe.text_encoder.to(dtype)
    pipe.vae.to(torch.float32)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    return pipe


def _calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__} does not support custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)
    if sigmas is not None:
        accepts_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_sigmas:
            raise ValueError(f"The current scheduler class {scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)

    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


def print_info(name: str, tensor: torch.Tensor) -> None:
    print(f"{name} shape: {tensor.shape}; min: {torch.min(tensor)}; max: {torch.max(tensor)}")


def _get_qwen_prompt_embeds(
    pipeline: QwenImageEditPlusPipeline,
    prompt: Union[str, List[str]],
    image: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    images = image if isinstance(image, list) else ([image] if image is not None else None)

    img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
    base_img_prompt = "".join(img_prompt_template.format(i + 1) for i in range(len(images))) if images else ""

    template = pipeline.prompt_template_encode
    drop_idx = pipeline.prompt_template_encode_start_idx
    text = [template.format(base_img_prompt + entry) for entry in prompt]

    device = device or pipeline._execution_device
    dtype = dtype or pipeline.text_encoder.dtype

    model_inputs = pipeline.processor(
        text=text,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    outputs = pipeline.text_encoder(
        **model_inputs,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1]
    attn_mask = model_inputs.attention_mask.bool()
    if getattr(pipeline, "use_only_text_hidden_states", USE_ONLY_TEXT_HIDDEN_STATES):
        image_token_id = getattr(pipeline.text_encoder.config, "image_token_id", None)
        if image_token_id is not None:
            attn_mask = (model_inputs.input_ids != image_token_id) & attn_mask

    split_hidden_states = pipeline._extract_masked_hidden(hidden_states, attn_mask)
    split_hidden_states = [entry[drop_idx:] for entry in split_hidden_states]
    attn_mask_list = [
        torch.ones(entry.size(0), dtype=model_inputs.attention_mask.dtype, device=entry.device)
        for entry in split_hidden_states
    ]
    max_seq_len = max(entry.size(0) for entry in split_hidden_states)
    prompt_embeds = torch.stack(
        [torch.cat([entry, entry.new_zeros(max_seq_len - entry.size(0), entry.size(1))]) for entry in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([entry, entry.new_zeros(max_seq_len - entry.size(0))]) for entry in attn_mask_list]
    )

    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    encoder_attention_mask = encoder_attention_mask.to(device)
    return prompt_embeds, encoder_attention_mask


def encode_prompt(
    pipeline: QwenImageEditPlusPipeline,
    prompt: Union[str, List[str]],
    image: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    max_sequence_length: int = 1024,
):
    del max_sequence_length
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embeds, prompt_embeds_mask = _get_qwen_prompt_embeds(
            pipeline,
            prompt,
            image,
            device,
        )

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
    return prompt_embeds, prompt_embeds_mask


def run_edit_plus(
    pipeline: QwenImageEditPlusPipeline,
    inputs: Dict[str, Any],
):
    image = inputs["image"]
    prompt = inputs["prompt"]
    negative_prompt = inputs.get("negative_prompt")
    images_in = image if isinstance(image, list) else [image]

    with torch.inference_mode():
        device = pipeline._execution_device
        vae_scale_factor = pipeline.vae_scale_factor

        image_size = images_in[-1].size
        calc_w, calc_h = _calculate_dimensions(VAE_IMAGE_SIZE * VAE_IMAGE_SIZE, image_size[0] / image_size[1])
        multiple_of = vae_scale_factor * 2
        width = (calc_w // multiple_of) * multiple_of
        height = (calc_h // multiple_of) * multiple_of

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        has_neg = negative_prompt is not None
        true_cfg_scale = float(inputs.get("true_cfg_scale", 1.0))
        do_true_cfg = (true_cfg_scale > 1.0) and has_neg

        condition_images = []
        neg_condition_images = []
        vae_images = []
        neg_vae_images = []
        vae_image_sizes = []
        for current_image in images_in:
            width_i, height_i = current_image.size
            ratio = float(width_i) / float(height_i)
            cond_w, cond_h = _calculate_dimensions(CONDITION_IMAGE_SIZE * CONDITION_IMAGE_SIZE, ratio)
            vae_w, vae_h = _calculate_dimensions(VAE_IMAGE_SIZE * VAE_IMAGE_SIZE, ratio)
            image_tensor = torch.from_numpy(np.array(current_image)).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0) * 2.0 - 1.0
            condition_images.append(
                F.interpolate(image_tensor, size=(cond_h, cond_w), mode="bilinear", align_corners=False)
            )
            neg_condition_images.append(-torch.ones_like(condition_images[-1]))
            vae_images.append(
                F.interpolate(image_tensor, size=(vae_h, vae_w), mode="bilinear", align_corners=False).unsqueeze(2)
            )
            neg_vae_images.append(-torch.ones_like(vae_images[-1]))
            vae_image_sizes.append((vae_w, vae_h))

        prompt_embeds, prompt_mask = encode_prompt(
            pipeline,
            image=condition_images,
            prompt=prompt,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=inputs.get("max_sequence_length", 512),
        )

        negative_embeds = None
        negative_mask = None
        if do_true_cfg:
            negative_embeds, negative_mask = encode_prompt(
                pipeline,
                image=neg_condition_images,
                prompt=negative_prompt,
                prompt_embeds=None,
                prompt_embeds_mask=None,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=inputs.get("max_sequence_length", 512),
            )

        num_channels_latents = pipeline.transformer.config.in_channels // 4
        latents, image_latents = pipeline.prepare_latents(
            vae_images,
            batch_size,
            num_channels_latents,
            height,
            width,
            pipeline.vae.dtype,
            device,
            inputs.get("generator"),
            inputs.get("latents"),
        )
        latents = latents.to(prompt_embeds.dtype)
        if image_latents is not None:
            image_latents = image_latents.to(prompt_embeds.dtype)
        if do_true_cfg:
            _, neg_image_latents = pipeline.prepare_latents(
                neg_vae_images,
                batch_size,
                num_channels_latents,
                height,
                width,
                pipeline.vae.dtype,
                device,
                inputs.get("generator", None),
                inputs.get("latents", None),
            )
            neg_image_latents = neg_image_latents.to(prompt_embeds.dtype)


        img_shapes = [[
            (1, height // vae_scale_factor // 2, width // vae_scale_factor // 2),
            *[
                (1, vae_h // vae_scale_factor // 2, vae_w // vae_scale_factor // 2)
                for vae_w, vae_h in vae_image_sizes
            ],
        ]] * batch_size

        num_inference_steps = int(inputs.get("num_inference_steps", 50))
        sigmas = inputs.get("sigmas")
        if sigmas is None:
            sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)

        image_seq_len = latents.shape[1]
        mu = _calculate_shift(
            image_seq_len,
            pipeline.scheduler.config.get("base_image_seq_len", 256),
            pipeline.scheduler.config.get("max_image_seq_len", 4096),
            pipeline.scheduler.config.get("base_shift", 0.5),
            pipeline.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = _retrieve_timesteps(
            pipeline.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        del num_inference_steps

        guidance_scale = float(inputs.get("guidance_scale", 1.0))
        if getattr(pipeline.transformer.config, "guidance_embeds", False):
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
        else:
            guidance = None

        txt_seq_lens = prompt_mask.sum(dim=1).tolist() if prompt_mask is not None else None
        neg_txt_seq_lens = negative_mask.sum(dim=1).tolist() if negative_mask is not None else None

        pipeline.scheduler.set_begin_index(0)
        for timestep_value in timesteps:
            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
                if do_true_cfg:
                    neg_latent_model_input = torch.cat([latents.clone(), neg_image_latents], dim=1)


            timestep = timestep_value.expand(latents.shape[0]).to(latents.dtype)

            with pipeline.transformer.cache_context("cond"):
                noise_pred = pipeline.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=prompt_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    attention_kwargs=getattr(pipeline, "_attention_kwargs", None),
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                with pipeline.transformer.cache_context("uncond"):
                    negative_noise_pred = pipeline.transformer(
                        hidden_states=neg_latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=negative_mask,
                        encoder_hidden_states=negative_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=neg_txt_seq_lens,
                        attention_kwargs=getattr(pipeline, "_attention_kwargs", None),
                        return_dict=False,
                    )[0]
                negative_noise_pred = negative_noise_pred[:, : latents.size(1)]
                combined_pred = negative_noise_pred + true_cfg_scale * (noise_pred - negative_noise_pred)

                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(combined_pred, dim=-1, keepdim=True)
                noise_pred = combined_pred * (cond_norm / (noise_norm + 1e-12))

            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(noise_pred, timestep_value, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                latents = latents.to(latents_dtype)

        output_type = inputs.get("output_type", "np")
        if output_type == "latent":
            images = latents
        else:
            latents = pipeline._unpack_latents(latents, height, width, vae_scale_factor).to(pipeline.vae.dtype)

            latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(
                1, pipeline.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(
                1, pipeline.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)
            latents = latents / latents_std + latents_mean

            images = pipeline.vae.decode(latents, return_dict=False)[0][:, :, 0]
            images = pipeline.image_processor.postprocess(images, output_type=output_type)

        pipeline.maybe_free_model_hooks()
    return images


def default_inputs(image_path: str) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    if image.size != (VAE_IMAGE_SIZE, VAE_IMAGE_SIZE):
        image = image.resize((VAE_IMAGE_SIZE, VAE_IMAGE_SIZE), Image.LANCZOS)
    return {
        "image": image,
        "prompt": DEFAULT_PROMPT,
        "generator": None,
        "true_cfg_scale": 1.2,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_inference_steps": 50,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="examples/face.png")
    args = parser.parse_args()

    # load pipeline
    pipeline = load_pipeline(
        "./pretrained_ckpt/Qwen-Image-Edit-2511/",
        "./pretrained_ckpt/TOPOS-Texture/",
        device="cuda",
        dtype=torch.bfloat16,
    )

    # inference
    inputs = default_inputs(args.image_path)
    outputs = run_edit_plus(pipeline, inputs)
    pred = (outputs[0] * 255).astype(np.uint8)
    input_img = np.asarray(inputs['image'].convert("RGB").resize((pred.shape[1], pred.shape[0]), Image.LANCZOS))
    output_path = os.path.join("logs", "Texture", os.path.basename(args.image_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(np.concatenate([pred, input_img], axis=1)).save(output_path)
