# The Unreasonable Effectiveness of Text Embedding Interpolation for Continuous Image Steering

[Yigit Ekin](https://yigitekin.github.io/)<sup>1</sup>, [Yossi Gandelsman](https://yossigandelsman.github.io/)<sup>1</sup>

<sup>1</sup>Reve

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Project Website](https://img.shields.io/badge/Project_website-red.svg)](https://yigitekin.github.io/diffusion-sliders/) [![arXiv](https://img.shields.io/badge/arXiv-2505.13344-b31b1b.svg)](https://arxiv.org/abs/2603.17998)

---

![Teaser](assets/teaser.png)

We present a training-free framework for continuous and controllable image editing at test time for text-conditioned generative models. In contrast to prior approaches that rely on additional training or manual user intervention, we find that a simple steering in the text-embedding space is sufficient to produce smooth edit control. Given a target concept (e.g., enhancing photorealism or changing facial expression), we use a large language model to automatically construct a small set of debiased contrastive prompt pairs, from which we compute a steering vector in the generator's text-encoder space. We then add this vector directly to the input prompt representation to control generation along the desired semantic axis. To obtain a continuous control, we propose an elastic range search procedure that automatically identifies an effective interval of steering magnitudes, avoiding both under-steering (no-edit) and over-steering (changing other attributes). Adding the scaled versions of the same vector within this interval yields smooth and continuous edits. Since our method modifies only textual representations, it naturally generalizes across text-conditioned modalities, including image and video generation. To quantify the steering continuity, we introduce a new evaluation metric that measures the uniformity of semantic change across edit strengths. We compare the continuous editing behavior across methods and find that, despite its simplicity and lightweight design, our approach is comparable to training-based alternatives, outperforming other training-free methods.

## Prompting Advice

Since we start from the original model's edit, the initial prompt has a significant impact on edit quality. We recommend the following formats:

**Stylization** — `without changing the layout of the scene and background, make the scene <style>`
> e.g. `without changing the layout of the scene and background, make the scene cartoon`

**Global edit** — `without changing the layout of the scene and background, make the object <edit>`
> e.g. `without changing the layout of the scene and background, make the scene in night time`

**Local edit (with person)** — `without changing the layout of the scene and the person's identity, make the person <edit>`
> e.g. `without changing the layout of the scene and the person's identity, make the person smile`

**Local edit (with object)** — `without changing the layout of the scene and the <object>, make the <object> <edit>`
> e.g. `without changing the layout of the scene and the motorcycle, make the motorcycle rusty`

TLDR: Anchoring the prompt with phrases like *without changing the layout of the scene*, *background*, or *the person's identity* helps preserve unrelated attributes and improves consistency across steering strengths.

## Setup

```bash
conda create -n sliders python=3.10 -y
conda activate sliders
pip install -r requirements.txt
```

For Qwen, download the Lora model weights as:

```bash
wget -O ckpts/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors
```
All other weights will be downloaded automatically.

Set your OpenAI API key (used for dataset generation and token selection):

```bash
export OPENAI_API_KEY=your_key_here
```

For token selection with a custom endpoint (e.g. a self-hosted Qwen model):

```bash
export OPENAI_BASE_URL=http://your-endpoint/v1
```

If you do not want to use OpenAI API key for debiased contrastive dataset. You can create one on your own. See [Debiased Dataset Generation and Steering Vector Computation](#debiased-dataset-generation-and-steering-vector-computation).

## End-to-End Usage

All pipelines share the same core workflow and are controlled via the provided shell scripts. Each step is skipped automatically if its output already exists, so partial runs can be resumed.

### Image editing (Flux 2 / Qwen)

`ELASTIC_BAND_CONFIG` selects the hyperparameter preset. Pick the one that matches your edit type:

| Config | Edit type | Description |
|--------|-----------|-------------|
| `configs/flux2_global.yaml` | Flux 2 — global/style | Whole-image edits (e.g. "make it cartoon") |
| `configs/flux2_local.yaml` | Flux 2 — local | Object/region edits (e.g. "make the hat red") |
| `configs/qwen_global.yaml` | Qwen — global/style | Whole-image edits |
| `configs/qwen_local.yaml` | Qwen — local | Object/region edits |

```bash
# Flux 2 — global/style edit
CONCEPT=cartoon \
PROMPT="make the scene cartoon" \
INPUT_IMAGE=photo.png \
ELASTIC_BAND_CONFIG=configs/flux2_global.yaml \
bash run_flux2.sh

# Flux 2 — local edit
CONCEPT=hat_color \
PROMPT="make the hat red" \
INPUT_IMAGE=photo.png \
ELASTIC_BAND_CONFIG=configs/flux2_local.yaml \
bash run_flux2.sh

# Qwen Image Edit — global/style edit
CONCEPT=beard \
PROMPT="without changing the identity of the person and the layout of the scene, make the person bearded" \
INPUT_IMAGE=0.png \
ELASTIC_BAND_CONFIG=configs/qwen_local.yaml \
bash run_qwen.sh

# Qwen Image Edit — local edit
CONCEPT=hat_color \
PROMPT="make the hat red" \
INPUT_IMAGE=photo.png \
ELASTIC_BAND_CONFIG=configs/qwen_local.yaml \
bash run_qwen.sh
```

Results are written to `outputs/{concept}/`. Key env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `ELASTIC_BAND_CONFIG` | *(required)* | Path to elastic-band YAML config |
| `OUT_DIR` | `outputs` | Root output directory |
| `NUM_EXAMPLES` | `100` | Contrastive pairs to generate |
| `NUM_OUTPUTS` | `8` | Images in the final inference grid |
| `OPENAI_MODEL` | `gpt-4o` | LLM for dataset generation |
| `TOKEN_MODEL` | `Qwen/Qwen3-8B` | LLM for token selection |
| `SEED` | `0` / `42` | Generation seed |

### Video generation (Wan)

Wan uses the same dataset + token-selection + steering-vector steps but skips elastic-band search. The steering strength range is set directly via `STRENGTH_MIN` / `STRENGTH_MAX`.

```bash
CONCEPT=boxing \
PROMPT="Two anthropomorphic cats boxing on a stage." \
bash run_wan.sh
```

Results are written to `outputs/{concept}/inference/` as `.mp4` files plus a `grid.png` of first frames. Key env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `OUT_DIR` | `outputs` | Root output directory |
| `NUM_EXAMPLES` | `100` | Contrastive pairs to generate |
| `NUM_OUTPUTS` | `5` | Videos in the final inference grid |
| `STRENGTH_MIN` | `0.0` | Minimum steering strength |
| `STRENGTH_MAX` | `5.0` | Maximum steering strength |
| `MODEL_ID` | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | Wan model |
| `HEIGHT` / `WIDTH` | `720` / `1280` | Output resolution (use `480`/`832` + `FLOW_SHIFT=3.0` for 480p) |
| `OPENAI_MODEL` | `gpt-4o` | LLM for dataset generation |
| `TOKEN_MODEL` | `Qwen/Qwen3-8B` | LLM for token selection |
| `SEED` | `42` | Generation seed |

## Debiased Dataset Generation and Steering Vector Computation

The first step generates a contrastive JSONL dataset via the OpenAI API. Each record contains a positive/negative prompt pair with corresponding style labels used to compute a difference-of-means steering vector from the model's text encoder.

> **No API key?** You can skip the API call entirely by copying the prompt template from [`system_prompts/unbiased_dataset_generation.txt`](system_prompts/unbiased_dataset_generation.txt) and pasting it (with your concept substituted in) into any LLM UI (ChatGPT, Claude, Gemini, etc.). Save the output as `outputs/{model}/{concept}/{concept}.jsonl` (e.g. `outputs/flux2/smile/smile.jsonl`) and pass it directly to the vector computation step via `--pairs_file`.

You can also run these steps individually:

```bash
# Generate dataset only
python -m dataset.generate \
    --concept cartoon --num_examples 100 --out_file outputs/cartoon/cartoon.jsonl

# Compute steering vectors (Flux 2)
python -m models.flux2.compute_vectors \
    --pairs_file outputs/cartoon/cartoon.jsonl \
    --out_dir outputs/cartoon

# Compute steering vectors (Qwen)
python -m models.qwen.compute_vectors \
    --pairs_file outputs/cartoon/cartoon.jsonl \
    --out_dir outputs/cartoon

# Compute steering vectors (Wan)
python -m models.wan.compute_vectors \
    --pairs_file outputs/boxing/boxing.jsonl \
    --out_dir outputs/boxing
```

Output files saved to `outputs/{concept}/`:
- `{concept}.jsonl` — contrastive prompt pairs
- `steering_last_layer.npy` — unit-norm steering vector
- `max_projection_value.npy` / `min_projection_value.npy` — projection statistics

## LLM Assisted Token Selection

An LLM identifies the minimal set of tokens in your prompt that anchor the steering concept. Tokens are categorized as belonging to a *Local*, *Stylization*, or *Global* edit and selected accordingly. By default this uses `Qwen3-8B` with thinking enabled, but you can substitute a different model via `TOKEN_MODEL`. If you prefer not to use the API at all, you can run the token selection prompt manually in any LLM UI and provide the result via `--tokens_to_edit`.

```bash
python -m dataset.select_tokens \
    --prompt "make the scene cartoon" \
    --concept cartoon
# → cartoon
```

The result is saved to `outputs/{concept}/tokens_to_edit.json` during the full pipeline run.

## Elastic Band Search

The elastic-band algorithm places control points along the steering strength axis `[a_min, 0]` so that consecutive generated images are roughly equidistant in perceptual space (DreamSim). It automatically finds the valid operating range — the largest contiguous interval where every steered image stays within a perceptual distance threshold of the unsteered reference.

Hyperparameters are loaded from a YAML config passed via `--config`. Choose the preset that matches your edit type:

```bash
# Flux 2 — standalone, global edit
python -m models.flux2.elastic_band \
    --config configs/flux2_global.yaml \
    --input_image path/to/image.png \
    --prompt "make the scene cartoon" \
    --tokens_to_edit cartoon \
    --steering_vector_dir outputs/cartoon

# Flux 2 — standalone, local edit
python -m models.flux2.elastic_band \
    --config configs/flux2_local.yaml \
    --input_image path/to/image.png \
    --prompt "make the hat red" \
    --tokens_to_edit hat \
    --steering_vector_dir outputs/hat_color

# Qwen — standalone, global edit
python -m models.qwen.elastic_band \
    --config configs/qwen_global.yaml \
    --input_image path/to/image.png \
    --prompt "make the scene cartoon" \
    --tokens_to_edit cartoon \
    --steering_vector_dir outputs/cartoon

# Qwen — standalone, local edit
python -m models.qwen.elastic_band \
    --config configs/qwen_local.yaml \
    --input_image path/to/image.png \
    --prompt "make the hat red" \
    --tokens_to_edit hat \
    --steering_vector_dir outputs/hat_color
```

Individual parameters can be overridden on the command line and take precedence over the config file, e.g. `--target_gap 0.05 --epsilon 0.2`.

Results are saved to `outputs/{concept}/elastic_band/summary.json`, which records the valid steering range used for final inference.

> **Wan** does not use elastic-band search. Run inference directly with explicit strengths:
>
> ```bash
> python -m models.wan.inference \
>     --prompt "Two anthropomorphic cats boxing on a stage." \
>     --tokens_to_edit boxing cats \
>     --steering_vector outputs/boxing/steering_last_layer.npy \
>     --strengths 0 1 2 3 4 \
>     --out_dir outputs/boxing/inference
> ```

---

If you found our work useful, please don't forget to cite our work.

```bibtex
@misc{ekin2026unreasonableeffectivenesstextembedding,
      title={The Unreasonable Effectiveness of Text Embedding Interpolation for Continuous Image Steering}, 
      author={Yigit Ekin and Yossi Gandelsman},
      year={2026},
      eprint={2603.17998},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.17998}, 
}
}
```

## Acknowledgements

Our codebase is built on top of [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers), [Accelerate](https://github.com/huggingface/accelerate), and [DreamSim](https://github.com/ssundaram21/dreamsim). We thank the authors of these libraries for their excellent work and for making them publicly available.

