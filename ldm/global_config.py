import torch

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import load_model
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_CONFIG_PATH = 'configs/refldm.yaml'
MODEL_CKPT_PATH = 'ckpts/refldm.ckpt'
VAE_CKPT_PATH = 'ckpts/vqgan.ckpt'

MODEL = load_model(MODEL_CONFIG_PATH, MODEL_CKPT_PATH, VAE_CKPT_PATH, DEVICE)
SAMPLER = DDIMSampler(MODEL, print_tqdm=False, schedule='uniform_trailing')