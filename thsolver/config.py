import argparse
import os
import shutil
import sys
from datetime import datetime

from yacs.config import CfgNode as CN

_C = CN()

# ---- SOLVER ----
_C.SOLVER = CN()
_C.SOLVER.alias = ''
_C.SOLVER.gpu = (0,)
_C.SOLVER.gpu_num = 0
_C.SOLVER.run = 'train'
_C.SOLVER.logdir = 'logs'
_C.SOLVER.ckpt = ''
_C.SOLVER.ckpt_num = 10

_C.SOLVER.type = 'adamw'
_C.SOLVER.weight_decay = 0.01
_C.SOLVER.clip_grad = -1.0
_C.SOLVER.max_epoch = 1000
_C.SOLVER.warmup_epoch = 20
_C.SOLVER.warmup_init = 0.001
_C.SOLVER.log_per_iter = 10

_C.SOLVER.lr_type = 'poly'
_C.SOLVER.lr = 1e-3
_C.SOLVER.lr_min = 1e-4
_C.SOLVER.gamma = 0.1
_C.SOLVER.milestones = (120, 180)
_C.SOLVER.lr_power = 0.9
_C.SOLVER.ema = False

_C.SOLVER.dist_url = 'tcp://localhost:10001'
_C.SOLVER.progress_bar = True
_C.SOLVER.rand_seed = -1
_C.SOLVER.empty_cache = True

# ---- DATA ----
_C.DATA = CN()

# Training data
_C.DATA.train = CN()
_C.DATA.train.depth = 8
_C.DATA.train.edge_file = ''
_C.DATA.train.cluster_file = ''
_C.DATA.train.template_vertices_path = ''
_C.DATA.train.root_dir = './'
_C.DATA.train.use_surface = True
_C.DATA.train.use_images = False
_C.DATA.train.max_rotation = 0.0
_C.DATA.train.max_translate = 0.0
_C.DATA.train.scale_range = [1.0, 1.0]
_C.DATA.train.image_nums = 8
_C.DATA.train.filelist = ''
_C.DATA.train.batch_size = 1
_C.DATA.train.num_workers = 4
_C.DATA.train.shuffle = True

# Inference: topology assets for rebuilding the template HGraph + per-mode
# input sources. There is no filelist for inference.
_C.DATA.test = CN()
_C.DATA.test.depth = 8
_C.DATA.test.edge_file = ''
_C.DATA.test.cluster_file = ''
_C.DATA.test.template_vertices_path = ''
_C.DATA.test.test_pointcloud_path = ''  # VAE single-pointcloud inference
_C.DATA.test.test_pointcloud_dir = ''   # VAE batch-pointcloud inference
_C.DATA.test.test_image_path = ''       # DiT single-image inference
_C.DATA.test.test_image_dir = ''        # DiT batch-image inference
_C.DATA.test.seeds = []                 # Optional seeds for multi-sample DiT inference

# ---- MODEL ----
_C.MODEL = CN()
_C.MODEL.base_mesh = ''
_C.MODEL.compile = True
_C.MODEL.sync_bn = False
_C.MODEL.find_unused_parameters = False

# VAE
_C.MODEL.vae_out_channel = 6
_C.MODEL.code_channel = 32
_C.MODEL.num_latents = 1513
_C.MODEL.width = 128
_C.MODEL.num_encoder_layers = 8
_C.MODEL.point_feats = 3
_C.MODEL.num_freqs = 8
_C.MODEL.include_pi = False
_C.MODEL.include_distance = False
_C.MODEL.decoder_blocks = []
_C.MODEL.decoder_channel = []

# DiT
_C.MODEL.vae_ckpt = ''
_C.MODEL.df_in_channels = 32
_C.MODEL.df_out_channels = 32
_C.MODEL.cond_channels = 1024
_C.MODEL.model_channels = 1024
_C.MODEL.num_blocks = 24
_C.MODEL.num_heads = 16
_C.MODEL.num_head_channels = 64
_C.MODEL.mlp_ratio = 4
_C.MODEL.pe_mode = 'ape'
_C.MODEL.qk_rms_norm = True
_C.MODEL.qk_rms_norm_cross = False
_C.MODEL.use_fp16 = False
_C.MODEL.use_checkpoint = False
_C.MODEL.p_uncond = 0.1
_C.MODEL.ema_rate = 0.9999

# Image encoder
_C.MODEL.image_cond_model_name = 'dinov2_vitl14_reg'
_C.MODEL.image_cond_model_dir = './pretrained_ckpt/facebookresearch_dinov2_main'
_C.MODEL.image_cond_model_path = './pretrained_ckpt/dinov2_vitl14_reg4_pretrain.pth'

# Sampling (DiT inference)
_C.MODEL.cfg_strength = 3.0
_C.MODEL.sample_steps = 50

# ---- LOSS ----
_C.LOSS = CN()
_C.LOSS.template_dir = ''
_C.LOSS.faces_path = ''
_C.LOSS.train_batch_size = 1
_C.LOSS.kl_weight = 0.001

# ---- SYS ----
_C.SYS = CN()
_C.SYS.cmds = ''

FLAGS = _C


def _update_config(FLAGS, args):
    FLAGS.defrost()
    if args.config:
        config_path = args.config
        base_name = os.path.basename(config_path).split('_')[0]
        base_config_path = os.path.join(os.path.dirname(config_path), f'{base_name}.yaml')
        if os.path.exists(base_config_path) and os.path.abspath(base_config_path) != os.path.abspath(config_path):
            FLAGS.merge_from_file(base_config_path)
        FLAGS.merge_from_file(config_path)

    if args.opts:
        FLAGS.merge_from_list(args.opts)
    FLAGS.SYS.cmds = ' '.join(sys.argv)

    if FLAGS.SOLVER.gpu_num > 0 and FLAGS.SOLVER.gpu == (0,):
        FLAGS.SOLVER.gpu = tuple(range(FLAGS.SOLVER.gpu_num))

    FLAGS.LOSS.train_batch_size = FLAGS.DATA.train.batch_size

    alias = FLAGS.SOLVER.alias.lower()
    if 'time' in alias:
        alias = alias.replace('time', datetime.now().strftime('%m%d%H%M'))
    if alias != '':
        FLAGS.SOLVER.logdir += '_' + alias
    FLAGS.freeze()


def _backup_config(FLAGS, args):
    logdir = FLAGS.SOLVER.logdir
    os.makedirs(logdir, exist_ok=True)
    if args.config:
        shutil.copy2(args.config, logdir)
    with open(os.path.join(logdir, 'all_configs.yaml'), 'w') as fid:
        fid.write(FLAGS.dump())


def get_config():
    return FLAGS


def parse_args(backup=True):
    parser = argparse.ArgumentParser(description='Training / inference configuration')
    parser.add_argument('--config', type=str, help='YAML config file path')
    parser.add_argument('opts', nargs=argparse.REMAINDER,
                        help='Override config options on the command line')
    args = parser.parse_args()
    _update_config(FLAGS, args)
    if backup:
        _backup_config(FLAGS, args)
    return FLAGS
