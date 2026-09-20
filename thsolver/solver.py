import os
import random
import time

import numpy as np
import torch
import torch.distributed
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import parse_args
from .lr_scheduler import get_lr_scheduler
from .sampler import DistributedInfSampler, InfSampler
from .tracker import AverageTracker


class Solver:
    """Tiny training framework: train on a filelist, inference on raw inputs.

    Run modes are dispatched via ``SOLVER.run``. There is no filelist-based
    test mode by design.

    Currently supported run modes:
      - ``train``                : iterate the training filelist and back-prop.
      - ``test_image``           : DiT — single image -> mesh.
      - ``test_image_dir``       : DiT — every image in a directory -> meshes.
      - ``test_pointcloud``      : VAE — single .npz point cloud -> mesh.
      - ``test_pointcloud_dir``  : VAE — every .npz in a directory -> meshes.
    """

    def __init__(self, FLAGS, is_master: bool = True):
        self.FLAGS = FLAGS
        self.is_master = is_master
        self.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else len(FLAGS.SOLVER.gpu)
        self.device = torch.cuda.current_device()
        self.disable_tqdm = not (is_master and FLAGS.SOLVER.progress_bar)
        self.start_epoch = 1
        self.ema = FLAGS.SOLVER.ema

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.summary_writer = None

    # ---- subclass hooks ----
    def get_model(self, flags):
        raise NotImplementedError

    def get_loss_function(self, flags):
        raise NotImplementedError

    def get_dataset(self, flags):
        raise NotImplementedError

    def train_step(self, batch):
        raise NotImplementedError

    # Inference hooks — implemented by the model-specific solver:
    def test_single_pointcloud(self):
        raise NotImplementedError

    def test_dir_pointclouds(self):
        raise NotImplementedError

    def test_single_image(self):
        raise NotImplementedError

    def test_dir_images(self):
        raise NotImplementedError

    # ---- training dataloader ----
    def config_train_dataloader(self):
        flags_train = self.FLAGS.DATA.train
        dataset, collate_fn = self.get_dataset(flags_train)

        if self.world_size > 1:
            sampler = DistributedInfSampler(dataset, shuffle=flags_train.shuffle)
        else:
            sampler = InfSampler(dataset, shuffle=flags_train.shuffle)

        self.train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=flags_train.batch_size,
            num_workers=flags_train.num_workers,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=flags_train.num_workers > 0,
            prefetch_factor=4 if flags_train.num_workers > 0 else None,
        )
        self.train_iter = iter(self.train_loader)

    # ---- model / optimizer ----
    def config_model(self):
        flags = self.FLAGS.MODEL
        model = self.get_model(flags)
        model.cuda(device=self.device)
        if self.world_size > 1:
            if flags.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[self.device],
                output_device=self.device, broadcast_buffers=False,
                find_unused_parameters=flags.find_unused_parameters)
        self.model = model

    def config_loss_fn(self):
        self.get_loss_function(self.FLAGS.LOSS)

    def config_optimizer(self):
        flags = self.FLAGS.SOLVER
        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = flags.type.lower()
        if opt == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=flags.lr, weight_decay=flags.weight_decay, momentum=0.9)
        elif opt == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=flags.lr, weight_decay=flags.weight_decay)
        elif opt == 'adamw':
            self.optimizer = torch.optim.AdamW(params, lr=flags.lr, weight_decay=flags.weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer: {flags.type}')

    def config_lr_scheduler(self):
        self.scheduler = get_lr_scheduler(self.optimizer, self.FLAGS.SOLVER)

    def configure_log(self, set_writer: bool = True):
        self.logdir = self.FLAGS.SOLVER.logdir
        self.ckpt_dir = os.path.join(self.logdir, 'checkpoints')
        self.train_log_file = os.path.join(self.logdir, 'train_log.csv')
        self.result_dir = os.path.join(self.logdir, 'results')

        if self.is_master:
            tqdm.write('Logdir: ' + self.logdir)

        if self.is_master and set_writer:
            self.summary_writer = SummaryWriter(self.logdir, flush_secs=20)
            os.makedirs(self.ckpt_dir, exist_ok=True)

    # ---- training loop ----
    def train_epoch(self, epoch: int):
        self.model.train()
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)

        tick = time.time()
        elapsed = {}
        tracker = AverageTracker()
        dataloader_length = len(self.train_loader)
        log_per_iter = self.FLAGS.SOLVER.log_per_iter

        for it in tqdm(range(dataloader_length), ncols=80, leave=False, disable=self.disable_tqdm):
            batch = self.train_iter.__next__()
            batch['iter_num'] = it
            batch['epoch'] = epoch
            elapsed['time/data'] = torch.Tensor([time.time() - tick])

            self.optimizer.zero_grad()
            output = self.train_step(batch)
            output['total_loss'].backward()

            if self.FLAGS.SOLVER.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.FLAGS.SOLVER.clip_grad)
            self.optimizer.step()

            elapsed['time/batch'] = torch.Tensor([time.time() - tick])
            tick = time.time()
            output.update(elapsed)
            tracker.update(output)

            if it % 50 == 0 and self.FLAGS.SOLVER.empty_cache:
                torch.cuda.empty_cache()

            total_iter = (epoch - 1) * dataloader_length + it + 1
            if self.is_master and log_per_iter > 0 and it % log_per_iter == 0:
                tracker.log(epoch, summary_writer=self.summary_writer,
                            log_file=self.train_log_file, msg_tag='- ',
                            notes=f'iter: {it}', print_time=False,
                            total_iter=total_iter)

        if self.world_size > 1:
            tracker.average_all_gather()
        if self.is_master:
            tracker.log(epoch)

    # ---- checkpoints ----
    def save_checkpoint(self, epoch: int):
        if not self.is_master:
            return
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        prefix = os.path.join(self.ckpt_dir, '%05d' % epoch)
        torch.save(model_dict, prefix + '.model.pth')
        torch.save({
            'model_dict': model_dict,
            'epoch': epoch,
            'optimizer_dict': self.optimizer.state_dict(),
            'scheduler_dict': self.scheduler.state_dict(),
        }, prefix + '.solver.tar')

    def load_checkpoint(self):
        ckpt = self.FLAGS.SOLVER.ckpt
        if not ckpt and os.path.isdir(self.ckpt_dir):
            ckpts = sorted(f for f in os.listdir(self.ckpt_dir) if f.endswith('solver.tar'))
            if ckpts:
                ckpt = os.path.join(self.ckpt_dir, ckpts[-1])
        if not ckpt:
            return

        trained = torch.load(ckpt, map_location='cuda')
        if ckpt.endswith('.solver.tar'):
            model_dict = trained['model_dict']
            self.start_epoch = trained['epoch'] + 1
            if self.optimizer:
                self.optimizer.load_state_dict(trained['optimizer_dict'])
            if self.scheduler:
                self.scheduler.load_state_dict(trained['scheduler_dict'])
        else:
            model_dict = trained

        prefix = '_orig_mod.'
        target = self.model.module if self.world_size > 1 else self.model
        target_has_prefix = next(iter(target.state_dict().keys()), '').startswith(prefix)
        ckpt_has_prefix = model_dict and next(iter(model_dict.keys())).startswith(prefix)
        if ckpt_has_prefix and not target_has_prefix:
            model_dict = {k[len(prefix):]: v for k, v in model_dict.items()}
        elif not ckpt_has_prefix and target_has_prefix:
            model_dict = {prefix + k: v for k, v in model_dict.items()}
        target.load_state_dict(model_dict)

        if self.is_master:
            tqdm.write(f'Loaded checkpoint: {ckpt} (start_epoch={self.start_epoch})')

    def manual_seed(self, rand_seed=None):
        if rand_seed is None:
            rand_seed = self.FLAGS.SOLVER.rand_seed
        if rand_seed > 0:
            random.seed(rand_seed)
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    # ---- entrypoints ----
    def train(self):
        self.manual_seed()
        self.config_model()
        self.config_loss_fn()
        self.config_train_dataloader()
        self.config_optimizer()
        self.config_lr_scheduler()
        self.configure_log()
        self.load_checkpoint()

        for epoch in tqdm(range(self.start_epoch, self.FLAGS.SOLVER.max_epoch + 1),
                          ncols=80, disable=self.disable_tqdm):
            self.train_epoch(epoch)
            self.scheduler.step()
            if self.is_master:
                self.summary_writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], epoch)
            self.save_checkpoint(epoch)

        if self.world_size > 1:
            torch.distributed.barrier()

    def _prepare_for_inference(self):
        self.manual_seed()
        self.config_model()
        self.configure_log(set_writer=False)
        self.load_checkpoint()

    def test_pointcloud(self):
        self._prepare_for_inference()
        self.test_single_pointcloud()

    def test_pointcloud_dir(self):
        self._prepare_for_inference()
        self.test_dir_pointclouds()

    def test_image(self):
        self._prepare_for_inference()
        self.test_single_image()

    def test_image_dir(self):
        self._prepare_for_inference()
        self.test_dir_images()

    def run(self):
        getattr(self, self.FLAGS.SOLVER.run)()

    @classmethod
    def worker(cls, rank, FLAGS):
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            local_rank = rank
            world_size = len(FLAGS.SOLVER.gpu)

        torch.cuda.set_device(local_rank)
        if world_size > 1:
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

        cls(FLAGS, is_master=(rank == 0)).run()

    @classmethod
    def main(cls):
        FLAGS = parse_args()
        cls.worker(0, FLAGS)
