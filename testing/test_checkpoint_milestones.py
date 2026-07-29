import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from nerf.utils import Trainer


class TinyModel(torch.nn.Module):
    cuda_ray = False

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, inputs):
        return self.layer(inputs)


def make_options(workspace, milestone_steps):
    return SimpleNamespace(
        workspace=str(workspace),
        patch_size=1,
        rand_pose=-1,
        entropy=False,
        smoothing=False,
        milestone_steps=list(milestone_steps),
    )


class CheckpointMilestoneTest(unittest.TestCase):
    def test_full_milestone_survives_resume_and_latest_rotation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory) / 'workspace'
            options = make_options(workspace, [3])
            trainer = Trainer(
                'ngp',
                options,
                TinyModel(),
                device=torch.device('cpu'),
                ema_decay=0.95,
                workspace=str(workspace),
                use_checkpoint='scratch',
                use_tensorboardX=False,
            )
            trainer.epoch = 1
            trainer.global_step = 3
            trainer.save_checkpoint(full=True)

            latest_path = Path(trainer.last_checkpoint_path)
            milestone_path = Path(trainer.save_milestone_checkpoint())
            self.assertEqual(
                milestone_path,
                workspace / 'checkpoints' / 'milestones' / 'ngp_step_000003.pth',
            )
            self.assertEqual(Path(trainer.last_checkpoint_path), latest_path)
            self.assertEqual(
                Trainer._sha256_file(latest_path),
                Trainer._sha256_file(milestone_path),
            )

            checkpoint = torch.load(milestone_path, map_location='cpu')
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertEqual(checkpoint['global_step'], 3)
            for key in (
                'model',
                'ema',
                'optimizer',
                'lr_scheduler',
                'scaler',
                'rng_state',
                'config',
            ):
                self.assertIn(key, checkpoint)

            resumed_options = make_options(workspace, [6])
            resumed_trainer = Trainer(
                'ngp',
                resumed_options,
                TinyModel(),
                device=torch.device('cpu'),
                ema_decay=0.95,
                workspace=str(workspace),
                use_checkpoint=str(milestone_path),
                use_tensorboardX=False,
            )
            self.assertEqual(resumed_trainer.epoch, 1)
            self.assertEqual(resumed_trainer.global_step, 3)

            resumed_trainer.epoch = 2
            resumed_trainer.global_step = 6
            resumed_trainer.save_checkpoint(full=True)
            second_milestone_path = Path(
                resumed_trainer.save_milestone_checkpoint()
            )

            self.assertTrue(milestone_path.is_file())
            self.assertTrue(second_milestone_path.is_file())
            self.assertFalse(latest_path.exists())
            self.assertFalse(
                resumed_trainer._is_managed_checkpoint_path(milestone_path)
            )
            self.assertTrue(
                resumed_trainer._is_managed_checkpoint_path(
                    resumed_trainer.last_checkpoint_path
                )
            )

    def test_existing_different_milestone_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory) / 'workspace'
            options = make_options(workspace, [3])
            trainer = Trainer(
                'ngp',
                options,
                TinyModel(),
                device=torch.device('cpu'),
                workspace=str(workspace),
                use_checkpoint='scratch',
                use_tensorboardX=False,
            )
            trainer.epoch = 1
            trainer.global_step = 3
            trainer.save_checkpoint(full=True)

            milestone_path = Path(trainer._milestone_checkpoint_path(3))
            milestone_path.parent.mkdir(parents=True, exist_ok=True)
            milestone_path.write_bytes(b'different checkpoint bytes')

            with self.assertRaises(FileExistsError):
                trainer.save_milestone_checkpoint()


if __name__ == '__main__':
    unittest.main()
