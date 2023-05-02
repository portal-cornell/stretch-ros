import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent.parent))

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
from tqdm import trange

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

from .misc_utils import get_end_eff

TensorBatch = List[torch.Tensor]


class ReplayBuffer(Dataset):
    def __init__(
        self,
        image_dim: int,
        joint_state_dim: int,
        action_dim: int,
        skill_data_dir: str,
        buffer_size: int,
        **kwargs,
    ):
        super(ReplayBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.pointer = 0
        self.size = 0
        self.action_dim = action_dim

        self.images = []
        self.joint_states = torch.zeros(
            (buffer_size, joint_state_dim), dtype=torch.float32
        )
        self.actions = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.next_images = []
        self.next_joint_states = torch.zeros(
            (buffer_size, joint_state_dim), dtype=torch.float32
        )
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32)
        self.skill_data_dir = skill_data_dir
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.augment = transforms.Compose(
            [
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomInvert(p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2
                ),
            ]
        )
        self.dataset_type = kwargs.get("dataset_type", None)

    def __len__(self):
        return self.size

    def __getitem__(self, indices):
        images = self.images[indices]
        images = self.augment(self.preprocess(Image.open(images)))

        next_images = self.next_images[indices]
        next_images = self.augment(self.preprocess(Image.open(next_images)))

        joint_states = self.joint_states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_joint_states = self.next_joint_states[indices]
        dones = self.dones[indices]

        sample = dict(
            images=images,
            joint_states=joint_states,
            actions=actions,
            rewards=rewards,
            next_images=next_images,
            next_joint_states=next_joint_states,
            dones=dones,
        )
        return sample

    def get_batch_by_indices(self, indices):
        return self.__getitem__(indices)

    # Loads data in csv format
    def load_dataset(self, data):
        if self.size != 0:
            return
            # raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = len(data) - 1
        self.size += n_transitions
        self.pointer = min(self.size, n_transitions)
        if n_transitions != self.buffer_size:
            raise ValueError(
                f"{self.buffer_size=} does not match loaded {n_transitions=}"
            )

        rewards = data["rewards"][:n_transitions].to_numpy()
        dones = data["dones"][:n_transitions].to_numpy()

        self.rewards = self._to_tensor(rewards, self.rewards)
        self.dones = self._to_tensor(dones, self.dones)

        imgs, js_ees, kps = [], [], []
        for idx in trange(n_transitions + 1):
            img_path = Path(self.skill_data_dir, data.loc[idx, "image_path"])
            # image = Image.open(img_path)
            # image = self.resnet_transform(image).cpu()
            # image = self.preprocess(image)
            joint_states = torch.from_numpy(
                np.array(
                    data.loc[idx, "gripper_aperture_pos":"wrist_extension_eff"],
                    dtype=np.float32,
                )
            )
            end_effs = get_end_eff(joint_states)
            js_ee = torch.cat((joint_states, end_effs))
            key_pressed = torch.from_numpy(np.array([data.loc[idx, "key_pressed"]]))
            # key_pressed = (
            #     F.one_hot(key_pressed, num_classes=self.action_dim)
            #     .squeeze()
            #     .to(torch.float32)
            # )

            imgs.append(img_path)
            js_ees.append(js_ee)
            kps.append(key_pressed)

        # self.images = self._to_tensor(imgs[:-1], self.images)
        self.images = imgs[:-1]
        self.joint_states = torch.clone(self._to_tensor(js_ees[:-1], self.joint_states))
        self.actions = self._to_tensor(kps[:-1], self.actions)
        self.next_images = imgs[1:]
        # self.next_images = self._to_tensor(imgs[1:], self.next_images)
        self.next_joint_states = torch.clone(
            self._to_tensor(js_ees[1:], self.next_joint_states)
        )

        print(f"Dataset size: {n_transitions}")

    def _to_tensor(self, data, target_tensor) -> torch.Tensor:
        if isinstance(data[0], torch.Tensor):
            data = torch.stack(data)
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        else:
            raise ValueError(f"Unsupport type {type(data)=} and {type(data[0])=}")

        if data.size() != target_tensor.size():
            data.reshape_as(target_tensor)
        return data
