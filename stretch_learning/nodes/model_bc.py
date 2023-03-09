from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models

from r3m import load_r3m
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_bc import VisuomotorDataset


class BC(pl.LightningModule):

    def __init__(self, skill_name, joint_state_dims, state_action_dims, training_path,
                 validation_path, use_imgs=True, use_joints=True, use_end_eff=True,
                 compress_img=True, img_comp_dim=8, js_modifications=""):
        super().__init__()
        self.save_hyperparameters()
        self.skill_name = skill_name

        self.conv_net = load_r3m("resnet18")
        self.img_squeeze_linear = nn.Linear(512, img_comp_dim)
        self.conv_net_output_dim = img_comp_dim if compress_img else 512
        self.joint_state_dims = joint_state_dims
        self.end_eff_dim = 2
        
        total_dims = self.conv_net_output_dim + self.joint_state_dims + self.end_eff_dim
        self.fc1 = nn.Linear(total_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_action_dims)

        self.transition_scalar = 100
        self.training_path = training_path
        self.validation_path = validation_path
        self.data_dir = training_path.parent.parent
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=state_action_dims, top_k=1)

        self.num_workers = 1 if str(Path.home()) == "/home/jlr429" else 16
        self.resnet_transform = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.augment = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomInvert(p=0.5),
                                           transforms.RandomVerticalFlip(
                                               p=0.5),
                                           transforms.RandomErasing(p=0.25)])

        self.use_imgs = use_imgs
        self.use_joints = use_joints
        self.use_end_eff = use_end_eff
        self.compress_img = compress_img
        self.js_modifications=js_modifications

    def forward(self, image, js_data):
        batch_size = image.size(0)
        img_t = self.conv_net(image)
        device = img_t.device
        if self.use_imgs and self.compress_img:
                img_t = self.img_squeeze_linear(img_t)
        else:
            img_t = torch.zeros((batch_size, self.conv_net_output_dim)).to(device)
        if self.use_joints:
            js_t = js_data.to(device)
        else:
            js_t = torch.zeros((batch_size, self.joint_state_dims)).to(device)
        if self.use_end_eff:
            ee_t = self._get_end_eff(js_data).to(device)
        else:
            ee_t = torch.zeros((batch_size, self.end_eff_dim)).to(device)
  
        x = torch.cat((js_t, ee_t), dim=1)
        x = torch.cat((img_t, x), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _get_end_eff(self, js):
        lift, pitch = js[:, 27], js[:, 30]
        gripper_len = 0.21
        gripper_delta_y = gripper_len * torch.sin(pitch)
        end_eff_y = lift + gripper_delta_y
        end_eff_x = js[:, 39]
        end_eff_features = torch.cat((end_eff_y.unsqueeze(1), end_eff_x.unsqueeze(1)), dim=1)
        return end_eff_features

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch["image"] = self.augment(batch["image"])
        return batch

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        training_data = VisuomotorDataset(
            data_dir=self.data_dir, csv_file=self.training_path, 
            transform=self.resnet_transform, js_modifications=self.js_modifications)
        train_dataloader = DataLoader(
            training_data, num_workers=self.num_workers, batch_size=32)
        return train_dataloader

    def training_step(self, train_batch, batch_idx):
        image, joint_state, key_pressed = train_batch["image"], train_batch["joint_states"], train_batch["key_pressed"]
        is_transition = train_batch["is_transition"]
        predicted_action = self(image, joint_state)
        actual_action = key_pressed.to(torch.int64).squeeze()
        loss = self.loss_fn(predicted_action, actual_action)       
        loss[is_transition] = loss[is_transition] * self.transition_scalar
        loss_mean = torch.mean(loss)

        tensorboard_logs = {"train_loss", loss_mean}
        self.logger.experiment.add_scalar(
            "Loss/Train", loss_mean, self.current_epoch)
        return {"loss": loss_mean, "log": tensorboard_logs}

    def val_dataloader(self):
        validation_data = VisuomotorDataset(
            data_dir=self.data_dir, csv_file=self.training_path, 
            transform=self.resnet_transform, js_modifications=self.js_modifications)
        validation_dataloader = DataLoader(
            validation_data, num_workers=1, batch_size=32)
        return validation_dataloader

    def validation_step(self, val_batch, batch_idx):
        image, joint_state, key_pressed = val_batch["image"], val_batch["joint_states"], val_batch["key_pressed"]
        is_transition = val_batch["is_transition"]
        predicted_action = self(image, joint_state)
        actual_action = key_pressed.to(torch.int64).squeeze()
        loss = self.loss_fn(predicted_action, actual_action)
        loss[is_transition] = loss[is_transition] * self.transition_scalar
        loss = torch.mean(loss)
        acc = self.accuracy(predicted_action, actual_action)
        return {"val_loss": loss, "val_acc": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        self.log("val_acc", avg_acc)

        tensorboard_logs = {"avg_val_loss": avg_loss, "avg_val_acc": avg_acc}
        self.logger.experiment.add_scalar(
            f"Validation/Loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            f"Validation/Acc", avg_acc, self.current_epoch)
        return {"val_loss": avg_loss, "val_acc": avg_acc, "log": tensorboard_logs}