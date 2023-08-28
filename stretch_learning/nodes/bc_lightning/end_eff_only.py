import torch
import torchmetrics
import torchvision.models as models
import torch.nn.functional as F
import pytorch_lightning as pl
from pathlib import Path
from torch import nn
from torch import optim
from r3m import load_r3m
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from stretch_camera_dataset import StretchCameraDataset
from torch.utils.data import DataLoader


class BC_End_Eff(pl.LightningModule):

    def __init__(self, joint_state_dims, state_action_dims, training_path="",
                 validation_path="", test_paths=[], test_names=[], use_r3m=True):
        super().__init__()
        self.save_hyperparameters()
        if use_r3m:
            self.conv_net = load_r3m("resnet18")
        else:
            resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            resnet18.fc = nn.Linear(resnet18.fc.in_features, 512)
            self.conv_net = resnet18
        self.conv_net_name = "R3M" if use_r3m else "ResNet18"
        self.fc1 = nn.Linear(joint_state_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_action_dims)

        self.transition_scalar = 50
        self.training_path = training_path
        self.validation_path = validation_path
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        # self.accuracy = torchmetrics.Accuracy(
        #     task='multiclass', num_classes=state_action_dims)

        self.num_workers = 1 if str(Path.home()) == "/home/jlr429" else 16
        self.test_paths = test_paths
        self.idx_to_names = {idx: name for idx, name in enumerate(test_names)}

        self.resnet_transform = transforms.Compose([transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.augment = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomInvert(p=0.5),
                                           transforms.RandomVerticalFlip(
                                               p=0.5),
                                           transforms.RandomErasing(p=0.25)])

    def forward(self, _, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     batch["image"] = self.augment(batch["image"])
    #     return batch

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        training_data = StretchCameraDataset(
            csv_file=self.training_path, transform=self.resnet_transform)
        train_dataloader = DataLoader(
            training_data, num_workers=self.num_workers, batch_size=32)
        return train_dataloader

    def training_step(self, train_batch, batch_idx):
        image, joint_state, key_pressed = train_batch["image"], train_batch[
            "joint_states"], train_batch["key_pressed"]
        is_transition = train_batch["is_transition"]
        predicted_action = self(image, joint_state)
        actual_action = key_pressed.to(torch.int64).squeeze()
        loss = self.loss_fn(predicted_action, actual_action)
        loss[is_transition] = loss[is_transition] * self.transition_scalar
        loss = torch.mean(loss)

        tensorboard_logs = {"train_loss", loss}
        self.logger.experiment.add_scalar(
            "Loss/Train", loss, self.current_epoch)
        return {"loss": loss, "log": tensorboard_logs}

    def val_dataloader(self):
        validation_data = StretchCameraDataset(
            csv_file=self.validation_path, transform=self.resnet_transform)
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
            f"{self.conv_net_name}: Validation/Loss", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar(
            f"{self.conv_net_name}: Validation/Acc", avg_acc, self.current_epoch)
        return {"val_loss": avg_loss, "val_acc": avg_acc, "log": tensorboard_logs}

    def test_dataloader(self):
        test_dataloaders = []
        for test_path in self.test_paths:
            test_data = StretchCameraDataset(
                csv_file=test_path, transform=self.resnet_transform)
            test_dataloaders.append(DataLoader(
                test_data, num_workers=1, batch_size=32))
        return test_dataloaders

    def test_step(self, test_batch, batch_idx, test_idx=0):
        image, joint_state, key_pressed = test_batch["image"], test_batch[
            "joint_states"], test_batch["key_pressed"]
        predicted_action = self(image, joint_state)
        actual_action = key_pressed.to(torch.int64).squeeze()
        loss = self.loss_fn(predicted_action, actual_action)
        acc = self.accuracy(predicted_action, actual_action)
        return {"test_loss": loss, "test_acc": acc, "test_idx": test_idx}

    def test_epoch_end(self, outputs):
        for idx, output in enumerate(outputs):
            name = self.idx_to_names[idx]
            avg_loss = torch.stack([x["test_loss"] for x in output]).mean()
            self.log(f"test_{name}_dataloader_idx_{idx}_loss", avg_loss)
            avg_acc = torch.stack([x["test_acc"] for x in output]).mean()
            self.log(f"test_{name}_dataloader_idx_{idx}_accu", avg_acc)
