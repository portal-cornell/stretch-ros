import warnings
import torch
import torchvision.models as models
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import nn
from torch import optim
from r3m import load_r3m
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from stretch_camera_dataset import StretchCameraDataset
from torch.utils.data import DataLoader


class BC(pl.LightningModule):

    def __init__(self, joint_state_dims, state_action_dims, training_path, validation_path, use_r3m):
        super().__init__()
        self.save_hyperparameters()
        if use_r3m:
            self.conv_net = load_r3m("resnet18")
        else:
            resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            resnet18.fc = nn.Linear(resnet18.fc.in_features, 512)
            self.conv_net = resnet18
        self.conv_net_name = "R3M" if use_r3m else "ResNet18"
        # joint_state_dims = int(joint_state_dims * 2/3)
        self.fc1 = nn.Linear(512 + joint_state_dims, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, state_action_dims)

        self.training_path = training_path
        self.validation_path = validation_path
        self.resnet_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.augment = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomInvert(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomErasing(p=0.25)])

    def forward(self, image, data):
        x1 = self.conv_net(image)
        x2 = data
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch["image"] = self.augment(batch["image"])
        return batch

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        return [optimizer], [scheduler] 

    def train_dataloader(self):
        training_data = StretchCameraDataset(csv_file=self.training_path, transform=self.resnet_transform)
        train_dataloader = DataLoader(training_data, num_workers=16, batch_size=32)
        return train_dataloader

    def val_dataloader(self):
        validation_data = StretchCameraDataset(csv_file=self.validation_path, transform=self.resnet_transform)
        validation_dataloader = DataLoader(validation_data, num_workers=16, batch_size=32)
        return validation_dataloader

    def training_step(self, train_batch, batch_idx):
        image, joint_state, key_pressed = train_batch["image"], train_batch["joint_states"], train_batch["key_pressed"]
        predicted_action_vector = self(image, joint_state)
        loss = F.cross_entropy(predicted_action_vector, key_pressed.to(torch.int64).squeeze())
        tensorboard_logs = {"train_loss", loss}
        self.logger.experiment.add_scalar(f"{self.conv_net_name}: Train/Loss", loss, self.current_epoch)
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        image, joint_state, key_pressed = val_batch["image"], val_batch["joint_states"], val_batch["key_pressed"] 
        predicted_action_vector = self(image, joint_state)
        loss = F.cross_entropy(predicted_action_vector, key_pressed.to(torch.int64).squeeze())
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_val_loss", avg_loss}
        self.logger.experiment.add_scalar(f"{self.conv_net_name}: Validation/Loss", avg_loss, self.current_epoch)
        return {"val_loss": avg_loss, "log": tensorboard_logs}


