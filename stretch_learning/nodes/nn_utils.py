import random
import numpy as np
import torch
import open_clip
import torch.nn as nn
import albumentations as A

from typing import List, Type
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from albumentations.pytorch import ToTensorV2
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# from util.print_colors import PrintColors as PC

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


@torch.no_grad()
def get_universal_sentence_encoder(
    ckpt_name: str = "openai/clip-vit-base-patch32", device="cpu"
):
    model, _, tokenizer = get_clip_model_processor_tokenizer(device=device)

    def encode_fn(texts):
        text_tokens = tokenizer(texts).to(device)
        return model.encode_text(text_tokens)

    return encode_fn


def get_owlvit_model_processor(
    ckpt_name: str = "google/owlv2-base-patch16-ensemble", device="cuda:0"
):
    processor = Owlv2Processor.from_pretrained(ckpt_name)

    model = Owlv2ForObjectDetection.from_pretrained(ckpt_name)
    model = model.to(device)
    model.eval()
    return model, processor


def get_clip_model_processor_tokenizer(
    model_name: str = "ViT-B-32",
    ckpt_name: str = "openai/clip-vit-base-patch32",
    device="cuda:0",
):
    # print(PC.OKCYAN + "Loading CLIP model" + PC.ENDC)
    model, _, processor = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device)
    model.eval()
    return model, processor, tokenizer


def crop_boxes(image, boxes):
    crops = []
    for box in boxes:
        cx, cy, w, h = box
        # Convert normalized coordinates to pixel coordinates
        image_width, image_height = image.size
        left = int((cx - (w / 2)) * image_width)
        top = int((cy - (h / 2)) * image_height)
        right = int((cx + (w / 2)) * image_width)
        bottom = int((cy + (h / 2)) * image_height)
        # Crop the image using PIL
        crop = np.array(
            image.crop(
                (left, top, right, bottom),
            )
        )
        crop = crop[:, :, ::-1]
        crops.append(Image.fromarray(crop))
    return crops


def get_top_k_boxes(predicted_boxes, scores, labels, k):
    # Sort the scores tensor in descending order along the num_boxes dimension
    sorted_scores = torch.argsort(scores, dim=1, descending=True)

    # Take the top 4 indices from each batch
    top_k_indices = sorted_scores[:, :k]
    top_k_boxes = (
        torch.gather(
            predicted_boxes,
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, predicted_boxes.size(-1)),
        )
        .cpu()
        .detach()
        .numpy()
        .reshape(k, 4)
    )
    top_k_scores = scores[:, top_k_indices].cpu().detach().numpy().flatten()
    top_k_labels = labels[:, top_k_indices].cpu().detach().numpy().flatten()
    return top_k_boxes, top_k_scores, top_k_labels


def get_train_val_augmentations(resize_height, resize_width):
    train_transform = A.Compose(
        [
            A.GaussianBlur(),
            A.augmentations.dropout.coarse_dropout.CoarseDropout(max_holes=16),
            A.augmentations.transforms.PixelDropout(),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            A.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
            ToTensorV2(),
        ]
    )
    return train_transform, val_transform


def batch_to_device(batch, device):
    for key, value in batch.items():
        if torch.is_tensor(value):
            batch[key] = value.to(device)
    return batch


class Mlp(nn.Module):
    def __init__(self, sequential):
        super().__init__()
        self.sequential = sequential

    def forward(self, x, text_embed):
        return self.sequential(x)


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    dropout_prob: List[float] = None,
    activation_fn: Type[nn.Module] = nn.ReLU,
) -> List[nn.Module]:
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    if dropout_prob is not None:
        assert len(dropout_prob) == len(
            net_arch
        ), "Length of dropout_prob should match the number of hidden layers"

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())
        if dropout_prob is not None and dropout_prob[idx] > 0:
            modules.append(nn.Dropout(p=dropout_prob[idx]))

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    return Mlp(nn.Sequential(*modules))


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
