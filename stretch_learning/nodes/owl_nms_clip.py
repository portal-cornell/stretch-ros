from PIL import Image
import torch
import open_clip

# from util.value_constants import PROMPT_MAPPING

from nn_utils import get_owlvit_model_processor, crop_boxes, get_top_k_boxes
from torchvision.ops import batched_nms, box_convert


# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# def get_owl_vit_processor():
model, processor = get_owlvit_model_processor()
model = model.to(device)
model.eval()

model_clip, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")


@torch.no_grad()
def owl_vit(image, text_query):
    # image = Image.open(image_path).convert("RGB")
    inputs = processor(text=text_query, images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = torch.max(outputs["logits"], dim=-1)
    scores = torch.sigmoid(logits.values)
    labels = logits.indices
    boxes, scores, labels = get_top_k_boxes(outputs["pred_boxes"], scores, labels, 10)
    return boxes, scores, labels


@torch.no_grad()
def nms_clip(image, boxes, scores, labels, text_query, iou_threshold=0.05):
    image_width = image.size[0]
    image_height = image.size[1]
    boxes = torch.tensor(boxes)
    boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

    scores = torch.tensor(scores)
    good_boxes_nms_idx = batched_nms(
        boxes,
        scores,
        torch.tensor([1] * len(boxes)),
        iou_threshold=iou_threshold,
    )
    new_boxes = boxes[good_boxes_nms_idx]
    new_boxes = box_convert(new_boxes, in_fmt="xyxy", out_fmt="cxcywh")
    crops = crop_boxes(image, new_boxes)
    tokenized_text = tokenizer(text_query)
    with torch.no_grad(), torch.cuda.amp.autocast():
        preprocessed_images = torch.stack([preprocess(crop) for crop in crops])
        image_features = model_clip.encode_image(preprocessed_images)
        text_features = model_clip.encode_text(tokenized_text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_scores = (20.0 * image_features @ text_features.T).softmax(dim=0)
    # for i, crop in enumerate(crops):
    #     crop.show()
    #     print(similarity_scores[i])
    highest_score_idx = torch.argmax(similarity_scores)
    boxes = [new_boxes[highest_score_idx]]
    scores = [similarity_scores[highest_score_idx]]
    labels = [labels[highest_score_idx]]
    xyxy_boxes = box_convert(boxes[0].unsqueeze(0), in_fmt="cxcywh", out_fmt="xyxy")
    xyxy_boxes[0][0] = xyxy_boxes[0][0] * image_width
    xyxy_boxes[0][1] = xyxy_boxes[0][1] * image_height
    xyxy_boxes[0][2] = xyxy_boxes[0][2] * image_width
    xyxy_boxes[0][3] = xyxy_boxes[0][3] * image_height
    return boxes, xyxy_boxes, scores, labels
