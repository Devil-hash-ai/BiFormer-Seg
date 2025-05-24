import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.optim as optim

from model.encoder import ImageEncoderWithCrossAttentionFusion
from model.text_encoder import TextPromptEncoder
from model.decoder import MaskDecoder
from loss.losses import DiceLoss
from data.load_qata_dataloader import QaTaCovid19DataLoader


def dice_score(pred, target, eps=1e-8):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def train():
    image_root = r"your/image/root/directory"
    mask_root = r"your/mask/root/directory"
    csv_path = r"your/annotation/csv/path.csv"
    save_path = r"your/output/checkpoint/folder"
    resume_model_path = os.path.join(save_path, "model_epoch25.pt")
    os.makedirs(save_path, exist_ok=True)
    resume = True  

    loader = QaTaCovid19DataLoader(image_root, mask_root, csv_path, batch_size=4)
    dataloader = loader.get_loader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_encoder = ImageEncoderWithCrossAttentionFusion().to(device)
    text_encoder = TextPromptEncoder().to(device)
    mask_decoder = MaskDecoder().to(device)

    optimizer = optim.AdamW(
        list(image_encoder.parameters()) +
        list(text_encoder.parameters()) +
        list(mask_decoder.parameters()),
        lr=1e-4,
        weight_decay=1e-2
    )

    bce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-general")

    start_epoch = 0
    if resume and os.path.exists(resume_model_path):
        checkpoint = torch.load(resume_model_path, map_location=device)
        image_encoder.load_state_dict(checkpoint['image_encoder'])
        text_encoder.load_state_dict(checkpoint['text_encoder'])
        mask_decoder.load_state_dict(checkpoint['mask_decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        print(f"✅ Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, start_epoch + 20):
        image_encoder.train()
        text_encoder.train()
        mask_decoder.train()

        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
        epoch_loss = 0.0
        dice_total = 0.0

        for step, (images, masks, texts) in enumerate(loop):
            images = images.to(device)
            masks = masks.to(device)

            tokenized = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            text_feats = text_encoder(tokenized['input_ids'], tokenized['attention_mask'])

            img_feats = image_encoder(images)
            pred_masks, _ = mask_decoder(img_feats, text_feats)

            if pred_masks.shape != masks.shape:
                masks = F.interpolate(masks, size=pred_masks.shape[-2:], mode="nearest")

            loss = 0.5 * bce_loss(pred_masks, masks) + 0.5 * dice_loss(pred_masks, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            dice_total += dice_score(pred_masks, masks)
            loop.set_postfix(loss=loss.item(), dice=dice_total / (step + 1))

        avg_loss = epoch_loss / len(loop)
        avg_dice = dice_total / len(loop)
        print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, Avg Dice = {avg_dice:.4f}")
        if (epoch + 1) % 5 == 0 or epoch == 0:
            torch.save({
                'image_encoder': image_encoder.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'mask_decoder': mask_decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1
            }, os.path.join(save_path, f"model_epoch{epoch + 1}.pt"))

            save_image(torch.sigmoid(pred_masks[:1]), os.path.join(save_path, f"pred_epoch{epoch + 1}.png"))
            save_image(masks[:1], os.path.join(save_path, f"gt_epoch{epoch + 1}.png"))


if __name__ == '__main__':
    train()
