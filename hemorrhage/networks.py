import os
import random

import numpy as np
import torch
import torch.nn.functional as functional
from monai.networks.nets import resnet34
from monai.networks.nets.swin_unetr import SwinTransformer
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .imaging import bounding_box


def set_seed(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


class SwinClassifier(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.encoder = SwinTransformer(in_chans=1, embed_dim=settings["swin_embed_dim"], window_size=tuple(settings["swin_window"]), patch_size=tuple(settings["swin_patch"]), depths=tuple(settings["swin_depths"]), num_heads=tuple(settings["swin_heads"]), mlp_ratio=settings["swin_mlp_ratio"], qkv_bias=True, drop_rate=settings["dropout"], attn_drop_rate=settings["dropout"], drop_path_rate=settings["drop_path"], patch_norm=False, use_checkpoint=settings["use_checkpoint"], spatial_dims=3, downsample="merging", use_v2=False)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(settings["swin_embed_dim"] * 16, 1)

    def forward(self, inputs):
        features = self.encoder(inputs, normalize=True)[-1]
        return self.classifier(self.pool(features).flatten(1))


def build_network(name, settings):
    if name == "resnet34":
        return resnet34(pretrained=False, spatial_dims=3, n_input_channels=1, num_classes=1, conv1_t_size=7, conv1_t_stride=2, shortcut_type="B", bias_downsample=False, widen_factor=settings["resnet_width_factor"])
    if name == "swin":
        return SwinClassifier(settings)
    raise ValueError(name)


class ROIDataset(Dataset):
    def __init__(self, patients, store, config, training=False):
        self.store, self.config = store, config
        copies = config["augmentation"]["copies"] if training else 0
        self.entries = [(patient, variant) for patient in patients for variant in range(copies + 1)]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        patient, variant = self.entries[index]
        volume = self.store.variant(patient, index=variant)
        bounds = bounding_box(volume.mask, self.config["deep"]["roi_margin"])
        image = np.array(volume.image[bounds], dtype=np.float32, copy=True)
        if self.config["deep"]["mask_outside_roi"]:
            image[~volume.mask[bounds]] = 0
        tensor = torch.from_numpy(image)[None, None]
        tensor = functional.interpolate(tensor, size=self.config["deep"]["input_shape"], mode="trilinear", align_corners=False)[0]
        label = float(patient.label) if patient.label is not None else 0.0
        return tensor, torch.tensor(label, dtype=torch.float32)


def complete_batches(length, batch_size, generator):
    if length < 2:
        raise ValueError("Network training needs at least two samples")
    order = torch.randperm(length, generator=generator).tolist()
    batches = [order[index:index + batch_size] for index in range(0, length, batch_size)]
    if len(batches[-1]) == 1 and len(batches) > 1:
        batches[-2].extend(batches.pop())
    return batches


class DeepClassifier:
    def __init__(self, name, store, config, seed):
        self.name, self.store, self.config, self.seed = name, store, config, seed

    def fit(self, patients):
        settings = self.config["deep"]
        set_seed(self.seed, settings["deterministic"])
        torch.set_num_threads(self.config["execution"]["torch_threads"])
        device = choose_device(settings["device"])
        self.network = build_network(self.name, settings).to(device)
        self.training_ids = sorted(p.patient_id for p in patients)
        if settings["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(self.network.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        else:
            optimizer = torch.optim.SGD(self.network.parameters(), lr=settings["learning_rate"], momentum=settings["momentum"], weight_decay=settings["weight_decay"])
        loss_function = nn.BCEWithLogitsLoss()
        dataset = ROIDataset(patients, self.store, self.config, training=True)
        generator = torch.Generator().manual_seed(self.seed)
        self.loss_history = []
        for epoch in range(settings["epochs"]):
            self.network.train()
            total, count = 0.0, 0
            loader = DataLoader(dataset, batch_sampler=complete_batches(len(dataset), settings["batch_size"], generator), num_workers=0)
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.network(images).reshape(-1)
                loss = loss_function(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {self.name} loss in epoch {epoch + 1}")
                loss.backward()
                if settings["gradient_clip"] is not None:
                    nn.utils.clip_grad_norm_(self.network.parameters(), settings["gradient_clip"])
                optimizer.step()
                total += float(loss.detach().cpu()) * len(labels)
                count += len(labels)
            self.loss_history.append({"epoch": epoch + 1, "training_loss": total / count})
        self.network = self.network.cpu().eval()
        self.fitted_device = str(device)
        return self

    def predict(self, patients):
        device = choose_device(self.config["deep"]["device"])
        self.network.to(device).eval()
        dataset = ROIDataset(patients, self.store, self.config, training=False)
        loader = DataLoader(dataset, batch_size=self.config["deep"]["batch_size"], shuffle=False, num_workers=0)
        result = []
        with torch.inference_mode():
            for images, _ in loader:
                probabilities = torch.sigmoid(self.network(images.to(device)).reshape(-1))
                result.extend(probabilities.cpu().numpy().tolist())
        self.network.cpu()
        return np.asarray(result)

    def metadata(self):
        return {"model": self.name, "training_patients": self.training_ids, "seed": self.seed, "device": self.fitted_device, "loss_history": self.loss_history, "parameters": sum(p.numel() for p in self.network.parameters()), "architecture": self.config["deep"]}
