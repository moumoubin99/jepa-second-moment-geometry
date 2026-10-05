"""CIFAR-100 loader straight from the pickled python archive (no torchvision)."""
import os
import pickle
import numpy as np
import torch

CIFAR_MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


def _load_split(root, split):
    path = os.path.join(root, "cifar-100-python", split)
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    labels = np.array(d[b"fine_labels"], dtype=np.int64)
    return torch.from_numpy(data), torch.from_numpy(labels)


class CIFAR100Memory:
    """Whole dataset resident in GPU memory; fast augmentation on device."""

    def __init__(self, root, split="train", device="cuda", subset=None):
        x, y = _load_split(root, split)
        x = (x - CIFAR_MEAN) / CIFAR_STD
        if subset is not None:
            x, y = x[:subset], y[:subset]
        self.x = x.to(device)
        self.y = y.to(device)
        self.device = device
        self.n = self.x.shape[0]

    def sample_batch(self, bs, augment=True, generator=None):
        idx = torch.randint(0, self.n, (bs,), device=self.device, generator=generator)
        imgs = self.x[idx]
        if augment:
            imgs = self._augment(imgs, generator)
        return imgs, self.y[idx]

    def _augment(self, imgs, generator=None):
        # random horizontal flip
        flip = torch.rand(imgs.shape[0], device=imgs.device, generator=generator) < 0.5
        imgs = torch.where(flip.view(-1, 1, 1, 1), imgs.flip(-1), imgs)
        # random crop with reflection pad 4
        pad = torch.nn.functional.pad(imgs, (4, 4, 4, 4), mode="reflect")
        b = imgs.shape[0]
        ox = torch.randint(0, 9, (b,), device=imgs.device, generator=generator)
        oy = torch.randint(0, 9, (b,), device=imgs.device, generator=generator)
        # vectorized crop gather (R3: the per-sample loop forced one GPU sync per image, which is
        # very slow on a shared/contended GPU); same RNG draws and identical output to the loop
        ar = torch.arange(32, device=imgs.device)
        return pad[torch.arange(b, device=imgs.device).view(b, 1, 1, 1), torch.arange(3, device=imgs.device).view(1, 3, 1, 1),
                   (oy.view(b, 1, 1, 1) + ar.view(1, 1, 32, 1)), (ox.view(b, 1, 1, 1) + ar.view(1, 1, 1, 32))]

    def eval_batch(self, n):
        n = min(n, self.n)
        return self.x[:n], self.y[:n]


STL_MEAN = torch.tensor([0.4467, 0.4398, 0.4066]).view(1, 3, 1, 1)
STL_STD = torch.tensor([0.2603, 0.2566, 0.2713]).view(1, 3, 1, 1)


def _read_stl_bin(path):
    arr = np.fromfile(path, dtype=np.uint8)
    arr = arr.reshape(-1, 3, 96, 96)
    arr = np.transpose(arr, (0, 1, 3, 2))  # STL stores column-major -> (N,C,H,W)
    return np.ascontiguousarray(arr)


class STL10Memory:
    """STL-10 96px. Unlabeled split (100k) for SSL pretraining (uint8 on GPU, normalized
    per-batch); labeled train/test (float) for kNN / linear probe."""

    def __init__(self, root, split="unlabeled", device="cuda", subset=None):
        base = os.path.join(root, "stl10_binary")
        self.device = device
        self.mean = STL_MEAN.to(device)
        self.std = STL_STD.to(device)
        if split == "unlabeled":
            x = _read_stl_bin(os.path.join(base, "unlabeled_X.bin"))
            if subset is not None:
                x = x[:subset]
            self.xu = torch.from_numpy(x).to(device)  # uint8 (N,3,96,96)
            self.y = None
            self.n = self.xu.shape[0]
        else:
            x = _read_stl_bin(os.path.join(base, f"{split}_X.bin"))
            y = np.fromfile(os.path.join(base, f"{split}_y.bin"), dtype=np.uint8).astype(np.int64) - 1
            xf = torch.from_numpy(x).float().to(device) / 255.0
            self.x = (xf - self.mean) / self.std
            self.y = torch.from_numpy(y).to(device)
            self.n = self.x.shape[0]

    def _norm(self, u8):
        return (u8.float() / 255.0 - self.mean) / self.std

    def sample_batch(self, bs, augment=True, generator=None):
        idx = torch.randint(0, self.n, (bs,), device=self.device)
        imgs = self._norm(self.xu[idx])
        if augment:
            flip = torch.rand(bs, device=self.device) < 0.5
            imgs = torch.where(flip.view(-1, 1, 1, 1), imgs.flip(-1), imgs)
            pad = torch.nn.functional.pad(imgs, (12, 12, 12, 12), mode="reflect")
            ox = torch.randint(0, 25, (bs,), device=self.device)
            oy = torch.randint(0, 25, (bs,), device=self.device)
            ar = torch.arange(96, device=self.device)          # vectorized crop (same values as the per-sample loop)
            rows = (oy[:, None] + ar)[:, None, :, None]; cols = (ox[:, None] + ar)[:, None, None, :]
            imgs = pad[torch.arange(bs, device=self.device)[:, None, None, None],
                       torch.arange(3, device=self.device)[None, :, None, None], rows, cols]
        return imgs, (self.y[idx] if self.y is not None else None)

    def eval_batch(self, n):
        n = min(n, self.n)
        return self.x[:n], self.y[:n]
