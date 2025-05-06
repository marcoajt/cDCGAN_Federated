# gan_federated/task.py
"""gan-federated: DCGAN condicional + Dirichlet + data augmentation."""

import os
from collections import OrderedDict, Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
from torchvision import transforms
from torchvision.utils import save_image

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr.common import Metrics

# ————— Parâmetros de dados —————
BATCH_SIZE    = 128
DEFAULT_ALPHA = 0.4
LATENT_DIM    = 100
NGF           = 64
NC            = 1
N_CLASSES     = 10

# Transformação de imagem
_default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

class TorchHFDataset(Dataset):
    """Wrap HuggingFace Dataset → torch.Dataset."""
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        img = _default_transform(sample["image"])
        label = sample["label"]
        return img, label

def load_dataloader(client_id: int, num_clients: int, alpha: float = DEFAULT_ALPHA):
    """
    Carrega DataLoader particionado via Dirichlet (SEM augmentação).
    """
    dirichlet_partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        alpha=alpha,
        partition_by="label",
    )
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": dirichlet_partitioner},
    )
    hf_train = fds.load_partition(partition_id=client_id, split="train")
    torch_ds = TorchHFDataset(hf_train)
    return DataLoader(torch_ds, batch_size=BATCH_SIZE, shuffle=True)

def augment_local_dataset(
    real_dataset: Dataset,
    generator_path: str = "generator_global.pt",
    latent_dim: int = LATENT_DIM,
    ngf: int = NGF,
    nc: int = NC,
    n_classes: int = N_CLASSES,
    batch_size: int = BATCH_SIZE,
    save_png: bool = False,
    out_dir: str = None,
):
    """
    Complementa `real_dataset` com sintéticos do gerador global,
    nivelando à média de amostras por classe. Retorna um DataLoader.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) Contagem de cada label
    labels = [lbl for _, lbl in real_dataset]
    counts = Counter(labels)
    avg = int(np.mean([counts.get(c, 0) for c in range(n_classes)]))

    # 2) Carrega gerador global
    if not os.path.exists(generator_path):
        raise FileNotFoundError(f"'{generator_path}' não encontrado.")
    gen = Generator(latent_dim, ngf, nc, n_classes).to(device)
    gen.load_state_dict(torch.load(generator_path, map_location=device))
    gen.eval()

    # 3) Gera sintéticos faltantes
    synthetic_chunks = []
    for c in range(n_classes):
        deficit = max(0, avg - counts.get(c, 0))
        if deficit <= 0:
            continue
        z    = torch.randn(deficit, latent_dim, 1, 1, device=device)
        lbls = torch.full((deficit,), c, dtype=torch.long, device=device)
        with torch.no_grad():
            imgs = gen(z, lbls).cpu()
        imgs = (imgs + 1) / 2  # [-1,1] → [0,1]

        if save_png:
            base = out_dir or f"synthetic"
            os.makedirs(base, exist_ok=True)
            for i, img in enumerate(imgs):
                save_image(img, os.path.join(base, f"class{c}_{i}.png"))

        synthetic_chunks.append(TensorDataset(imgs, lbls.cpu()))

    # 4) Concatena real + sintético
    if synthetic_chunks:
        synth_ds = ConcatDataset(synthetic_chunks)
        final_ds = ConcatDataset([real_dataset, synth_ds])
    else:
        final_ds = real_dataset

    # 5) Retorna DataLoader balanceado
    return DataLoader(final_ds, batch_size=batch_size, shuffle=True)

# ------------------------------------------------------------------
# Peso init e DCGAN condicional (iguais ao antes)
# ------------------------------------------------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, n_classes):
        super().__init__()
        self.nz = nz
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + n_classes, ngf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        B = noise.size(0)
        lbl = self.label_emb(labels).view(B, -1, 1, 1)
        x = torch.cat([noise, lbl], dim=1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, ndf, nc, n_classes):
        super().__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.main = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(nc + n_classes, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, 3, 1, 0, bias=False)),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*4*5*5, 1)),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        B = img.size(0)
        lbl = self.label_emb(labels).view(B, -1, 1, 1).repeat(1,1,28,28)
        x   = torch.cat([img, lbl], dim=1)
        return self.main(x)

def train_cgan(generator, discriminator, train_loader,
               optimizer_G, optimizer_D, adversarial_loss,
               epochs, device):
    generator.train(); discriminator.train()
    total_G, total_D = 0.0, 0.0
    history = {"loss_G": [], "loss_D": []}
    for _ in range(epochs):
        for imgs, labels in train_loader:
            bs    = imgs.size(0)
            valid = torch.ones(bs, 1, device=device)
            fake  = torch.zeros(bs, 1, device=device)
            real  = imgs.to(device); lbls = labels.to(device)

            optimizer_D.zero_grad()
            loss_real = adversarial_loss(discriminator(real, lbls), valid)
            z         = torch.randn(bs, generator.nz,1,1,device=device)
            gen_imgs  = generator(z, lbls)
            loss_fake = adversarial_loss(discriminator(gen_imgs.detach(),lbls), fake)
            loss_D    = (loss_real + loss_fake)/2
            loss_D.backward(); optimizer_D.step()

            optimizer_G.zero_grad()
            output = discriminator(gen_imgs, lbls)
            loss_G  = adversarial_loss(output, valid)
            loss_G.backward(); optimizer_G.step()

            total_G += loss_G.item(); total_D += loss_D.item()
            history["loss_G"].append(loss_G.item())
            history["loss_D"].append(loss_D.item())

    n_batches = len(train_loader)*epochs
    return total_G/n_batches, total_D/n_batches, history

def evaluate_cgan(generator, discriminator, train_loader, adversarial_loss, device):
    generator.eval(); discriminator.eval()
    total, count = 0.0,0
    valid = torch.ones(BATCH_SIZE,1,device=device)
    fake  = torch.zeros(BATCH_SIZE,1,device=device)
    with torch.no_grad():
        for imgs, labels in train_loader:
            bs        = imgs.size(0)
            real      = imgs.to(device); lbls=labels.to(device)
            loss_real = adversarial_loss(discriminator(real,lbls), valid[:bs])
            z         = torch.randn(bs, generator.nz,1,1,device=device)
            gen_imgs  = generator(z,lbls)
            loss_fake = adversarial_loss(discriminator(gen_imgs,lbls), fake[:bs])
            total    += (loss_real+loss_fake).item(); count+=1
    return {"loss": total/count if count>0 else float("inf")}

def get_parameters(net: nn.Module):
    return [ val.cpu().numpy() for _, val in net.state_dict().items() ]

def set_parameters(net: nn.Module, parameters):
    params = zip(net.state_dict().keys(), parameters)
    sd     = OrderedDict({k: torch.tensor(v) for k, v in params})
    net.load_state_dict(sd, strict=True)

# Alias para compatibilidade com server_app.py antigo
get_weights = get_parameters

def weighted_average_loss(metrics: list[tuple[int, Metrics]]):
    losses = [n * m["loss"] for n, m in metrics]
    total  = sum(n for n, _ in metrics)
    return {"loss": sum(losses)/total}
