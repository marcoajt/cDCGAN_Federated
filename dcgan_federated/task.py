# gan_federated/task.py
"""gan-federated: toda a lógica do DCGAN condicional + utilitários Flower + particionamento Dirichlet."""

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Flower Datasets
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

# Hyperparâmetros de dados
BATCH_SIZE = 128
DEFAULT_ALPHA = 0.4  # controla o grau de heterogeneidade

# Transformação padrão
_default_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


class TorchHFDataset(Dataset):
    """Wrapper que converte um HuggingFace Dataset em torch.Dataset."""
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
    Carrega o DataLoader particionado via DirichletPartitioner.

    Args:
        client_id: índice da partição (0 .. num_clients-1)
        num_clients: total de clientes
        alpha: parâmetro Dirichlet (quanto menor, mais não-iid)
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


# ------------------------------------------------------------------
# Peso init e DCGAN condicional
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
            nn.utils.spectral_norm(
                nn.Conv2d(nc + n_classes, ndf, 4, 2, 1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
            ),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 0, bias=False)
            ),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 4 * 5 * 5, 1)),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        B = img.size(0)
        lbl = self.label_emb(labels).view(B, -1, 1, 1).repeat(1, 1, 28, 28)
        x = torch.cat([img, lbl], dim=1)
        return self.main(x)


def train_cgan(generator, discriminator, train_loader,
               optimizer_G, optimizer_D, adversarial_loss,
               epochs, device):
    generator.train()
    discriminator.train()
    total_G, total_D = 0.0, 0.0
    history = {"loss_G": [], "loss_D": []}
    for _ in range(epochs):
        for imgs, labels in train_loader:
            bs = imgs.size(0)
            valid = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)
            real = imgs.to(device)
            lbls = labels.to(device)

            optimizer_D.zero_grad()
            loss_real = adversarial_loss(discriminator(real, lbls), valid)
            z = torch.randn(bs, generator.nz, 1, 1, device=device)
            gen_imgs = generator(z, lbls)
            loss_fake = adversarial_loss(discriminator(gen_imgs.detach(), lbls), fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            output = discriminator(gen_imgs, lbls)
            loss_G = adversarial_loss(output, valid)
            loss_G.backward()
            optimizer_G.step()

            total_G += loss_G.item()
            total_D += loss_D.item()
            history["loss_G"].append(loss_G.item())
            history["loss_D"].append(loss_D.item())

    n_batches = len(train_loader) * epochs
    return total_G / n_batches, total_D / n_batches, history


def evaluate_cgan(generator, discriminator, train_loader, adversarial_loss, device):
    generator.eval()
    discriminator.eval()
    total, count = 0.0, 0
    valid = torch.ones(BATCH_SIZE, 1, device=device)
    fake = torch.zeros(BATCH_SIZE, 1, device=device)
    with torch.no_grad():
        for imgs, labels in train_loader:
            bs = imgs.size(0)
            real = imgs.to(device)
            lbls = labels.to(device)
            loss_real = adversarial_loss(discriminator(real, lbls), valid[:bs])
            z = torch.randn(bs, generator.nz, 1, 1, device=device)
            gen_imgs = generator(z, lbls)
            loss_fake = adversarial_loss(discriminator(gen_imgs, lbls), fake[:bs])
            total += (loss_real + loss_fake).item()
            count += 1
    return {"loss": total / count if count > 0 else float("inf")}


def get_parameters(net: nn.Module):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: nn.Module, parameters):
    params = zip(net.state_dict().keys(), parameters)
    sd = OrderedDict({k: torch.tensor(v) for k, v in params})
    net.load_state_dict(sd, strict=True)

# Alias para compatibilidade com o server_app.py antigo
get_weights = get_parameters

from flwr.common import Metrics

def weighted_average_loss(metrics: list[tuple[int, Metrics]]):
    losses = [n * m["loss"] for n, m in metrics]
    total = sum(n for n, _ in metrics)
    return {"loss": sum(losses) / total}
