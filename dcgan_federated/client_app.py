# client_app.py
"""gan-federated: A Flower / PyTorch cDCGAN client app."""

import torch
import json
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from dcgan_federated.task import (
    Generator,
    Discriminator,
    weights_init,
    load_dataloader,
    train_cgan,
    evaluate_cgan,
    get_parameters,
    set_parameters,
)

class FlowerClient(NumPyClient):
    def __init__(self, client_id: int, num_partitions: int):
        self.client_id = client_id
        self.num_partitions = num_partitions
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # ——— Hiperparâmetros do modelo (igual ao antes) ———
        self.nz = 100
        self.ngf = 64
        self.ndf = 64
        self.nc = 1
        self.n_classes = 10

        # ——— 1 época local por rodada (permaneceu 1) ———
        self.local_epochs = 1

        # ——— Carrega DataLoader SEM augmentação ———
        self.train_loader = load_dataloader(client_id, num_partitions)

        # ——— Inicializa redes e pesos (igual ao antes) ———
        self.generator = Generator(self.nz, self.ngf, self.nc, self.n_classes).to(self.device)
        self.discriminator = Discriminator(self.ndf, self.nc, self.n_classes).to(self.device)
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # ——— Otimizadores e loss (igual ao antes) ———
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.adversarial_loss = torch.nn.BCELoss()

    def get_parameters(self, config):
        return get_parameters(self.generator) + get_parameters(self.discriminator)

    def set_parameters(self, parameters, config):
        gen_len = len(self.generator.state_dict())
        set_parameters(self.generator, parameters[:gen_len])
        set_parameters(self.discriminator, parameters[gen_len:])

    def fit(self, parameters, config):
        # RECEBE parâmetros globais e treina 1 época local
        self.set_parameters(parameters, config)
        loss_G, loss_D, history = train_cgan(
            generator=self.generator,
            discriminator=self.discriminator,
            train_loader=self.train_loader,
            optimizer_G=self.optimizer_G,
            optimizer_D=self.optimizer_D,
            adversarial_loss=self.adversarial_loss,
            epochs=self.local_epochs,
            device=self.device,
        )
        # Cliente 0 salva gerador local e histórico (igual ao antes)
        if self.client_id == 0:
            torch.save(self.generator.state_dict(), "generator_client0.pt")
            with open("loss_history.json", "w") as f:
                json.dump(history, f)
        return (
            self.get_parameters(config),
            len(self.train_loader.dataset),
            {"loss_G": loss_G, "loss_D": loss_D},
        )

    def evaluate(self, parameters, config):
        # Avaliação igual ao antes
        self.set_parameters(parameters, config)
        metrics = evaluate_cgan(
            generator=self.generator,
            discriminator=self.discriminator,
            train_loader=self.train_loader,
            adversarial_loss=self.adversarial_loss,
            device=self.device,
        )
        return float(metrics["loss"]), len(self.train_loader.dataset), {"loss": metrics["loss"]}

def client_fn(context: Context):
    pid = int(context.node_config.get("partition-id", context.node_id))
    nparts = int(context.node_config.get("num-partitions", pid + 1))
    return FlowerClient(pid, nparts).to_client()

# ——— Permaneceu igual: declaração do app do cliente ———
app = ClientApp(client_fn)
