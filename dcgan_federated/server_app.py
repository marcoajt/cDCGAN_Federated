# server_app.py
"""gan-federated: A Flower / PyTorch cDCGAN server app com agregação de losses."""

from typing import List, Tuple
import torch
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from dcgan_federated.task import Generator, Discriminator, get_weights, set_parameters

def weighted_accuracy(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [n * m.get("accuracy", 0.0) for n, m in metrics]
    total = sum(n for n, _ in metrics)
    return {"accuracy": sum(accuracies) / total}

def aggregate_fit_losses(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(n for n, _ in metrics)
    avg_G = sum(n * m.get("loss_G", 0.0) for n, m in metrics) / total
    avg_D = sum(n * m.get("loss_D", 0.0) for n, m in metrics) / total
    return {"loss_G": avg_G, "loss_D": avg_D}

def server_fn(context: Context) -> ServerAppComponents:
    # ——— Modificação: fixar 50 rounds de comunicação ———
    rounds = 50

    frac_fit = context.run_config.get("fraction-fit", 0.5)

    # Mesmos hiperparâmetros do Generator e Discriminator
    nz, ngf, ndf, nc, n_classes = 100, 64, 64, 1, 10
    gen = Generator(nz, ngf, nc, n_classes)
    disc = Discriminator(ndf, nc, n_classes)

    # Parâmetros iniciais aleatórios (serão sobrescritos no primeiro fit)
    ndarrays = get_weights(gen) + get_weights(disc)
    params = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=frac_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_accuracy,
        fit_metrics_aggregation_fn=aggregate_fit_losses,
    )
    config = ServerConfig(num_rounds=rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# ——— Modificação: declara app no nível do módulo ———
app = ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    app.run()

    # ——— Modificação: após os 50 rounds, salvar o gerador global ———
    nz, ngf, nc, n_classes = 100, 64, 1, 10
    gen = Generator(nz, ngf, nc, n_classes)
    params = app.strategy.parameters           # parâmetros finais
    gen_len = len(gen.state_dict())
    set_parameters(gen, params.tensors[:gen_len])
    torch.save(gen.state_dict(), "generator_global.pt")
    print("✔ Gerador global salvo em generator_global.pt")
