import argparse
import torch
import matplotlib.pyplot as plt
from dcgan_federated.task import Generator

def generate_synthetic(class_label: int, num_samples: int, model_path: str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Parâmetros do DCGAN condicional (devem bater com os usados no client_app)
    latent_dim = 100   # nz
    ngf = 64           # tamanho dos feature maps do Generator
    nc = 1             # canais da imagem (MNIST é 1)
    n_classes = 10

    # Instancia e carrega pesos
    gen = Generator(latent_dim, ngf, nc, n_classes).to(device)
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.eval()

    # Gera ruído e rótulos
    # Para o DCGAN condicional usamos shape (B, nz, 1, 1)
    noise = torch.randn(num_samples, latent_dim, 1, 1, device=device)
    labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)

    # Geração
    with torch.no_grad():
        imgs = gen(noise, labels).cpu()

    # Mapeia de [-1,1] para [0,1]
    return (imgs + 1) / 2  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera um grid 10×N de amostras sintéticas, uma linha por classe."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Número de amostras por classe (colunas).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="generator_client0.pt",
        help="Caminho do arquivo do gerador salvo.",
    )
    args = parser.parse_args()

    num_cols = args.num_samples
    model_path = args.model_path

    # Cria figura com subplots
    fig, axes = plt.subplots(
        nrows=10, ncols=num_cols,
        figsize=(num_cols, 10), dpi=100,
        gridspec_kw={"wspace": 0.05, "hspace": 0.05}
    )
    plt.subplots_adjust(left=0.2)  # Espaço extra à esquerda para labels

    # Para cada classe, gera e plota
    for class_label in range(10):
        imgs = generate_synthetic(class_label, num_cols, model_path)
        for i, img in enumerate(imgs):
            ax = axes[class_label, i]
            ax.imshow(img.squeeze(), cmap="gray")
            ax.axis("off")

        # Desenha o texto da label
        ax0 = axes[class_label, 0]
        bbox = ax0.get_position()
        y = bbox.y0 + bbox.height / 2
        fig.text(
            0.02, y,
            f"Label {class_label}",
            va="center", ha="left",
            fontsize=12, fontweight="bold"
        )

    plt.show()
