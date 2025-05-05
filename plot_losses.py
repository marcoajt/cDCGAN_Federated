#!/usr/bin/env python3
# plot_losses.py

import json
import matplotlib.pyplot as plt
import argparse

def main(json_path: str, output_path: str = None):
    # 1) Carrega o arquivo JSON
    with open(json_path, "r") as f:
        history = json.load(f)

    loss_G = history.get("loss_G", [])
    loss_D = history.get("loss_D", [])

    # 2) Eixo x (índice de batch/época)
    x = list(range(1, len(loss_G) + 1))

    # 3) Plota as curvas
    plt.figure(figsize=(8, 5))
    plt.plot(x, loss_G, label="Generator Loss")
    plt.plot(x, loss_D, label="Discriminator Loss")
    plt.xlabel("Batch / Época")
    plt.ylabel("Loss")
    plt.title("Histórico de Loss do Generator e Discriminator")
    plt.legend()
    plt.grid(True)

    # 4) Salva ou exibe
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figura salva em: {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plota losses de Generator e Discriminator a partir de um JSON."
    )
    # Agora o positional é opcional (nargs='?') e assume o arquivo padrão se não for passado
    parser.add_argument(
        "json_path",
        nargs="?",
        default="loss_history.json",
        help="Caminho para o loss_history.json (padrão: %(default)s)"
    )
    parser.add_argument(
        "--out", "-o", type=str, default=None,
        help="(Opcional) Caminho para salvar a figura (PNG). Se omitido, exibe na tela."
    )
    args = parser.parse_args()

    main(args.json_path, args.out)
