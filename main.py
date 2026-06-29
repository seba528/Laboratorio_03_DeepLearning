import argparse
import sys
from src.e1_baseline import ejecutar_baseline
from src.e2_mlp import ejecutar_mlp
from src.e3_cnn import ejecutar_cnn
from src.e4_resnet import ejecutar_resnet
from src.e5_finetune import ejecutar_finetuning
def main():
    parser = argparse.ArgumentParser(description="Gestor de Experimentos Lab 3")
    parser.add_argument('--exp', type=int, required=True, help="Número del experimento (1, 2, 3...)")

    args = parser.parse_args()

    if args.exp == 1:
        ejecutar_baseline()
    elif args.exp == 2:
        ejecutar_mlp()
    elif args.exp == 3:
        ejecutar_cnn()
    elif args.exp == 4:
        ejecutar_resnet()
    elif args.exp == 5:
        ejecutar_finetuning()
    else:
        print("Experimento no implementado aún.")

if __name__ == "__main__":
    main()
