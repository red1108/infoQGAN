import argparse

def main(G_lr, M_lr, D_lr, epochs, use_mine, DIGITS_STR, latent_dim, num_images_per_class, DIGIT):
    DIGITS = list(map(int, DIGITS_STR))
    print(f"Generator Learning Rate: {G_lr}")
    print(f"Mine Learning Rate: {M_lr}")
    print(f"Discriminator Learning Rate: {D_lr}")
    print(f"Epochs: {epochs}")
    print(f"Use Mine: {use_mine}")
    print(f"DIGITS: {DIGITS}")
    print(f"Latent Dimension: {latent_dim}")
    print(f"Number of Images per Class: {num_images_per_class}")
    print(f"DIGIT: {DIGIT}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--G_lr", type=float, required=True, help="Learning rate for generator")
    parser.add_argument("--M_lr", type=float, required=True, help="Learning rate for mine")
    parser.add_argument("--D_lr", type=float, required=True, help="Learning rate for discriminator")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--use_mine", type=bool, default=False, help="Use mine option")
    parser.add_argument("--DIGITS_STR", type=str, required=True, help="String of digits to parse into DIGITS array")
    parser.add_argument("--latent_dim", type=int, required=True, help="Dimension of latent space")
    parser.add_argument("--num_images_per_class", type=int, required=True, help="Number of images per class")
    parser.add_argument("--DIGIT", type=int, required=True, help="Target digit")

    args = parser.parse_args()
    main(args.G_lr, args.M_lr, args.D_lr, args.epochs, args.use_mine, args.DIGITS_STR, args.latent_dim, args.num_images_per_class, args.DIGIT)
