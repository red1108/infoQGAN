# python mnist_train.py --model_type QGAN --DIGITS_STR 0123456789 --DIGIT 1 --G_lr 0.01 --M_lr 0.0001 --D_lr 0.001 --coeff 0.05 --epochs 300 --latent_dim 16 --num_images_per_class 2000

# Standard Libraries
import math
import os
import pickle
import random
import time
from datetime import datetime

# Data Manipulation and Visualization
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # For saving figures
from IPython.display import clear_output
from tqdm import tqdm

# PyTorch Libraries and Tools
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


# Custom Modules (Autoencoder, QGAN2, Discriminator, MINE)
from modules import Autoencoder, QGAN, Discriminator, MINE
from modules.utils import convert_ipynb_to_html  # For saving HTML files
import importlib  # For reloading modules
importlib.reload(Autoencoder)
importlib.reload(QGAN)
importlib.reload(Discriminator)
importlib.reload(MINE)

# quantum computing
import pennylane as qml

import argparse
import json

# 전역 변수 선언
train_type = "InfoQGAN"
use_mine = True if train_type == "InfoQGAN" else False
DIGITS_STR = ""
DIGITS = []
TARGETS_STR = ""
TARGETS = []
G_lr = 0.005
M_lr = 0.0001
D_lr = 0.001
smooth = 0.0
SEED_R = 1.5
SEED_DIM = 10
epoch_num = 300
gamma = 0.8
latent_dim = 16
num_images_per_class = 2000
COEFF = 0.05
code_qubits = 3
ARGS = None

def visualize_autoencoder(autoencoder, data):

    # 클래스별로 10개의 latent 벡터를 선택하고 복원
    reconstructed_images = []
    for digit in tqdm(DIGITS):
        latent_data = data[f'{digit}_latent']  # 각 클래스의 latent 벡터 불러오기
        # 10개의 latent 벡터를 텐서로 변환
        latent_vectors = torch.tensor(latent_data[:10], dtype=torch.float32)
        
        # 오토인코더의 디코더로 복원
        with torch.no_grad():
            reconstructed = autoencoder.decoder(latent_vectors)  # (10, 1, 28, 28)
        reconstructed_images.append(reconstructed)

    # DIGITSx10 그리드에 시각화

    fig, axs = plt.subplots(len(DIGITS), 10, figsize=(10, len(DIGITS)))
    if len(DIGITS) == 1:
        axs = axs[np.newaxis, :]
    for i in range(len(DIGITS)):
        for j in range(10):
            # (1, 28, 28) 형태를 (28, 28)으로 변환하여 시각화
            axs[i, j].imshow(reconstructed_images[i][j].squeeze().detach().numpy(), cmap='gray')
            axs[i, j].axis('off')

    plt.suptitle(f"Reconstructed Images, dim={latent_dim}, data={num_images_per_class} per class")
    plt.show()

    # 시각화
    plt.figure(figsize=(12, 6))
    for i in DIGITS:
        plt.plot(data[f'{i}_latent'].mean(axis=0), label=f"{i}-latent")

    plt.title("Mean of Latent Vectors for Categories 0-9")
    plt.xlabel("Dimension")
    plt.ylabel("Mean Value")
    plt.legend(title="Category")
    plt.show()

# python mnist_train.py --model_type QGAN --DIGITS_STR 0123456789 --DIGIT 1 --G_lr 0.01 --M_lr 0.0001 --D_lr 0.001 --coeff 0.05 --epochs 300 --latent_dim 16 --num_images_per_class 2000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--model_type", choices=['InfoQGAN', 'QGAN'], required=True, help="Model type to use: InfoQGAN or QGAN")
    parser.add_argument("--DIGITS", type=str, required=True, help="Autoencoder trained digits")
    parser.add_argument("--TARGETS", type=str, required=True, help="Target digits")
    parser.add_argument("--G_lr", type=float, default=0.005, help="Learning rate for generator")
    parser.add_argument("--M_lr", type=float, default=0.0001, help="Learning rate for mine")
    parser.add_argument("--D_lr", type=float, default=0.001, help="Learning rate for discriminator")
    parser.add_argument("--coeff", type=float, default=0.05, help="Coefficient value used for InfoQGAN (not used for QGAN)")
    parser.add_argument("--seed", type=float, default=1.5, help="Seed value range (1, seeed)")
    parser.add_argument("--seed_dim", type=int, default=10, help="Seed dimension")
    parser.add_argument("--smooth", type=float, default=0.0, help="Discriminator label smoothing (efficient for QGAN)")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--gamma", type=float, default=0.8, help="Learning rate scheduler gamma (step = 30 epochs)")
    parser.add_argument("--latent_dim", type=int, required=True, help="Dimension of latent space")
    parser.add_argument("--code", type=int, default=3, help="Number of code qubits")
    parser.add_argument("--num_images_per_class", type=int, default=2000, help="Number of images per class")

    args = parser.parse_args()
    ARGS = args

    train_type = args.model_type
    use_mine = (train_type == 'InfoQGAN')
    DIGITS = list(map(int, args.DIGITS))
    DIGITS_STR = args.DIGITS
    TARGETS = list(map(int, args.TARGETS))
    TARGETS_STR = args.TARGETS

    G_lr = args.G_lr
    M_lr = args.M_lr
    D_lr = args.D_lr
    COEFF = args.coeff
    smooth = args.smooth
    SEED_R = args.seed
    assert SEED_R > 1, "Error: SEED_R must be greater than 1."
    SEED_DIM = args.seed_dim
    epoch_num = args.epochs
    gamma = args.gamma
    latent_dim = args.latent_dim
    num_images_per_class = args.num_images_per_class
    code_qubits = args.code

    print(f"Use Mine: {use_mine}")
    print(f"DIGITS: {DIGITS}")
    print(f"TARGETS: {TARGETS}")
    # assert that TARGETS are in DIGITS
    for target in TARGETS:
        assert target in DIGITS, f"Error: TARGET {target} is not in the list of DIGITS {DIGITS}."
    
    print(f"Generator Learning Rate: {G_lr}")
    print(f"Mine Learning Rate: {M_lr}")
    print(f"Discriminator Learning Rate: {D_lr}")
    if use_mine:
        print(f"InfoQGAN coefficient: {COEFF}")
    print(f"Smooth: {smooth}")
    print(f"Seed Range: 1 ~ {SEED_R}")
    print(f"Seed Dimension: {SEED_DIM}")
    print(f"Epochs: {epoch_num}")
    print(f"Gamma: {gamma}")
    print(f"Latent Dimension: {latent_dim}")
    print(f"Code Qubits: {code_qubits}")
    print(f"Number of Images per Class: {num_images_per_class}")


# 1. autoencoder 모델 준비
autoencoder = Autoencoder.Autoencoder(latent_dim=latent_dim)
autoencoder_epochs = 100
autoencoder_lr = 0.0001
autoencoder_coeff = 0.0005
autoencoder.load_state_dict(torch.load(f'savepoints/InfomaxEncoder_{DIGITS_STR}_{latent_dim}_ep{autoencoder_epochs}_lr{autoencoder_lr}_{autoencoder_coeff}.pth', weights_only=True))
autoencoder.eval()  # 평가 모드로 전환

# 2. 데이터 로드
data = np.load(f'./data/MNIST/{DIGITS_STR}_{latent_dim}_{autoencoder_epochs}_{autoencoder_lr}_{autoencoder_coeff}/mnist_{DIGITS_STR}_{latent_dim}_{num_images_per_class}.npz')

# 3. autoencoder 모델 시각화
visualize_autoencoder(autoencoder, data)

# 4. 학습 준비:
    # 학습 데이터, 테스트 데이터, 검증 데이터를 2:1:1로 나눈다.
    # cpu/gpu 설정 및 quantum device설정
    # n_qubits, code_qubits, noise_qubits, output_qubits 설정
print("이번 학습으로 생성할 숫자는", TARGETS, "입니다.")
# 원래는 DIGIT 하나만이었는데, 이제는 TARGETS 내부 숫자들을 모두 학습해야 한다.
train_dataset = np.concatenate([data[f'{target}_latent'][:num_images_per_class//4] for target in TARGETS], axis=0)
test_dataset = np.concatenate([data[f'{target}_latent'][num_images_per_class//2:num_images_per_class*3//4] for target in TARGETS], axis=0)
val_dataset = np.concatenate([data[f'{target}_latent'][num_images_per_class*3//4:] for target in TARGETS], axis=0)
train_size, test_size, val_size = len(train_dataset), len(test_dataset), len(val_dataset)
print("train_size =", train_size, "test_size =", test_size, "val_size =", val_size)

n_qubits = 5
noise_qubits = n_qubits - code_qubits
output_qubits = 5 # 출력 차원은 2**output_qubits 만큼.
assert(2**output_qubits >= latent_dim)
assert(code_qubits <= 2**n_qubits)
n_layers = 20
BATCH_SIZE = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("학습에 사용할 device =",device)
dev = qml.device("default.qubit", wires=n_qubits)


# 5. 생성자, 판별자, MINE, optimizer 초기화
generator_initial_params = Variable(torch.tensor(np.random.normal(-np.pi/3, np.pi/3, (n_layers, n_qubits, 1))), requires_grad=True)
generator = QGAN.QGAN4(n_qubits, output_qubits, n_layers, generator_initial_params, dev)
discriminator = Discriminator.LinearDiscriminator(input_dim = latent_dim, hidden_size=100) # 50 --> 25 변경
mine = MINE.LinearMine(code_dim=code_qubits, output_dim=latent_dim, size=100) # 50 --> 100 변경
print("n_qubits = {} n_layers = {} 총 파라미터 수 = {}".format(n_qubits, n_layers, generator_initial_params.numel()))

G_opt = torch.optim.Adam([generator.params], lr=G_lr)
D_opt = torch.optim.Adam(discriminator.parameters(), lr=D_lr)
M_opt = torch.optim.Adam(mine.parameters(), lr=M_lr)

G_scheduler = torch.optim.lr_scheduler.StepLR(G_opt, step_size=30, gamma=gamma)
D_scheduler = torch.optim.lr_scheduler.StepLR(D_opt, step_size=30, gamma=gamma)
M_scheduler = torch.optim.lr_scheduler.StepLR(M_opt, step_size=30, gamma=gamma)

# 학습에 사용할 train_step과 disc_cost_fn 정의 
def generator_train_step(generator_seed, coeff, use_mine = False):
    '''
    params (torch.Tensor(레이어,큐빗,3)): a parameter
    generator_input (torch.Tensor(BATCH_SIZE, n_qubits)): 생성기 입력 seed (noise + code). -1~1 사이의 값
    '''
    code_input = generator_seed[:, :code_qubits] # 입력중에서 code만 뽑는다. (BATCH_SIZE, code_qubits)

    # reverse generator_seed order
    generator_output = generator.forward(generator_seed) # 출력을 뽑아낸다 (BATCH_SIZE, 2**output_qubits)

    generator_output = generator_output.to(torch.float32) # (BATCH_SIZE,  2**output_qubits)
    # clipping first latent_dim
    generator_output = generator_output[:, :latent_dim] # (BATCH_SIZE, latent_dim)
    disc_output = discriminator(generator_output) # 밑에 코드에서 정의됨
    gan_loss = torch.log(1-disc_output).mean()
    
    if use_mine:
        pred_xy = mine(code_input, generator_output)
        code_input_shuffle = code_input[torch.randperm(BATCH_SIZE)]
        pred_x_y = mine(code_input_shuffle, generator_output)
        mi = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        gan_loss -= coeff * mi

    return generator_output, gan_loss# TODO: 이건 분석용으로 넣어놓음.지워야 함.

disc_loss_fn = nn.BCELoss()
def disc_cost_fn(real_input, fake_input):
    batch_num = real_input.shape[0]

    disc_real = discriminator(real_input)
    disc_fake = discriminator(fake_input)

    real_label = torch.ones((batch_num, 1)).to(device)
    fake_label = torch.zeros((batch_num, 1)).to(device)
    
    if smooth > 0.00001:
        real_label = real_label - smooth*torch.rand(real_label.shape).to(device)
    
    loss = 0.5 * (disc_loss_fn(disc_real, real_label) + disc_loss_fn(disc_fake, fake_label))
    
    return loss


def visualize_output_simple(gen_outputs, gen_codes, epoch, writer, image_file_path):
    # gen_outputs를 decoder에 통과시켜 이미지로 변환
    latent_vectors = torch.tensor(gen_outputs[:100], dtype=torch.float32)
    
    # 오토인코더의 디코더로 복원
    with torch.no_grad():
        reconstructed = autoencoder.decoder(latent_vectors)  # (100, 1, 28, 28)
    
    # 1. 첫 번째 플롯: 10*10 그리드에 reconstructed 이미지 시각화 (랜덤 순서)
    fig, axs = plt.subplots(10, 10, figsize=(10, 10), dpi=80)
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(reconstructed[i*10+j].squeeze().detach().numpy(), cmap='gray')
            axs[i, j].axis('off')
    plt.suptitle(f"TARGETS={TARGETS_STR}/{DIGITS_STR} epoch={epoch} dim={latent_dim}")
    
    writer.add_figure(f'2D Distribution', fig, epoch)
    fig.savefig(f'{image_file_path}/generated_epoch{epoch:03d}.png')
    plt.close(fig)

    # 2. code_qubits 값별로 정렬하여 10*10 이미지 배치 생성 및 저장
    code_qubits = gen_codes.shape[1]
    front_100_codes = gen_codes[:100]  # gen_codes에서 앞 100개의 코드만 사용
    for q in range(code_qubits):
        # 각 code_qubit 값으로 정렬
        sorted_indices = front_100_codes[:, q].argsort()
        sorted_reconstructed = reconstructed[sorted_indices]  # 정렬된 상위 100개 사용

        fig, axs = plt.subplots(10, 10, figsize=(10, 10), dpi=80)
        for i in range(10):
            for j in range(10):
                axs[i, j].imshow(sorted_reconstructed[i*10+j].squeeze().detach().numpy(), cmap='gray')
                axs[i, j].axis('off')
        plt.suptitle(f"TARGETS={TARGETS_STR} epoch={epoch} dim={latent_dim} code_qubit={q}")
        writer.add_figure(f'Sorted by Code Qubit {q}', fig, epoch) # TensorBoard에 기록
        fig.savefig(f'{image_file_path}/sorted_{q}_epoch{epoch:03d}.png') # 이미지 파일로 저장
        plt.close(fig)
    
    # latent vector의 평균값과 비교
    fig, ax = plt.subplots(dpi=80)
    for digit in TARGETS:
        ax.plot(data[f'{digit}_latent'].mean(axis=0), label=f"{digit}-latent")
    ax.plot(gen_outputs.mean(axis=0), label="Generated")
    ax.set_title(f"Latent compare TARGETS={TARGETS_STR}/{DIGITS_STR} epoch={epoch} dim={latent_dim}")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean Value")
    ax.legend(title="Category")

    writer.add_figure(f'Latent Compare', fig, epoch)
    fig.savefig(f'{image_file_path}/compare_epoch{epoch:03d}.png')
    plt.close(fig)
        
# 실제 학습 진행
from scipy.linalg import sqrtm

def calculate_frechet_distance(gen_outputs, val_dataset):
    # gen_outputs: (_, 2**output_qubits), val_dataset: (_, 2**output_qubits)
    # 평균과 공분산 계산
    mu1, sigma1 = gen_outputs.mean(axis=0), np.cov(gen_outputs, rowvar=False)
    mu2, sigma2 = val_dataset.mean(axis=0), np.cov(val_dataset, rowvar=False)

    # Frechet Distance 계산
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):  # 실수 부분만 사용
        covmean = covmean.real

    frechet_distance = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return frechet_distance


current_time = datetime.now().strftime("%b%d_%H_%M_%S")  # "Aug13_14_12_30" 형식
save_dir = f"./runs/MNIST{DIGITS_STR}_{TARGETS_STR}_ld{latent_dim}_{train_type}_{current_time}"
scalar_save_path = os.path.join(save_dir, f"MNIST{TARGETS_STR}_{train_type}_{current_time}.csv")
image_save_dir = os.path.join(save_dir, "images")
param_save_dir = os.path.join(save_dir, "params")
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(param_save_dir, exist_ok=True)

# ======================파이썬 코드를 html 로 만듦=======================
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

def convert_py_to_html(py_file_path, html_file_path):
    """Converts a Python script to a syntax-highlighted HTML file."""
    with open(py_file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    html_code = highlight(code, PythonLexer(), HtmlFormatter(full=True, linenos=True))
    
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_code)

    print(f"Converted {py_file_path} to {html_file_path} with syntax highlighting.")
convert_py_to_html('mnist_train2.py', os.path.join(save_dir, 'mnist_train2.html'))
# ==========================================================

# save ARGS in save_dir/args.txt
with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
    json.dump(ARGS.__dict__, f, indent=4)
    print(f"args 객체가 {save_dir}/args.txt 파일에 저장되었습니다.")

# CSV 파일 초기화 (헤더 작성)
df = pd.DataFrame(columns=['epoch', 'D_loss', 'G_loss', 'MI', 'FD', 'time'])

# TensorBoard SummaryWriter 초기화
writer = SummaryWriter(log_dir=save_dir)

start_time = time.time()

def categorical_distribution(S, E, T, size): # S~E를 T개로 내분하는 categorical distribution.
    if T == 1:
        categories = [(S+E)/2]
    else:
        categories = np.linspace(S, E, T)
    return torch.tensor(np.random.choice(categories, size))

from torch.utils.data import DataLoader, TensorDataset

train_tensor = torch.tensor(train_dataset, dtype=torch.float32)
train_loader = DataLoader(
    TensorDataset(train_tensor),
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    drop_last=True  # 마지막 배치 크기가 작으면 무시
)

for epoch in range(1, epoch_num+1):
    G_loss_sum = 0.0
    D_loss_sum = 0.0
    mi_sum = 0.0
    batch_num = train_size // BATCH_SIZE

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epoch_num}", unit="batch")

    # 그림 그릴때 필요하다
    gen_outputs = [] # (데이터수, 2**output_qubits) 출력들
    gen_codes = [] # (데이터수, code_qubits) 코드들
    coeff = COEFF * (0.1 + 0.9 * (epoch/epoch_num)) # 0.1 ~ 1비율로 선형적으로 증가

    for batch_idx, (batch,) in enumerate(pbar):  # batch unpack
        # # train generator
        generator_seed = torch.empty((BATCH_SIZE, SEED_DIM)).uniform_(1, SEED_R)
        generator_seed[:, 0] = categorical_distribution(1, SEED_R, len(TARGETS), BATCH_SIZE)
        generator_output, generator_loss = generator_train_step(generator_seed, coeff, use_mine=use_mine)
        G_opt.zero_grad()
        generator_loss.requires_grad_(True)
        generator_loss.backward()
        G_opt.step()
        
        # train discriminator
        fake_input = generator_output.detach().to(torch.float32)
        disc_loss = disc_cost_fn(batch, fake_input)
        D_opt.zero_grad()
        disc_loss.requires_grad_(True)
        disc_loss.backward()
        D_opt.step()

        # train mine
        code_input = generator_seed[:, :code_qubits] # (BATCH_SIZE, code_qubits) 코드만 추출
        pred_xy = mine(code_input, fake_input)
        code_input_shuffle = code_input[torch.randperm(BATCH_SIZE)]
        pred_x_y = mine(code_input_shuffle, fake_input)
        mi = -torch.mean(pred_xy) + torch.log(torch.mean(torch.exp(pred_x_y)))
        M_opt.zero_grad()
        mi.requires_grad_(True)
        mi.backward()
        M_opt.step()

        D_loss_sum += disc_loss.item()
        G_loss_sum += generator_loss.item()
        mi_sum -= mi.item() # (-1)곱해져 있어서 빼야함.

        gen_outputs.append(fake_input.detach().numpy())
        gen_codes.append(code_input.detach().numpy())

        pbar.set_postfix({'G_loss': G_loss_sum/(batch_idx+1), 'D_loss': D_loss_sum/(batch_idx+1), 'MI': mi_sum/(batch_idx+1)})

    G_scheduler.step()
    D_scheduler.step()
    M_scheduler.step()
    
    gen_outputs = np.concatenate(gen_outputs, axis=0) # (train_num, 2**output_qubits)
    gen_codes = np.concatenate(gen_codes, axis=0) # (train_num, code_qubits)

    D_loss, G_loss, mi = D_loss_sum/batch_num, G_loss_sum/batch_num, mi_sum/batch_num

    # calculate FD between generated data and val data
    frechet_distance = calculate_frechet_distance(gen_outputs, val_dataset)
    

    writer.add_scalar('Loss/d_loss', D_loss, epoch)
    writer.add_scalar('Loss/g_loss', G_loss, epoch)
    writer.add_scalar('Metrics/mi', mi, epoch)
    writer.add_scalar('Metrics/FD', frechet_distance, epoch)

    # 스칼라 값 CSV로 덮어쓰기 저장
    file_exists = os.path.isfile(scalar_save_path)
    new_data = pd.DataFrame({
        'epoch': [epoch],
        'D_loss': [D_loss],
        'G_loss': [G_loss],
        'MI': [mi],
        'FD': [frechet_distance],
        'time': [int((time.time() - start_time)*1000)]
    })

    new_data.to_csv(scalar_save_path, mode='a', header=not file_exists)
    
    visualize_output_simple(gen_outputs, gen_codes, epoch, writer, image_save_dir) # save fig here

    # 각 epoch마다 generator 파라미터 저장
    torch.save(generator.params, f'{param_save_dir}/generator_params_epoch{epoch}.pth')
    
    print("epoch: {}, D_loss: {}, G_loss: {}, MI = {}, FD = {}".format(epoch, D_loss, G_loss, mi, frechet_distance))
