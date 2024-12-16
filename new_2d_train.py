# QGAN 구조 통일. 기존엔 Rx Ry Rz 썼지만, Ry 만 사용하도록 수정.
# Standard Libraries
import math
import pickle
import random
import numpy as np

# Data Manipulation and Visualization
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # For saving figures
from IPython.display import clear_output
from tqdm import tqdm

# Quantum Computing
import pennylane as qml

# Deep Learning
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

# Utility Functions
from functools import reduce
import ndtest # 2D 분포 검정에 사용
from datetime import datetime
import os
import time
from modules.utils import convert_ipynb_to_html # 현재 html파일 저장을 위해 사용
import argparse
import json


train_type = "InfoQGAN"
use_mine = True if train_type == "InfoQGAN" else False

data_num = 1000
data_type = "biased_diamond"
noise_qubits = 3
code_qubits = 2
n_qubits = noise_qubits + code_qubits
output_qubits = 2
assert(output_qubits <= n_qubits) # 출력 큐빗은 qubit이하여야 한다.
n_layers = 10
BATCH_SIZE = 16

G_lr = 0.001
D_lr = 0.0003
M_lr = 0.001
coeff = 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--model_type", choices=['InfoQGAN', 'QGAN'], required=True, help="Model type to use: InfoQGAN or QGAN")
    parser.add_argument("--n_layers", type=int, default=10, help="Number of layers for QGAN")
    parser.add_argument("--G_lr", type=float, default=0.001, help="Learning rate for generator")
    parser.add_argument("--M_lr", type=float, default=0.0003, help="Learning rate for mine")
    parser.add_argument("--D_lr", type=float, default=0.001, help="Learning rate for discriminator")
    parser.add_argument("--coeff", type=float, default=0.1, help="Coefficient value used for InfoQGAN (not used for QGAN)")
    parser.add_argument("--seed", type=float, default=1.5, help="Seed value range")
    parser.add_argument("--smooth", type=float, default=0.0, help="Discriminator label smoothing (efficient for QGAN)")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--code", type=int, default=3, help="Code dimension")
    
    args = parser.parse_args()
    ARGS = args

    train_type = args.model_type
    use_mine = (train_type == 'InfoQGAN')
    DIGITS = list(map(int, args.DIGITS))
    DIGITS_STR = args.DIGITS
    TARGETS = list(map(int, args.TARGETS))
    TARGETS_STR = args.TARGETS

    n_layers = args.n_layers
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
    
    print(f"Number of Layers: {n_layers}")
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


# Load data
train_in = np.loadtxt(f'data/2D/{data_type}_{data_num}_1.txt')

# setting torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dev = qml.device("default.qubit", wires=n_qubits)

import importlib
from modules import QGAN, Discriminator, MINE  # 초기 import
importlib.reload(QGAN)  # 모듈 갱신
importlib.reload(Discriminator)  # 모듈 갱신
importlib.reload(MINE)  # 모듈 갱신

# 생성자 파라미터 초기화 및 모듈 불러오기
generator_initial_params = Variable(torch.tensor(np.random.normal(-np.pi/2 , np.pi/2, (n_layers, n_qubits, 3))), requires_grad=True)
generator = QGAN.QGAN(n_qubits, output_qubits, n_layers, generator_initial_params, dev)

# 판별자, MINE 초기화
discriminator = Discriminator.LinearDiscriminator(input_dim = output_qubits)
mine = MINE.LinearMine(code_dim=code_qubits, output_dim=output_qubits)

G_opt = torch.optim.Adam([generator.params], lr=G_lr)
D_opt = torch.optim.Adam(discriminator.parameters(), lr=D_lr)
M_opt = torch.optim.Adam(mine.parameters(), lr=M_lr)


def bitwise_sums(arr):
    n = len(arr).bit_length() - 1  # 비트 길이를 계산하여 반복 횟수를 정함
    sums = torch.zeros(n, dtype=arr.dtype, device=arr.device)  # 결과를 저장할 텐서
    for bit in range(n):
        # 조건에 맞는 인덱스 선택을 위해 i-th 비트를 검사
        mask = (torch.arange(len(arr), device=arr.device) >> bit) & 1
        sums[bit] = arr[mask.bool()].sum()  # 조건에 맞는 원소들의 합산
    return sums

def output_postprocessing(arr):
    # arr: (BATCH_SIZE, output_qubits**2)
    # return: (BATCH_SIZE, output_qubits)
    ret = torch.stack([bitwise_sums(arr[i]) for i in range(len(arr))])
    return ret


def generator_train_step(generator_input, use_mine = False):
    '''
    params (torch.Tensor(레이어,큐빗,3)): a parameter
    generator_input (torch.Tensor(BATCH_SIZE, n_qubits)): 생성기 입력 seed (noise + code). -1~1 사이의 값
    '''
    code_input = generator_input[:, -code_qubits:] # 입력중에서 code만 뽑는다. (BATCH_SIZE, code_qubits)

    generator_output = generator.forward(generator_input) # 출력을 뽑아낸다 (BATCH_SIZE, 2**output_qubits)
    generator_output = output_postprocessing(generator_output) # (BATCH_SIZE, output_qubits)

    generator_output = generator_output.to(torch.float32) # (BATCH_SIZE, output_qubits)
    
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
def disc_cost_fn(real_input, fake_input, smoothing=False):
    batch_num = real_input.shape[0]

    disc_real = discriminator(real_input)
    disc_fake = discriminator(fake_input)

    real_label = torch.ones((batch_num, 1)).to(device)
    fake_label = torch.zeros((batch_num, 1)).to(device)
    
    if smoothing:
        real_label = real_label - 0.2*torch.rand(real_label.shape).to(device)
    
    loss = 0.5 * (disc_loss_fn(disc_real, real_label) + disc_loss_fn(disc_fake, fake_label))
    
    return loss

def visualize_output_simple(log_gen_outputs, log_gen_codes, epoch, writer, image_file_path):
    # 1. 첫 번째 플롯: log_gen_outputs의 2차원 점 분포
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(log_gen_outputs[:, 0], log_gen_outputs[:, 1], s=10, alpha=0.5)
    ax1.set_title(f'Epoch {epoch} - 2D Distribution')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid()

    # 2. 두 번째 플롯: log_gen_codes의 첫 번째 열을 색상으로 사용
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    scatter2 = ax2.scatter(log_gen_outputs[:, 0], log_gen_outputs[:, 1], s=10, c=log_gen_codes[:, 0], cmap='RdYlBu', alpha=0.5)
    ax2.set_title(f'Epoch {epoch} - Code 1 Color')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid()
    fig2.colorbar(scatter2, ax=ax2)  # 색상 막대 추가

    # 3. 세 번째 플롯: log_gen_codes의 두 번째 열을 색상으로 사용
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    scatter3 = ax3.scatter(log_gen_outputs[:, 0], log_gen_outputs[:, 1], s=10, c=log_gen_codes[:, 1], cmap='RdYlBu', alpha=0.5)
    ax3.set_title(f'Epoch {epoch} - Code 2 Color')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.grid()
    fig3.colorbar(scatter3, ax=ax3)  # 색상 막대 추가

    # TensorBoard에 기록
    writer.add_figure(f'2D Distribution', fig1, epoch)
    writer.add_figure(f'Code 1 Color', fig2, epoch)
    writer.add_figure(f'Code 2 Color', fig3, epoch)

    # fig1, fig2, fig3 를 image file로 저장
    fig1.savefig(f'{image_file_path}/dist_epoch_{epoch}.png')
    fig2.savefig(f'{image_file_path}/code1_epoch_{epoch}.png')
    fig3.savefig(f'{image_file_path}/code2_epoch_{epoch}.png')

    # 메모리 관리를 위해 plt를 닫음
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)