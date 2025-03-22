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
from modules.utils import convert_ipynb_to_html, convert_py_to_html # 현재 html파일 저장을 위해 사용
import argparse
import json


train_type = "InfoGAN"
use_mine = True if train_type == "InfoGAN" else False

data_num = 1000
data_type = "biased_diamond"

BATCH_SIZE = 16
SEED = 1
SEED_DIM = 5
code_dim = 2
epoch_num = 300
output_dim = 2

G_lr = 0.001
D_lr = 0.0003
M_lr = 0.001
coeff = 0.1
hidden_dim = 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--model_type", choices=['InfoGAN', 'GAN'], required=True, help="Model type to use: InfoQAN or GAN")
    parser.add_argument("--data_num", type=int, default=1000, help="Number of data points")
    parser.add_argument("--data_type", choices=['biased_diamond', 'biased_circle', '2box', 'bigdiamond'], required=True, help="Data type to use")

    parser.add_argument("--seed", type=float, default=1.0, help="Seed value range")
    parser.add_argument("--seed_dim", type=int, default=5, help="Seed dimension")
    parser.add_argument("--code_dim", type=int, default=2, help="Number of latent code space")

    parser.add_argument("--G_lr", type=float, default=0.001, help="Learning rate for generator")
    parser.add_argument("--M_lr", type=float, default=0.0003, help="Learning rate for mine")
    parser.add_argument("--D_lr", type=float, default=0.001, help="Learning rate for discriminator")
    parser.add_argument("--coeff", type=float, default=0.1, help="Coefficient value used for InfoQGAN (not used for QGAN)")

    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--hidden_dim", type=int, default=5, help="Number of epochs")

    
    args = parser.parse_args()
    ARGS = args
    
    train_type = args.model_type
    use_mine = (train_type == 'InfoGAN')
    data_type = args.data_type
    data_num = args.data_num
    SEED = args.seed
    SEED_DIM = args.seed_dim
    code_dim = args.code_dim
    
    G_lr = args.G_lr
    M_lr = args.M_lr
    D_lr = args.D_lr
    COEFF = args.coeff
    epoch_num = args.epochs
    hidden_dim = args.hidden_dim


    print(f"Use Mine: {use_mine}")
    print(f"Data Type: {data_type}, Data size: {data_num}")

    print(f"seed dim = {SEED_DIM}, code dim = {code_dim}")
    print(f"Seed Range: {-SEED} ~ {SEED}, Epochs: {epoch_num}")

    print(f"Generator Learning Rate: {G_lr}")
    print(f"Mine Learning Rate: {M_lr}")
    print(f"Discriminator Learning Rate: {D_lr}")
    if use_mine:
        print(f"InfoQGAN coefficient: {COEFF}")
    print(f"Epochs: {epoch_num}")

# Load data
train_in = np.loadtxt(f'data/2D/{data_type}_{data_num}_1.txt')

# setting torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


import importlib
from modules import Discriminator, MINE, Generator  # 초기 import
importlib.reload(Generator)  # 모듈 갱신
importlib.reload(Discriminator)  # 모듈 갱신
importlib.reload(MINE)  # 모듈 갱신

# 생성자 파라미터 초기화 및 모듈 불러오기
generator = Generator.LinearGeneratorSigmoid(input_dim=SEED_DIM, output_dim=2, hidden_size=hidden_dim)
# 판별자, MINE 초기화
discriminator = Discriminator.LinearDiscriminator(input_dim = output_dim)
mine = MINE.LinearMine(code_dim=code_dim, output_dim=output_dim)

# Function to calculate total trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print total number of trainable parameters
print(f"Total trainable parameters in Generator: {count_parameters(generator)}")
print(f"Total trainable parameters in Discriminator: {count_parameters(discriminator)}")
print(f"Total trainable parameters in MINE: {count_parameters(mine)}")

G_opt = torch.optim.Adam(generator.parameters(), lr=G_lr)
D_opt = torch.optim.Adam(discriminator.parameters(), lr=D_lr)
M_opt = torch.optim.Adam(mine.parameters(), lr=M_lr)

def generator_train_step(generator_input, use_mine = False):
    '''
    params (torch.Tensor(레이어,큐빗,3)): a parameter
    generator_input (torch.Tensor(BATCH_SIZE, n_qubits)): 생성기 입력 seed (noise + code). -1~1 사이의 값
    '''
    code_input = generator_input[:, -code_dim:] # 입력중에서 code만 뽑는다. (BATCH_SIZE, code_qubits)

    generator_output = generator.forward(generator_input) # 출력을 뽑아낸다 (BATCH_SIZE, output_dim)

    generator_output = generator_output.to(torch.float32) # (BATCH_SIZE, output_dim)
    
    disc_output = discriminator(generator_output) # 밑에 코드에서 정의됨
    gan_loss = torch.log(1-disc_output).mean()
    
    if use_mine:
        pred_xy = mine(code_input, generator_output)
        code_input_shuffle = code_input[torch.randperm(BATCH_SIZE)]
        pred_x_y = mine(code_input_shuffle, generator_output)
        mi = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        gan_loss -= coeff * mi

    return generator_output, gan_loss

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

current_time = datetime.now().strftime("%b%d_%H_%M_%S")  # "Aug13_14_12" 형식
base_name = f'{train_type}_sd{SEED_DIM}_hd{hidden_dim}_{data_type}_{current_time}'
save_dir = f"./2d_runs/{base_name}"
scalar_save_path = os.path.join(save_dir, f"scalars.csv")
image_save_dir = os.path.join(save_dir, "images")
numpy_save_dir = os.path.join(save_dir, "numpy")
param_save_dir = os.path.join(save_dir, "params")
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(numpy_save_dir, exist_ok=True)
os.makedirs(param_save_dir, exist_ok=True)

# ==================Convert code to HTML================
convert_py_to_html('2d_train_classical.py', os.path.join(save_dir, '2d_train_classical.html'))
# ==========================================================

# save ARGS in save_dir/args.txt
with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
    json.dump(ARGS.__dict__, f, indent=4)
    print(f"args 객체가 {save_dir}/args.txt 파일에 저장되었습니다.")


# CSV 파일 초기화 (헤더 작성)
if not os.path.exists(scalar_save_path):
    df = pd.DataFrame(columns=['epoch', 'D_loss', 'G_loss', 'MI', 'D_ks', 'p_value', 'angle', 'time'] + 
                  [f'Corr/code{i}-{axis}' for i in range(code_dim) for axis in ['x', 'y']])

# TensorBoard SummaryWriter 초기화
writer = SummaryWriter(log_dir=save_dir)

start_time = time.time()

for epoch in range(1, epoch_num+1):
    G_loss_sum = 0.0
    D_loss_sum = 0.0
    mi_sum = 0.0
    batch_num = len(train_in) // BATCH_SIZE
    pbar = tqdm(range(batch_num))

    # 그림 그릴때 필요하다
    gen_outputs = [] # (데이터수, 2) 생성한 모든 점의 좌표들
    gen_codes = [] # (데이터수, 2) 점 찍는데 들어간 code들

    for batch_idx in pbar:
        batch = torch.FloatTensor(train_in[BATCH_SIZE * batch_idx : BATCH_SIZE * batch_idx + BATCH_SIZE])

        # train generator
        generator_seed = torch.empty((BATCH_SIZE, SEED_DIM)).uniform_(-SEED, SEED)
        generator_output, generator_loss = generator_train_step(generator_seed, use_mine=use_mine)
        G_opt.zero_grad()
        generator_loss.requires_grad_(True)
        generator_loss.backward()
        G_opt.step()
        
        # train discriminator
        fake_input = generator_output.detach().to(torch.float32)
        disc_loss = disc_cost_fn(batch, fake_input, smoothing=False)
        D_opt.zero_grad()
        disc_loss.requires_grad_(True)
        disc_loss.backward()
        D_opt.step()

        # train mine
        code_input = generator_seed[:, -code_dim:] # (BATCH_SIZE, code_qubits) 코드만 추출
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

        gen_outputs.append(fake_input.numpy())
        gen_codes.append(code_input.numpy())

        pbar.set_postfix({'G_loss': G_loss_sum/(batch_idx+1), 'D_loss': D_loss_sum/(batch_idx+1), 'MI': mi_sum/(batch_idx+1)})

    # G_scheduler.step()
    # D_scheduler.step()
    # M_scheduler.step()
    
    gen_outputs = np.concatenate(gen_outputs, axis=0) # (train_num, 2)
    gen_codes = np.concatenate(gen_codes, axis=0) # (train_num, 2)

    D_loss, G_loss, mi = D_loss_sum/batch_num, G_loss_sum/batch_num, mi_sum/batch_num
    p_value, D_ks = ndtest.ks2d2s(gen_outputs[:, 0], gen_outputs[:, 1], train_in[:, 0], train_in[:, 1], extra=True)

    writer.add_scalar('Loss/d_loss', D_loss, epoch)
    writer.add_scalar('Loss/g_loss', G_loss, epoch)
    writer.add_scalar('Metrics/mi', mi, epoch)
    writer.add_scalar('Metrics/D_ks', D_ks, epoch)
    writer.add_scalar('Metrics/p_value', p_value, epoch)

    

    # code와 x, y의 상관관계를 측정 후 기록
    df = pd.DataFrame({'x': gen_outputs[:, 0], 'y': gen_outputs[:, 1]})
    for i in range(code_dim):
        df[f'code{i}'] = gen_codes[:, i]
    corr_mat = df.corr().to_numpy()
    for i in range(code_dim):
        writer.add_scalar(f'Corr/code{i}-x', corr_mat[0, i+2], epoch)
        writer.add_scalar(f'Corr/code{i}-y', corr_mat[1, i+2], epoch)

    # code0과 code1 사이의 각도 계산 (벡터의 내적 및 크기 사용)
    cos_theta = (corr_mat[0, 2] * corr_mat[0, 3] + corr_mat[1, 2] * corr_mat[1, 3]) / (
        np.sqrt(corr_mat[0, 2]**2 + corr_mat[1, 2]**2) * np.sqrt(corr_mat[0, 3]**2 + corr_mat[1, 3]**2)
    )
    theta_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    # 예각으로 변환
    theta_degrees = min(theta_degrees, 180 - theta_degrees)
    writer.add_scalar('Corr/angle', theta_degrees, epoch)


    # 스칼라 값 CSV로 덮어쓰기 저장
    file_exists = os.path.isfile(scalar_save_path)
    new_data = pd.DataFrame({
        'epoch': [epoch],
        'D_loss': [D_loss],
        'G_loss': [G_loss],
        'MI': [mi],
        'D_ks': [D_ks],
        'p_value': [p_value],
        'time': [int((time.time() - start_time)*1000)],
        **{f'Corr/code{i}-{axis}': [corr_mat[j, i+2]] for i in range(code_dim) for j, axis in enumerate(['x', 'y'])},
        'angle': [theta_degrees]  # code0과 code1 사이의 예각 추가
    })

    new_data.to_csv(scalar_save_path, mode='a',  header=not file_exists)
    
    if epoch % 5 == 0:
        visualize_output_simple(gen_outputs, gen_codes, epoch, writer, image_save_dir) # save fig here

    # 각 epoch마다 numpy의 savetxt를 사용하여 저장
    output_file_path = os.path.join(numpy_save_dir, f"gen_outputs_epoch_{epoch}.txt")
    codes_file_path = os.path.join(numpy_save_dir, f"gen_codes_epoch_{epoch}.txt")

    np.savetxt(output_file_path, gen_outputs)
    np.savetxt(codes_file_path, gen_codes)
    torch.save(generator.state_dict(), f'{param_save_dir}/generator_params_epoch{epoch}.pth') # QGAN 파라미터 저장
    
    #print("epoch: {}, D_loss: {}, G_loss: {}, MI = {}".format(epoch, D_loss, G_loss, mi))
    #print("좌표값 평균 = ", np.mean(gen_outputs[:,0]), np.mean(gen_outputs[:,1]), "디버그 =", generator.params[0][0][0].item())