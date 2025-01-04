import math
from itertools import product
import os
import random
import time
from datetime import datetime
import argparse
import json

# Data Manipulation and Visualization
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # For saving figures
import seaborn as sns
from IPython.display import clear_output
from tqdm import tqdm

# PyTorch Libraries and Tools
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

import importlib
from modules import QGAN, Discriminator, MINE  # 초기 import
importlib.reload(QGAN)  # 모듈 갱신
importlib.reload(Discriminator)  # 모듈 갱신
importlib.reload(MINE)  # 모듈 갱신

# quantum computing
import pennylane as qml
from modules.utils import generate_orthonormal_states

# 전역 변수 선언
train_type = "InfoQGAN"
data_type = torch.float32
use_mine = True if train_type == "InfoQGAN" else False
number_of_basis = 2
n_qubits = 3
G_layers = 5
D_layers = 5
dim = 2**n_qubits
basis_states = generate_orthonormal_states(dim, number_of_basis)
code_qubits = number_of_basis

train_size = 300
BATCH_SIZE = 16

epoch_num = 300
G_lr = 0.001
D_lr = 0.001
M_lr = 0.001
gamma = 0.8
smooth = 0.0
COEFF = 0.05

SEED = 0.5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument("--model_type", choices=['InfoQGAN', 'QGAN'], required=True, help="Model type to use: InfoQGAN or QGAN")
    parser.add_argument("--data_type", choices=['real', 'complex'], required=True, help="Data type to use: real or complex")
    parser.add_argument("--n_qubits", type=int, default=3, help="Number of qubits")
    parser.add_argument("--n_basis", type=int, required=True, help="Number of basis states")
    parser.add_argument("--train_size", type=int, default=300, help="Training data size")
    parser.add_argument("--G_layers", type=int, default=5, help="Number of layers for generator")
    parser.add_argument("--D_layers", type=int, default=5, help="Number of layers for discriminator")
    parser.add_argument("--G_lr", type=float, default=0.001, help="Learning rate for generator")
    parser.add_argument("--D_lr", type=float, default=0.001, help="Learning rate for discriminator")
    parser.add_argument("--M_lr", type=float, default=0.001, help="Learning rate for mine")
    parser.add_argument("--coeff", type=float, default=0.05, help="Coefficient value used for InfoQGAN (not used for QGAN)")
    parser.add_argument("--seed", type=float, default=0.5, help="Seed value range (-seed, seeed)")
    parser.add_argument("--smooth", type=float, default=0.0, help="Discriminator label smoothing (efficient for QGAN)")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--gamma", type=float, default=0.8, help="Learning rate scheduler gamma (step = 30 epochs)")
    parser.add_argument("--code", type=int, default=3, help="Code dimension")

    args = parser.parse_args()
    ARGS = args

    train_type = args.model_type
    if args.data_type == 'real':
        data_type = torch.float32
    elif args.data_type == 'complex':
        data_type = torch.complex128
    train_size = args.train_size
    use_mine = (train_type == 'InfoQGAN')
    n_qubits = args.n_qubits
    dim = 2**n_qubits
    number_of_basis = args.n_basis
    basis_states = generate_orthonormal_states(dim, number_of_basis, args.data_type) # 학습에 사용할 기본 basis들

    G_layers = args.G_layers
    D_layers = args.D_layers

    code_qubits = args.code

    if code_qubits != number_of_basis:
        for i in range(5):
            print(f"Warning: code_qubits ({code_qubits}) is not equal to number_of_basis ({number_of_basis}).")

    G_lr = args.G_lr
    M_lr = args.M_lr
    D_lr = args.D_lr
    COEFF = args.coeff
    smooth = args.smooth
    SEED = args.seed
    epoch_num = args.epochs
    gamma = args.gamma

    print(f"Model Type: {train_type}, Data Type: {data_type}")
    print(f"Basis States: ({number_of_basis}, {dim})")
    print(f"Train Size: {train_size}")
    print(f"Use Mine: {use_mine}")
    print(f"n_qubits: {n_qubits}")
    print(f"Number of Basis: {number_of_basis}")
    print(f"Generator Layers: {G_layers}, Discriminator Layers: {D_layers}")
    print(f"Learning Rate: G: {G_lr} D: {D_lr} M: {M_lr}")

    if use_mine:
        print(f"InfoQGAN coefficient: {COEFF}")
    print(f"Smooth: {smooth}")
    print(f"Seed Range: {-SEED} ~ {SEED}")
    print(f"Epochs: {epoch_num}")
    print(f"Gamma: {gamma}")
    print(f"Code Qubits: {code_qubits}")

print("Orthogonal states shape:", basis_states.shape)  # (number_of_basis, dim)
assert np.allclose(basis_states @ basis_states.T.conj(), np.eye(number_of_basis)), "state vectors are not orthogonal"

ml_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
quantum_device = qml.device("default.qubit", wires=n_qubits)
print("고전 머신러닝 device =", ml_device, "양자 회로 backend =", quantum_device)

# 모델 정의
if data_type == torch.float32:
    generator_initial_params = Variable(torch.tensor(np.random.normal(-np.pi, np.pi, (G_layers, n_qubits, 1))), requires_grad=True)
elif data_type == torch.complex128:
    generator_initial_params = Variable(torch.tensor(np.random.normal(-np.pi, np.pi, (G_layers, n_qubits, 3))), requires_grad=True)
generator_mode = "RY" if data_type == torch.float32 else "RXYZ"
generator = QGAN.QGenerator(n_qubits, n_qubits, G_layers, generator_initial_params, quantum_device, mode=generator_mode) # 상태를 얻어야 Discriminator에 넣음.

if data_type == torch.float32:
    discriminator = Discriminator.LinearDiscriminator(input_dim = 2**n_qubits, hidden_size=100)

elif data_type == torch.complex128:
    discriminator_initial_params = Variable(torch.tensor(np.random.normal(-np.pi, np.pi, (D_layers, n_qubits, 3))), requires_grad=True)
    discriminator = Discriminator.QDiscriminator(n_qubits, D_layers, discriminator_initial_params, quantum_device)

mine = MINE.LinearMine(code_dim=code_qubits, output_dim=2**n_qubits, size=100)

G_opt = torch.optim.Adam(generator.parameters(), lr=G_lr)
D_opt = torch.optim.Adam(discriminator.parameters(), lr=D_lr)
M_opt = torch.optim.Adam(mine.parameters(), lr=M_lr)
G_scheduler = torch.optim.lr_scheduler.StepLR(G_opt, step_size=30, gamma=gamma)
D_scheduler = torch.optim.lr_scheduler.StepLR(D_opt, step_size=30, gamma=gamma)
M_scheduler = torch.optim.lr_scheduler.StepLR(M_opt, step_size=30, gamma=gamma)

from itertools import product
def combine_quantum_states(states, train_size, combine_mode):
    """
    states: shape = (num_of_states, dim_of_state)
    train_size: 최종적으로 만들고자 하는 데이터셋 크기
    combine_mode: "linspace" or "uniform"
    """
    assert combine_mode in ["linspace", "uniform"], "combine_mode should be 'linspace' or 'uniform'"

    num_of_states = len(states)

    if combine_mode == "uniform":
        # 전부 uniform 방식 (Dirichlet)으로 생성
        alpha = np.ones(num_of_states)
        coefs = np.random.dirichlet(alpha, size=train_size)

    else:  # linspace
        x = int(train_size**(1/num_of_states)) # linspace 비율 분할 개수
        combos = np.array(list(product(range(x), repeat=num_of_states)))
        combos = combos[1:] # 0, 0, 0, ... 0 제외
        combos = combos / combos.sum(axis=1, keepdims=True)
        coefs = combos
        
        # 만약 linspace로 만든 개수가 train_size보다 적으면, 부족분을 uniform으로 채워서 결합
        if len(coefs) < train_size:
            shortfall = train_size - len(coefs)
            alpha = np.ones(num_of_states)
            extra_coefs = np.random.dirichlet(alpha, size=shortfall)
            coefs = np.concatenate([coefs, extra_coefs], axis=0)

        # 혹시 linspace로 만들었을 때 x**num_of_states가 train_size보다 많아도
        # 여기서 슬라이싱으로 잘라서 정확히 train_size만 맞춰줌
        coefs = coefs[:train_size]

    # 최종적으로 sqrt(coefficients) 한 뒤 상태들을 섞어서 반환
    return np.dot(np.sqrt(coefs), states)


train_dataset = combine_quantum_states(basis_states, train_size, "linspace")
train_tensor = torch.tensor(train_dataset, dtype=data_type).to(ml_device)
assert np.allclose(np.linalg.norm(train_dataset, axis=1), np.ones(train_size)), "combined states are not normalized"

def generator_train_step(generator_seed, coeff, use_mine = False):
    '''
    params (torch.Tensor(레이어,큐빗,3)): a parameter
    generator_input (torch.Tensor(BATCH_SIZE, seed_dim)): 생성기 입력 seed (code+noise).
    '''
    code_input = generator_seed[:, :code_qubits] # 입력중에서 code만 뽑는다. (BATCH_SIZE, code_qubits)
    generator_states = generator.forward(generator_seed) # 출력을 뽑아낸다 (BATCH_SIZE, 2**output_qubits) * 2
    generator_probs = abs(generator_states).to(torch.float32)
    generator_states = generator_states.to(data_type)
    
    disc_output = discriminator(generator_states) # 밑에 코드에서 정의됨
    # gan_loss = torch.log(1-disc_output).mean()
    gan_loss = torch.log(1-disc_output).mean()
    
    if use_mine:
        pred_xy = mine(code_input, generator_probs)
        code_input_shuffle = code_input[torch.randperm(BATCH_SIZE)]
        pred_x_y = mine(code_input_shuffle, generator_probs)
        mi = torch.mean(pred_xy) - torch.log(torch.mean(torch.exp(pred_x_y)))
        gan_loss -= coeff * mi

    return generator_probs, generator_states, gan_loss

disc_loss_fn = nn.BCELoss()
def disc_cost_fn(real_input, fake_input):
    batch_num = real_input.shape[0]


    disc_real = discriminator(real_input)
    disc_fake = discriminator(fake_input)
    #disc_real = discriminator.forward(real_input).unsqueeze(1).to(torch.float32)
    #disc_fake = discriminator.forward(fake_input).unsqueeze(1).to(torch.float32)

    real_label = torch.ones((batch_num, 1)).to(ml_device)
    fake_label = torch.zeros((batch_num, 1)).to(ml_device)
    
    if smooth > 0.00001:
        real_label = real_label - smooth*torch.rand(real_label.shape).to(ml_device)
    
    loss = 0.5 * (disc_loss_fn(disc_real, real_label) + disc_loss_fn(disc_fake, fake_label))
    
    return loss



def visualize_output_simple(magnitudes, correlation_matrix, epoch, writer, image_file_path):
    # 1. 첫 번째 플롯: 각 basis_states에 사영시켰을 때, 차지하는 평균 비중 시각화
    avg_magnitudes = magnitudes.mean(axis=0)
    plt.figure()
    plt.bar(range(number_of_basis), avg_magnitudes)
    plt.title(f"epoch = {epoch}")
    plt.xlabel("Index")
    plt.ylabel("Projection Magnitude")
    # save plt
    plt.savefig(f'{image_file_path}/projection_epoch{epoch:03d}.png')
    writer.add_figure(f'Projection Magnitude', plt.gcf(), epoch)
    plt.close()

    # 2. 두 번째 플롯: 각 code에 대해 basis state의 설명력 시각화
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=[f"Basis {j+1}" for j in range(number_of_basis)], 
                yticklabels=[f"Code {i+1}" for i in range(n_qubits)])
    plt.title(f"Correlation Heatmap between Code and Basis magnitudes (Epoch = {epoch})")
    plt.xlabel("Output Dimensions")
    plt.ylabel("Input Dimensions")
    plt.savefig(f'{image_file_path}/correlation_heatmap{epoch:03d}.png')
    writer.add_figure(f'Correlation Heatmap', plt.gcf(), epoch)
    plt.close()

current_time = datetime.now().strftime("%b%d_%H_%M_%S")  # "Aug13_14_12_30" 형식
save_dir = f"./runs/pure_{train_type}_{number_of_basis}_{n_qubits}_{current_time}"
scalar_save_path = os.path.join(save_dir, f"pure_{train_type}_{number_of_basis}_{n_qubits}_{current_time}.csv")
image_save_dir = os.path.join(save_dir, "images")
param_save_dir = os.path.join(save_dir, "params")
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(param_save_dir, exist_ok=True)
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
convert_py_to_html('pure_state.py', os.path.join(save_dir, 'pure_state.html'))
# ==========================================================

# save ARGS in save_dir/args.txt
with open(os.path.join(save_dir, 'args.txt'), 'w') as f:
    json.dump(ARGS.__dict__, f, indent=4)
    print(f"args 객체가 {save_dir}/args.txt 파일에 저장되었습니다.")

writer = SummaryWriter(log_dir=save_dir)

# CSV 파일 초기화 (헤더 작성)
df = pd.DataFrame(columns=['epoch', 'D_loss', 'G_loss', 'MI', 'Rsum', 'time'] + 
                  [f'c{i}-b{j}' for i in range(code_qubits) for j in range(number_of_basis)] + [f'R{i}' for i in range(number_of_basis)])

start_time = time.time()

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

    gen_states = [] # (데이터수, 2**output_qubits) 출력 statevectors
    gen_seeds = [] # (데이터수, n_qubits)
    
    coeff = COEFF # 나중에 epoch별로 coeff다르게 할 때를 위한 코드

    for batch_idx, (batch,) in enumerate(pbar):  # batch unpack
        # train generator
        generator_seed = torch.empty((BATCH_SIZE, n_qubits)).uniform_(-SEED, SEED).to(ml_device) # 실제 범위 = +-SEED * np.pi/2.
        generator_probs, generator_state, generator_loss = generator_train_step(generator_seed, coeff, use_mine=use_mine)
        G_opt.zero_grad()
        generator_loss.requires_grad_(True)
        generator_loss.backward()
        G_opt.step()
        # train discriminator
        fake_input = generator_state.detach().to(data_type)
        disc_loss = disc_cost_fn(batch, fake_input)
        D_opt.zero_grad()
        disc_loss.requires_grad_(True)
        disc_loss.backward()
        D_opt.step()
        # train MINE
        generator_probs = generator_probs.detach().to(torch.float32)
        code_input = generator_seed[:, :code_qubits] # (BATCH_SIZE, code_qubits) 코드만 추출
        pred_xy = mine(code_input, generator_probs)
        code_input_shuffle = code_input[torch.randperm(BATCH_SIZE)]
        pred_x_y = mine(code_input_shuffle, generator_probs)
        mi = -torch.mean(pred_xy) + torch.log(torch.mean(torch.exp(pred_x_y)))
        M_opt.zero_grad()
        mi.requires_grad_(True)
        mi.backward()
        M_opt.step()

        D_loss_sum += disc_loss.item()
        G_loss_sum += generator_loss.item()
        mi_sum -= mi.item() # (-1)곱해져 있어서 빼야함.

        gen_states.append(fake_input.detach().numpy())
        gen_seeds.append(generator_seed.detach().numpy())

        pbar.set_postfix({'G_loss': G_loss_sum/(batch_idx+1), 'D_loss': D_loss_sum/(batch_idx+1), 'MI': mi_sum/(batch_idx+1)})

    G_scheduler.step()
    D_scheduler.step()
    M_scheduler.step()

    gen_states = np.concatenate(gen_states, axis=0) # (train_num, 2**n_qubits)
    gen_seeds = np.concatenate(gen_seeds, axis=0) # (train_num, code_qubits)
    D_loss, G_loss, mi = D_loss_sum/batch_num, G_loss_sum/batch_num, mi_sum/batch_num
    
    magnitudes = np.abs(gen_states.conjugate() @ basis_states.T)**2 # 각 output 별로 basis state가 차지하는 비중
    correlation_matrix = np.zeros((n_qubits, number_of_basis))      # 각 code와 basis state간의 상관관계
    for i in range(n_qubits):
        for j in range(number_of_basis):
            correlation_matrix[i, j] = np.corrcoef(gen_seeds[:, i], magnitudes[:, j])[0, 1]

    # calculation all angle between correlation_matrix[i, :] and correlation_matrix[j, :]
    code_angle_sum = 0
    noise_angle_sum = 0
    cross_angle_sum = 0
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            cos_theta = np.dot(correlation_matrix[i, :], correlation_matrix[j, :]) / (np.linalg.norm(correlation_matrix[i, :]) * np.linalg.norm(correlation_matrix[j, :]))
            theta_degrees = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
            theta_degrees = min(theta_degrees, 180 - theta_degrees) # 예각으로 변환
            writer.add_scalar(f'Angle/angle{i}-{j}', theta_degrees, epoch)
            if j < code_qubits:
                code_angle_sum += theta_degrees
            elif i >= code_qubits:
                noise_angle_sum += theta_degrees
            else:
                cross_angle_sum += theta_degrees
    code_angle_avg = code_angle_sum / (code_qubits * (code_qubits - 1) / 2)
    noise_angle_avg = noise_angle_sum / ((n_qubits - code_qubits) * (n_qubits - code_qubits - 1) / 2)
    cross_angle_avg = cross_angle_sum / (code_qubits * (n_qubits - code_qubits))
    writer.add_scalar('Angle/Code_Angle', code_angle_avg, epoch)
    writer.add_scalar('Angle/Noise_Angle', noise_angle_avg, epoch)
    writer.add_scalar('Angle/Cross_Angle', cross_angle_avg, epoch)

    writer.add_scalar('Loss/d_loss', D_loss, epoch)
    writer.add_scalar('Loss/g_loss', G_loss, epoch)
    writer.add_scalar('Metrics/mi', mi, epoch)
    for i in range(number_of_basis):
        writer.add_scalar(f'Metrics/R{i}', magnitudes.mean(axis=0)[i], epoch)
    writer.add_scalar('Metrics/Rsum', magnitudes.mean(axis=0).sum(), epoch)

    for i in range(n_qubits):
        for j in range(number_of_basis):
            writer.add_scalar(f'Correlation/c{i}-b{j}', correlation_matrix[i, j], epoch)

    # 스칼라 값 CSV로 덮어쓰기 저장
    file_exists = os.path.isfile(scalar_save_path)
    new_data = pd.DataFrame({
        'epoch': [epoch],
        'D_loss': [D_loss],
        'G_loss': [G_loss],
        'MI': [mi],
        'Rsum': [magnitudes.mean(axis=0).sum()],
        'time': [int((time.time() - start_time)*1000)],
        **{f'c{i}-b{j}': correlation_matrix[i, j] for i in range(n_qubits) for j in range(number_of_basis)},
        **{f'R{j}': magnitudes.mean(axis=0)[j] for j in range(number_of_basis)}
    })

    new_data.to_csv(scalar_save_path, mode='a', header=not file_exists)
    
    visualize_output_simple(magnitudes, correlation_matrix, epoch, writer, image_save_dir)
    torch.save(generator.params, f'{param_save_dir}/generator_params_epoch{epoch}.pth')
