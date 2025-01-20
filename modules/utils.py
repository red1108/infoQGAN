import nbformat
from nbconvert import HTMLExporter
import os
import numpy as np
import torch

def convert_ipynb_to_html(ipynb_file_path, output_html_path):
    # ipynb 파일 로드
    with open(ipynb_file_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)
    
    # HTML 변환기 초기화
    html_exporter = HTMLExporter()
    
    # 변환
    (body, resources) = html_exporter.from_notebook_node(notebook_content)
    
    # HTML 파일 저장
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(body)
    
    print(f"HTML 파일이 {output_html_path}에 저장되었습니다.")


def generate_orthonormal_states(dim, m, mode="complex"):
    assert dim >= m, "dim >= number of orthogonal states"
    assert mode in ["real", "complex"], "mode should be 'real' or 'complex'"
    states = []
    max_attempts = 1000 * m  # 무한 루프 방지를 위해 충분히 큰 반복횟수 제한
    attempts = 0

    while len(states) < m and attempts < max_attempts:
        attempts += 1
        # 랜덤 복소 상태 벡터 생성
        if mode == "real":
            state = np.random.normal(size=(dim,))
        elif mode == "complex":
            state = np.random.normal(size=(dim,)) + 1j * np.random.normal(size=(dim,))

        # 이전에 선택된 상태들과 직교화 (Gram-Schmidt)
        for prev_state in states:
            overlap = np.vdot(prev_state, state)  # 복소 내적(전치 켤레)
            state = state - prev_state * overlap

        # 노름 확인 후 정규화
        norm = np.linalg.norm(state)
        if norm > 1e-10:
            state = state / norm
            states.append(state)

    if len(states) < m:
        raise ValueError("Could not generate enough orthogonal states within attempt limit.")

    # (m, dim) 형태의 NumPy 배열로 변환
    states = np.array(states)
    return states

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


def odd_intervals_seed(batch_size, n_qubits, n, seed, independent = True):
    d = 2*seed / (2*n - 1)                # 각 구간의 길이
    if independent:
        k = torch.randint(n, (batch_size, n_qubits))
    else:
        k = torch.randint(n, (batch_size,))   # batch마다 공통으로 쓸 k 값 한 개씩
        k = k[:, None].expand(-1, n_qubits)   # n_qubits만큼 브로드캐스트
    return -seed + 2*k*d + d*torch.rand(batch_size, n_qubits)

from scipy.linalg import sqrtm

def calculate_frechet_distance(gen_outputs, dataset):
    # gen_outputs: (_, 2**output_qubits), val_dataset: (_, 2**output_qubits)
    # 평균과 공분산 계산
    mu1, sigma1 = gen_outputs.mean(axis=0), np.cov(gen_outputs, rowvar=False)
    mu2, sigma2 = dataset.mean(axis=0), np.cov(dataset, rowvar=False)

    # Frechet Distance 계산
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):  # 실수 부분만 사용
        covmean = covmean.real

    frechet_distance = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return frechet_distance