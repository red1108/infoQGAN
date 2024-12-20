import nbformat
from nbconvert import HTMLExporter
import os
import numpy as np

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


def generate_orthonormal_states(dim, m):
    assert dim >= m, "dim >= number of orthogonal states"
    states = []
    max_attempts = 1000 * m  # 무한 루프 방지를 위해 충분히 큰 반복횟수 제한
    attempts = 0

    while len(states) < m and attempts < max_attempts:
        attempts += 1
        # 랜덤 복소 상태 벡터 생성
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