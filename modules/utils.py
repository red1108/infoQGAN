import nbformat
from nbconvert import HTMLExporter
import os

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