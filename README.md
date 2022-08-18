# 예보지원 중간보고 2022: 샘플 코드 및 데이터

### 데이터 접근 
COMS 위성의 전처리된 영상(375 X 300, 흑백)은 공개되어 있습니다. **다만, 본 샘플코드를 실행하기 위해 다운로드 받을 필요는 없습니다.**

먼저, [data_list.pkl](data_list.pkl)를 로드하여 존재하는 파일들을 확인할 수 있습니다.
```python
import pickle as pkl

with open('data_list.pkl', 'rb') as f:
  data = pkl.load(f)

e.g.,
print(data[10])
# '201103/03/coms_mi_le1b_ir01_ea040ps_201103030800.png'
```

다음으로, 서버에 아래와 같은 URL 주소로 영상을 다운로드 받을 수 있습니다.
```
http://dmlab.kaist.ac.kr/~geonlee/1300_1500/[파일명]

e.g., 
http://dmlab.kaist.ac.kr/~geonlee/1300_1500/201103/03/coms_mi_le1b_ir01_ea040ps_201103030800.png
```


### 임베딩 다운로드 (대용량 파일)
모델 학습을 통해 얻은 임베딩은 다음 공유 링크로부터 다운로드 받은 후, 본 디렉토리에 업로드하세요.

* [emb_metric_6_noaug.pkl](https://drive.google.com/file/d/14r5mO_-TnPenn2-ISKlVbQhu4cb8vbBU/view?usp=sharing)
* [contrastive_model_embeddings.pt](https://drive.google.com/file/d/1ZExk6cW1lTTo8mDt2k0ExDNdgsaLm-Ch/view?usp=sharing)
* [autoencoder_embeddings.pkl](https://drive.google.com/file/d/1DgLWQu8qomvtCXwAMl7IRh2kyGXblkcx/view?usp=sharing)

### 샘플코드
다음 샘플 코드를 통해 모델을 검증 및 평가할 수 있습니다.

* [augmentation.ipynb](augmentation.ipynb): 이미지 증강 방법
* [image_search.ipynb](image_search.ipynb): 학습한 모델을 활용한 빠른 유사사례 검색
* [anomaly_detection.ipynb](anomaly_detection.ipynb): 이상 영상 탐지
