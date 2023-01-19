- To do list
  - 벤치마크 데이터 세팅
  - 모델 벤치마크 데이터 train & inference flow 만들기
  - how to test? 다른 라이브러리 참고

# 과정

- catchMinor용 가상환경 생성 완료
  - python 3.10.4 버전
- conventional commits
  - https://www.conventionalcommits.org/ko/v1.0.0/
  - https://github.com/commitizen-tools/commitizen
  - https://blog.dnd.ac/github-commitzen-template/
- lint
  - flake8, mypy
- format
  - black, isort
- python package 생성 및 배포
  - https://devocean.sk.com/blog/techBoardDetail.do?ID=163566


# 다른 라이브러리
## pyod
- 각 모델은 하나의 .py 파일에 class로 구현되어 있다.
- 각 모델 class는 [BaseDetector](https://github.com/yzhao062/pyod/blob/31db6d153a2aea02652a2d6325d0a1f4c8a38ec8/pyod/models/base.py)라는 class를 상속한다.
- docstring은 구글것을 따른다.
- unsupervised model의 경우에도 `y=None`과 같은 식으로 진행했다.
- anomaly score와 관련하여 raw, scaled score, confidence, proba 등으로 제공한다.
- `@property` 사용
- 속도향상을 위해 numba, jit 사용
### API
- https://pyod.readthedocs.io/en/latest/pyod.html
- 각 모델들이 아래의 함수들을 갖고 있다.
  - `fit(X)`, `decision_function(X)`, `predict(X)`, `predict_proba`, `predict_confidence`
- 각 모델들의 key attributes는
  - `decision_scores_`, `labels_`


## [anomalib](https://github.com/openvinotoolkit/anomalib)
- python cli 환경에서 사용하도록 구현되어있다.
- pytorch lightning을 이용했다.
- lightning module을 만들 때 [base](https://github.com/openvinotoolkit/anomalib/blob/main/anomalib/models/components/base/anomaly_module.py) class를 만들어서 모든 모델들이 상속받았다.

## [pytorch_tabular](https://github.com/manujosephv/pytorch_tabular)
- config 아이디어!