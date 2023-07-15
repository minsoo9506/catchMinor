- To do list
  - [ ] time series feature
    - [ ] 실제 데이터로 훈련해보기 jupyter로 진행해보자, backward하는 부분에서 문제 발생 ㅠ
    - [ ] Llinear, Dlinear pred based 모델: https://today-1.tistory.com/60
  - [ ] XAI 관련 기능 구현
  - [ ] Matrix Profile 공부 및 기존 라이브러리 이용해서 구현?!
  - [ ] feature: 모델 벤치마크 데이터 train & inference flow 만들기 (for문으로 데이터 넣고 결과 뽑아내는)
  - [ ] refactor: python advanced 하게 코드 수정

- git origin 최신화 과정
```bash
git fetch upstream  # upstream에서 모든 브랜치 로컬로 다운로드
git checkout main  # merge하려는 브랜치로 이동
git merge upstream/main  # merge
git push origin main  # origin으로 push해서 최신화
```

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

# TIL
- abc를 이용하여 추상클래스 사용
  - 추상클래스는 인스턴스를 만들지 못하고 상속에만 사용
  - 추상메서드 -> 상속받는 클래스에서 메서드 구현을 강제
  - https://dojang.io/mod/page/view.php?id=2389


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


# tabular data
| Number | Data | # Samples | # Features | # Anomaly | % Anomaly | Category |
|:--:|:---:|:---------:|:----------:|:---------:|:---------:|:---:|
|1| ALOI                    |   49534   |     27     |   1508    |   3.04    |     Image     |
|2| annthyroid   |   7200    |     6      |    534    |   7.42    |      Healthcare    |
|3| backdoor|   95329   |    196     |   2329    |   2.44    | Network|
|4| breastw                              |    683    |     9      |    239    |   34.99   | Healthcare  |
|5|campaign|   41188   |     62     |   4640    |   11.27   | Finance|
|6| cardio                               |   1831    |     21     |    176    |   9.61    | Healthcare |        
|7| Cardiotocography    |   2114    |     21     |    466    |   22.04   | Healthcare         |
|8|celeba|  202599   |     39     |   4547    |   2.24    | Image|
|9|census|  299285   |    500     |   18568   |   6.20    | Sociology|
|10| cover                                |  286048   |     10     |   2747    |   0.96    | Botany    | 
|11|donors|  619326   |     10     |   36710   |   5.93    | Sociology|
|12| fault                      |   1941    |     27     |    673    |   34.67   | Physical         |
|13|fraud|  284807   |     29     |    492    |   0.17    | Finance|
|14| glass |    214    |     7      |     9     |   4.21    | Forensic          |
|15| Hepatitis           |    80     |     19     |    13     |   16.25   | Healthcare         |
|16| http                                 |  567498   |     3      |   2211    |   0.39    | Web   |      
|17| InternetAds   |   1966    |    1555    |    368    |   18.72   | Image         |
|18| Ionosphere        |    351    |     32     |    126    |   35.90   | Oryctognosy         |
|19| landsat                         |   6435    |     36     |   1333    |   20.71   | Astronautics    |     
|20| letter                               |   1600    |     32     |    100    |   6.25    | Image     |    
|21| Lymphography       |    148    |     18     |     6     |   4.05    | Healthcare       |  
|22| magic.gamma                     |   19020   |     10     |   6688    |   35.16   | Physical        | 
|23| mammography                          |   11183   |     6      |    260    |   2.32    | Healthcare  |       
|24| mnist                                |   7603    |    100     |    700    |   9.21    | Image      |   
|25| musk                                 |   3062    |    166     |    97     |   3.17    | Chemistry   |      
|26| optdigits                            |   5216    |     64     |    150    |   2.88    | Image     |    
|27| PageBlocks         |   5393    |     10     |    510    |   9.46    | Document         |
|28| pendigits                            |   6870    |     16     |    156    |   2.27    | Image        | 
|29| Pima                |    768    |     8      |    268    |   34.90   | Healthcare         |
|30| satellite                            |   6435    |     36     |   2036    |   31.64   | Astronautics     |    
|31| satimage-2                           |   5803    |     36     |    71     |   1.22    | Astronautics    |     
|32| shuttle                              |   49097   |     9      |   3511    |   7.15    | Astronautics  |       
|33| skin                            |  245057   |     3      |   50859   |   20.75   |    Image      |
|34| smtp                                 |   95156   |     3      |    30     |   0.03    | Web        | 
|35| SpamBase            |   4207    |     57     |   1679    |   39.91   | Document         |
|36| speech                               |   3686    |    400     |    61     |   1.65    | Linguistics    |     
|37| Stamps              |    340    |     9      |    31     |   9.12    | Document         |
|38| thyroid                              |   3772    |     6      |    93     |   2.47    | Healthcare      |   
|39| vertebral                            |    240    |     6      |    30     |   12.50   | Biology       |  
|40| vowels                               |   1456    |     12     |    50     |   3.43    | Linguistics  |       
|41| Waveform           |   3443    |     21     |    100    |   2.90    | Physics         |
|42| WBC                |    223    |     9      |    10     |   4.48    | Healthcare         |
|43| WDBC               |    367    |     30     |    10     |   2.72    | Healthcare         |
|44| Wilt                |   4819    |     5      |    257    |   5.33    | Botany         |
|45| wine                                 |    129    |     13     |    10     |   7.75    | Chemistry   |      
|46| WPBC             |    198    |     33     |    47     |   23.74   | Healthcare   |      
|47| yeast                           |   1484    |     8      |    507    |   34.16   | Biology|
|48| CIFAR10| 5263 |    512    |    263     |   5.00    |   Image   |
|49| FashionMNIST| 6315|    512    |    315     |   5.00    |   Image   |
|50| MNIST-C| 10000|    512    |    500     |   5.00    |   Image   |
|51| MVTec-AD| See Table B2. |       |          |       |   Image   |
|52| SVHN| 5208 |512| 260 |5.00 |Image |
|53| Agnews| 10000 |768 |500 |5.00| NLP |
|54| Amazon| 10000 |768| 500 |5.00| NLP |
|55| Imdb| 10000| 768| 500 |5.00 |NLP |
|56| Yelp| 10000| 768| 500 |5.00| NLP |
|57| 20newsgroups| See Table B3. |     |          |       |   NLP   |