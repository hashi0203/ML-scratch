# Locality Preserving Projection

ガウシアン類似度行列に対する局所性保存射影を実装した．<br>
行列の各要素は具体的には以下のようになっている．

<img src="https://render.githubusercontent.com/render/math?math=W_{i, j} = \exp{(- \|\boldsymbol{x}_i - \boldsymbol{x}_{j} \|^2)}">

青の best-1 が最適な射影を表す．
- data1: 1つしかクラスタなく，横長のクラスタになっているので横向きに分割されている．
- data2: 少し微妙だが，2つのクラスタをある程度認識して斜めの直線が描かれている．
- data3: クラスタをしっかりと認識して，縦方向の直線が描かれている．

<img src="output-data1.png" alt="data1 output" width="32%"> <img src="output-data2.png" alt="data2 output" width="32%"> <img src="output-data3.png" alt="data3 output" width="32%">