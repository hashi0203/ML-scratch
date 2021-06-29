# Multi-Layer Perceptron (多層パーセプトロン)

MNIST データセットに対する多層パーセプトロンを実装した．

構造
- 3層ニューラルネットワーク
- 隠れ層のニューロンはそれぞれ 1024，512
- 活性化関数は ReLU
- 活性化関数の後に dropout を挿入

左図は loss の変化，右図は accuracy の変化を表している．

<img src="loss.png" alt="loss" width="49%"> <img src="accuracy.png" alt="accuracy" width="49%">