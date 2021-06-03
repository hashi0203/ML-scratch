# ML-scratch

機械学習の手法のスクラッチ実装

- **Least Squares Regression** (最小二乗回帰)
    - ガウスカーネルモデルに対する l2 正則化を用いた最小二乗回帰の交差確認法
- **Sparse Regression** (スパース回帰)
    - ガウスカーネルモデルに対する交互方向乗数法を用いたスパース回帰の交差確認法
- **Robust Regression** (ロバスト回帰)
    - 直線モデルに対するフーバー回帰及びテューキー回帰の繰り返し最小二乗アルゴリズム
- **Least Squares Classification** (最小二乗分類)
    - ガウスカーネルモデルに対する一対他の最小二乗回帰を用いた手書き数字のパターン認識
    - `digit.mat` を使用
    - PC のスペックによっては**実行にかなり時間がかかる**ため注意
- **Support Vector Machine** (サポートベクトルマシン)
    - 線形モデルに対するサポートベクトルマシンの劣勾配アルゴリズム
- **Least Squares Stochastic Classification** (最小二乗確率的分類)
    - ガウスカーネルモデルに対する最小二乗確率的分類
