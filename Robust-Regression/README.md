# Robust Regression (ロバスト回帰)

直線モデルに対するフーバー回帰及びテューキー回帰の繰り返し最小二乗アルゴリズムを実装した．

フーバー回帰 (左図) では外れ値に引っ張られて直線の傾きが緩やかになっているが，**テューキー回帰** (右図) では **外れ値の影響を無視して残りの点をうまくたどる**ような直線が引けている.

<img src="output-huber.png" alt="huber output" title="フーバー回帰の結果" width="49%"> <img src="output-huber.png" alt="huber output" title="テューキー回帰の結果" width="49%">