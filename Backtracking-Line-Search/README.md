# Backtracking Line Search (バックトラック直線探索)

<img src="https://render.githubusercontent.com/render/math?math=f%28x_1%2Cx_2%29%3D10%28x_1%29%5E2%2B%28x_2%29%5E2%0A" alt="f(x_1,x_2)=10(x_1)^2+(x_2)^2"> に対するバックトラック直線探索を用いた勾配法の実装を行った．<br>
アルミホ基準が成り立つまでステップ幅を減衰させることで，反復回数を削減．

左がステップ幅を 0.08 に固定したときの探索結果で，右がバックトラック直線探索を行った結果である．<br>
確かに，バックトラック直線探索を行った方が収束が早くなっていることがわかる．

<img src="output-default.png" alt="default output" width="49%"> <img src="output-backtrack.png" alt="backtrack output" width="49%">
