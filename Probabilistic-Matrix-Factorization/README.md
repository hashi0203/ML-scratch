# Probabilistic Matrix Factorization (確率的行列分解)

確率的行列分解を用いた映画の推薦データの補間を実装した．

映画の推薦用データセットは [MovieLens](https://grouplens.org/datasets/movielens/) の [ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) を使用した．

行列サイズが小さい場合には，各行列の要素が0に近い値になってしまい，あまりうまく補完することができなかった．<br>
縦や横に同じ行列を並べて行列サイズを大きくして実験を行ったところ，行列サイズが小さい場合よりはよかったが，あまり精度は良くなかった．<br>
また，映画の推薦用データセットでも実験を行ったが，先程の行列よりきれいに補完されていた．<br>
初期値については，値によって収束値が異なるため，手動で幾つか試し，良さそうなものを選んだ．

入力のデータは ratings.csv を用いた．<br>
生成された推薦データなどは `output` ディレクトリの中にある．<br>
(ratings_result.csv と ratings_ans.csv はファイルサイズが大きいので zip 圧縮している)

- ratings_init.csv: データ処理後の元の行列
- ratings_result.csv: 計算結果
- ratings_ans.csv: 計算結果に元からわかっている評価を置き換えたもの
- ratings_diff.csv: `ans - result` の結果