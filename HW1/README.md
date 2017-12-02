
# Homework 1


## Purpose: Predict PM2.5

本次作業的資料是從中央氣象局網站下載的真實觀測資料，希望利用linear regression或其他方法預測PM2.5的數值。

## Data 簡介

* 本次作業使用豐原站的觀測記錄，分成train set跟test set，train set是豐原站每個月的前20天所有資料。test set則是從豐原站剩下的資料中取樣出來。

* train.csv：每個月前20天的完整資料，總共有5652筆train data，每筆資料維度為162。

* test.csv：從剩下的資料當中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。一共取出240筆不重複的test data，請根據feauure預測這240筆的PM2.5。


## Reference

* [原始課程作業說明](https://docs.google.com/presentation/d/1L1LwpKm5DxhHndiyyiZ3wJA2mKOJTQ2heKo45Me5yVg/edit#slide=id.g1eabbd760e_0_487)

* [Adagrad](https://www.youtube.com/watch?v=yKKNr-QKz2Q&feature=youtu.be&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&t=705)