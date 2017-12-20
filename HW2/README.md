
# Homework 2


## Purpose: Binary Classification

本次作業是需要從給定的個人資訊，預測此人的年收入是否大於50K。

## Data 簡介

* 本次作業使用 [ADULT dataset](https://archive.ics.uci.edu/ml/datasets/Adult)

Barry Becker從1994年的人口普查數據庫中進行了提取。 
（（AGE> 16）&&（AGI> 100）&&（AFNLWGT> 1）&&（HRSWK> 0））提取了一組合理清潔的記錄。

* 共有32561筆訓練資料，16281筆測試資料，其中資料維度為106。


## Summary 總結

本次作業執行generative model和discriminative model。

* Logistic Regression


* Probabilstic Generative Model

由於我們的目標是將資料進行二元分類，可以假設年收入大於50(y=1)和年收入小於50(y=0)各為106維的常態分配，且每個樣本是獨立，其中變異數矩陣共用，最後藉由最大估計法直接獲得參數的最佳解。

擁有了模型的參數，我們藉由機率的方式來決定資料是屬於哪個類別，也就是說，分別計算

<a href="https://www.codecogs.com/eqnedit.php?latex=x^{10}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{10}" title="x^{10}" /></a>

<img src="https://latex.codecogs.com/gif.latex?x^{10}" title="x^{10}" />

## Reference

* [原始課程作業說明](https://docs.google.com/presentation/d/12wP13zwBWSmmYq4DufsxiMjmXociERW7VnjPWscXZO8/edit#slide=id.g1ef9a0916d_0_0)
