
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

由於我們的目標是將資料進行二元分類，可以假設年收入大於50(y=1)和年收入小於50(y=0)各為106維的常態分配，且每個特徵是獨立的，其中變異數矩陣共用，最後藉由最大估計法直接計算參數<img src="https://latex.codecogs.com/gif.latex?\mu&space;_{1},&space;\mu&space;_{2},&space;\Sigma" title="\mu _{1}, \mu _{2}, \Sigma" />的最佳解。

擁有了模型的參數，我們藉由機率的方式來決定資料是屬於哪個類別，也就是說，分別計算資料來自於第一類的機率<img src="https://latex.codecogs.com/gif.latex?P(C_{1})" title="P(C_{1})" />和第二類的機率<img src="https://latex.codecogs.com/gif.latex?P\left&space;(C_{2}&space;\right&space;)" title="P\left (C_{2} \right )" />以及資料在第一類的機率<img src="https://latex.codecogs.com/gif.latex?P(x\mid&space;C_{1})" title="P(x\mid C_{1})" />和第二類的機率<img src="https://latex.codecogs.com/gif.latex?P(x\mid&space;C_{2})" title="P(x\mid C_{2})" />，最後藉由上述這些機率去計算資料屬於第一類的機率<img src="https://latex.codecogs.com/gif.latex?P(x\mid&space;C_{1})=&space;\frac{P(x\mid&space;C_{1})P(C_{1})}{P(x\mid&space;C_{1})P(C_{1})&plus;P(x\mid&space;C_{2})P(C_{2})}" title="P(x\mid C_{1})= \frac{P(x\mid C_{1})P(C_{1})}{P(x\mid C_{1})P(C_{1})+P(x\mid C_{2})P(C_{2})}" />和第二類的機率<img src="https://latex.codecogs.com/gif.latex?1-P(x\mid&space;C_{1})" title="1-P(x\mid C_{1})" />，最後藉此機率決定資料類別。

在此作業我們假設資料來自於常態分配，主要的原因還是因為數學推導相對而言比較簡單加上常態分配相對而言比較直觀，當然要假設其他機率分配也是可行的，例如像是0和1的類別資料，假設百努力分配相對於常態分配就會比較合理，另外假設每個特徵是獨立的也就是使用Naive Bayes Classifier。

在這case底下我們的預測精準度大約76%，相對於discriminative model的Logistic Regression略差一些。另外我們做了很多的假設，像是資料來自於兩個常態分配且變異數矩陣使用相同的參數，以及特徵之間是獨立，但可能這些資料並不符合這些假設，這也是這個模型的預測率相對於Logistic Regression差的原因。

## Reference

* [原始課程作業說明](https://docs.google.com/presentation/d/12wP13zwBWSmmYq4DufsxiMjmXociERW7VnjPWscXZO8/edit#slide=id.g1ef9a0916d_0_0)
