# Homework 6

## Purpose: Unsupervised learning & dimension reduction

### PART1: PCA of colored faces

#### Data 簡介
[數據集](https://drive.google.com/file/d/1IfPN_emmgGKZVqACjNj8fymDqwsDECyJ/view?usp=drive_open)來自 Aberdeen University 的 Prof. Ian Craw，並經過挑選及對齊，總共有 415 張 600 X 600 X 3 的彩圖。

#### Summary

這次 PCA 主要是要利用少數的維度來代表全部的維度，假設 <a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X" title="X" /></a> 為 <a href="https://www.codecogs.com/eqnedit.php?latex=n\times&space;p" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n\times&space;p" title="n\times p" /></a> 的矩陣，其中 n 代表照片數量，p 代表照片維度。扣掉平均照片 <a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a> (平均臉)，可以得到中心矩陣 <a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X" title="X" /></a> 。

透過解 <a href="https://www.codecogs.com/eqnedit.php?latex=X^{T}X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X^{T}X" title="X^{T}X" /></a> 可以得到 <a href="https://www.codecogs.com/eqnedit.php?latex=p\times&space;k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p\times&space;k" title="p\times k" /></a> 的 eigenvectors matrix <a href="https://www.codecogs.com/eqnedit.php?latex=V" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V" title="V" /></a>，其中 <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a> 為我們要重建照片所使用的 eigenvectors，而通常都是前 <a href="https://www.codecogs.com/eqnedit.php?latex=k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k" title="k" /></a> 大的 eigenvalues 所對應到的 eigenvectors。則利用 <a href="https://www.codecogs.com/eqnedit.php?latex=V" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V" title="V" /></a> 可以投影到低維的 <a href="https://www.codecogs.com/eqnedit.php?latex=n\times&space;k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n\times&space;k" title="n\times k" /></a> 矩陣 <a href="https://www.codecogs.com/eqnedit.php?latex=Z=&space;XV" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z=&space;XV" title="Z= XV" /></a>。
  
另外為了可以從低維度的 <a href="https://www.codecogs.com/eqnedit.php?latex=Z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z" title="Z" /></a> 重建回 <a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X" title="X" /></a>，所以將 <a href="https://www.codecogs.com/eqnedit.php?latex=Z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Z" title="Z" /></a> 乘上 <a href="https://www.codecogs.com/eqnedit.php?latex=V^{T}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?V^{T}" title="V^{T}" /></a> 可得 <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{X}=&space;ZV^{T}=&space;XVV^{T}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{X}=&space;ZV^{T}=&space;XVV^{T}" title="\widehat{X}= ZV^{T}= XVV^{T}" /></a>。但別忘了一開始有扣掉平均照片 <a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a> 。所以 <a href="https://www.codecogs.com/eqnedit.php?latex=\widehat{X}_{reconstruction}=&space;ZV^{T}&space;&plus;&space;\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widehat{X}_{reconstruction}=&space;ZV^{T}&space;&plus;&space;\mu" title="\widehat{X}_{reconstruction}= ZV^{T} + \mu" /></a> 。



### PART2: Visualization of Chinese word embedding

#### Data 簡介



#### Summary 


### PART3: Image clustering

#### Data 簡介


#### Summary 

## Reference

* [原始課程作業說明](https://docs.google.com/presentation/d/1v2aJnjqplnQ5YSprp6IXbWM_VPavtolqpgbGWM4HidY/edit)

* [人臉識別算法-特徵臉方法（Eigenface）及 python 實現](https://blog.csdn.net/u010006643/article/details/46417127)

* [PCA](https://stats.stackexchange.com/questions/229092/how-to-reverse-pca-and-reconstruct-original-variables-from-several-principal-com)