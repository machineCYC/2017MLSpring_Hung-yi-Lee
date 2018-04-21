# Homework 4


## Purpose: Text Sentiment Classification

本次作業為 twitter 上收集到的推文，每則推文都會被標注為正面或負面。1：正面、0：負面。希望利用 training dataset 訓練一個 RNN 的 model，來預測每個句子所帶有的情緒。


## Data 簡介
 
* training dataset 分為兩個部分。
    * training label data 含有20萬個句子，以及每一句所對應到的情緒(注意：每一句都只有一種情緒)，總共有只有兩種情緒(0：負面，1:正面)。
    * training unlabel data 含有130萬句左右的句子，但並不含有 label，提供做 semi-supervised。

* testing dataset 則是20萬句句子。


## Summary



## File Stucture

```
HW4
|    README.md
|    main.py
|
└─── Base
|      __init__.py
|      DataProcessing.py
|      Model.py
|
└─── 01-RAWData
|       training_label.txt
|       training_nplabel.txt
|       testing_data.txt
|       sampleSubmission.csv
|
└─── 02-APData
|       TokenizerDictionary
|
|___
```


## Reference

* [原始課程作業說明](https://docs.google.com/presentation/d/1HnyZowEamy8N4cUT0gY4aoRZuBTluIuoe8uYQdFxhI0/edit#slide=id.p)

