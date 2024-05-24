# 某大一天文萌新的IDM课题
重点复现IDM理论的第一个模型

## 关于此文件夹的说明
1. 独立的文件包括原始论文3篇、总画图notebook与mma测试代码
>mma经过测试速度过慢故抛弃
2. 下设不同方法的文件夹

### 关于CMB文件夹
1. 理论参考文献已在文件夹中,理论参考书籍为
>《Cosmology》 S.Weinberg
《Modern Cosmology 2nd》 S.Dodelson & F.Schmidt 
2. 数据来源Planck 2018(PR3) TT,TE,EE & Low-E
3. 正在学习ing

### 关于OHD文件夹
1. 数据来源见参考文献,数据文件为.csv
2. chi2.py用于复现论文图像 -> chi2_1 & chi2_2
3. chi2-H0.py用于搜索不同H0下的chi2最小值 -> Figure_3
4. test.py实现MCMC -> Figure_1 & Figure_2

### 关于SNe Ia文件夹
1. 数据来源Pantheon+,数据文件为.txt & .csv
2. 其余文件结构与OHD文件夹类似
