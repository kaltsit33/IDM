# 某大一天文萌新的IDM课题(现在大二了)
重点复现IDM理论的第一个模型

#### 2024.9.15更新
### 主文件夹
1. 原始论文3篇
>mma用于复现CMB中weinberg给出的近似TT曲线
2. 全部重改,所有主要代码放在[all-in-one.ipynb](all-in-one.ipynb)中
3. 各方法的数据与参考论文分在不同文件夹中

### BAO
1. 数据来源DESI 2024,[数据文件](/BAO/BAO.csv)
2. 参考文献[DESI 2024 VI](https://arxiv.org/abs/2404.03002)

### CMB
1. 理论参考书籍为
>《Cosmology》 S.Weinberg  
>《Modern Cosmology 2nd》 S.Dodelson & F.Schmidt
2. 数据来源Planck 2018(PR3) TT,TE,EE & Low-E
>[camb-python](https://camb.readthedocs.io/en/latest/index.html)  
>[camb-github](https://github.com/cmbant/CAMB)

### OHD
1. 数据来源见参考文献,[数据文件](/OHD/OHD.csv)
2. 数据只采用CC方法

### SNe Ia
1. Pantheon data used [数据文件](/SNe%20Ia/Pantheon.txt)
2. Pantheon Plus data from https://github.com/PantheonPlusSH0ES
3. Pantheon Plus use the $\mu$