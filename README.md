# 某大二天文萌新的IDM课题(准备paper中)
重点复现IDM理论的第一个模型

#### 2024.11.22更新
1. 原始论文3篇
2. 展示用代码在[all-in-one.ipynb](all-in-one.ipynb)中
3. 各方法的数据与参考论文分在不同文件夹中
4. 由于jupyter无法使用multiprocessing,all-in-one调用不同文件下的单独程序
5. article and pictures

### CMB与solution
1. 理论参考书籍为
>《Cosmology》 S.Weinberg  
>《Modern Cosmology 2nd》 S.Dodelson & F.Schmidt
2. 数据来源Planck 2018(PR3) TT,TE,EE & Low-E
>[camb-python](https://camb.readthedocs.io/en/latest/index.html)  
>[camb-github](https://github.com/cmbant/CAMB)
3. 方程的是刚性的,刚性的点与$\log kC_1$的取值有关,并在z较大时取到
>该刚性方程无法用mma与matlab的刚性求解器求解
4. 由于刚性的存在,$r_{*},r_d$这类需要到0的积分无法给出--BAO与CMB

### BAO
1. 数据来源DESI 2024 & SDSS-IV,[数据文件](/BAO/BAO.csv)
2. 参考文献[DESI 2024 VI](https://arxiv.org/abs/2404.03002)

### OHD
1. 数据来源见article,[数据文件](/OHD/OHD.csv)
2. 数据只采用CC方法

### SNe Ia
1. Pantheon Plus data from https://github.com/PantheonPlusSH0ES
2. Pantheon Plus use the $\mu$ (maybe wrong)

### QSO
1. 验证$\log kC_1$的上限与数据集的z上限有关(方程刚性点)

### multimethods
1. 利用不同方法的combination进行联合限制