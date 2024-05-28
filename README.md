# 某大一天文萌新的IDM课题
重点复现IDM理论的第一个模型

## 关于此文件夹的说明
1. 独立的文件包括原始论文3篇、总画图[notebook](test.ipynb)与[mma测试代码](test.nb)
>mma经过测试速度过慢故抛弃
2. 下设不同方法的文件夹

### 关于CMB文件夹
1. 理论参考文献已在文件夹中,理论参考书籍为
>《Cosmology》 S.Weinberg  
>《Modern Cosmology 2nd》 S.Dodelson & F.Schmidt 
2. 数据来源Planck 2018(PR3) TT,TE,EE & Low-E
3. 正在学习ing

### 关于OHD文件夹
1. 数据来源见参考文献,数据文件为[.csv](/OHD/OHD.csv)
2. [chi2.py](OHD/chi2.py)用于复现论文图像 -> [chi2_1](/OHD/chi2_1.png) & [chi2_2](/OHD/chi2_2.png)
3. [chi2-H0.py](/OHD/chi2-H0.py)用于搜索不同H0下的chi2最小值 -> [Figure_1](/OHD/Figure_1.png)
4. [test.py](/OHD/test.py)实现MCMC -> [Figure_2](OHD/Figure_2.png) & [Figure_3](OHD/Figure_3.png)

### 关于SNe Ia文件夹
1. 数据来源Pantheon+,数据文件为[.txt](/SNe%20Ia/Pantheon.txt) & [.csv](/SNe%20Ia/Pantheon.csv)
2. 其余文件结构与OHD文件夹类似
