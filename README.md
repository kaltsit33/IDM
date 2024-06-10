# 某大一天文萌新的IDM课题
重点复现IDM理论的第一个模型

#### 2024.6.8更新
## 关于此文件夹的说明
1. 独立的文件包括原始论文3篇、总画图[notebook](test.ipynb)与[mma测试代码](test.nb)
>mma用于复现CMB中weinberg给出的近似TT曲线
2. 下设不同方法的文件夹

### 关于CMB文件夹
1. 理论参考书籍为
>《Cosmology》 S.Weinberg  
>《Modern Cosmology 2nd》 S.Dodelson & F.Schmidt 
2. 数据来源Planck 2018(PR3) TT,TE,EE & Low-E
3. 修改CAMB源码(暂时失败)
4. 正在将CAMB源码(f90)重构为python
>[camb-python](https://camb.readthedocs.io/en/latest/index.html)  
>[camb-github](https://github.com/cmbant/CAMB)

### 关于OHD文件夹
1. 数据来源见参考文献,数据文件为[.csv](/OHD/OHD.csv)
2. [test.py](/OHD/test.py)测试能否限制H0 -> [test.png](/OHD/test.png)
3. [result.py](/OHD/result.py)完成全部内容的实现,包括mcmc与传统的格点法(格点法用于精细化作图)

### 关于SNe Ia文件夹
1. 数据来源Pantheon+,数据文件为[.txt](/SNe%20Ia/Pantheon.txt) & [.csv](/SNe%20Ia/Pantheon.csv)
2. 其余文件结构与OHD文件夹类似
3. 注意:由于chi2的特殊性,理论上不可限制H0,[test.py](/SNe%20Ia/test.py)也验证了这一情况
