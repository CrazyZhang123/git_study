![](https://picx.zhimg.com/v2-775be95caff9cf018b0fbb2ee06bfd0b_1440w.jpg)

[互信息](https://zhida.zhihu.com/search?content_id=241989631&content_type=Article&match_order=1&q=%E4%BA%92%E4%BF%A1%E6%81%AF&zhida_source=entity)(Mutual Information)是一种用于衡量两个变量之间相关性的指标，它能够量化通过一个变量获取的关于另一个变量的信息量。其优势在于能够捕捉到线性和非线性关系，远超传统相关性度量。

在数据科学中，互信息的应用十分广泛。其中，[特征选择](https://zhida.zhihu.com/search?content_id=241989631&content_type=Article&match_order=1&q=%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9&zhida_source=entity)是一个重要领域，通过计算特征与目标变量之间的互信息，可以识别出最相关的特征，提升模型性能和可解释性，同时简化模型并降低计算成本。

在网络安全方面，互信息被用于[异常检测](https://zhida.zhihu.com/search?content_id=241989631&content_type=Article&match_order=1&q=%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B&zhida_source=entity)。通过分析不同网络流量模式之间的信息增益，可以准确地发现异常活动，从而保护数字基础设施。此外，互信息还在[基因学](https://zhida.zhihu.com/search?content_id=241989631&content_type=Article&match_order=1&q=%E5%9F%BA%E5%9B%A0%E5%AD%A6&zhida_source=entity)领域发挥着重要作用，帮助揭示基因之间的复杂依赖关系，从而促进了遗传疾病的理解和有针对性的治疗方法的开发。

互信息不仅是一个理论概念，而且是一个在数据科学、网络安全和基因学等多个领域具有广泛应用的实用工具。

**互信息(Mutual Information)的本质和定义**
---------------------------------

在其核心，互信息衡量了一个随机变量提供给另一个随机变量的信息增益。在数据科学项目中，揭示变量之间的依赖关系至关重要，而这个概念是其基础。它超越了相关性的局限，捕捉到了所有形式的关系，使其在分析复杂系统时不可或缺。

对于两个随机变量，互信息(Mutual Information，简称：MI)是一个随机变量由于已知另一个随机变量而减少的“信息量”（单位通常为比特）。互信息的概念与随机变量的熵紧密相关，熵是[信息论](https://zhida.zhihu.com/search?content_id=241989631&content_type=Article&match_order=1&q=%E4%BF%A1%E6%81%AF%E8%AE%BA&zhida_source=entity)中的基本概念，它量化的是随机变量中所包含的“信息量”。

MI不仅仅是度量实值随机变量和线性相关性(如相关系数)，它更为通用。MI决定了随机变量(X,Y)的联合分布与X和Y的边缘分布的乘积之间的差异。MI是点互信息（Pointwise Mutual Information，PMI）的期望。[克劳德·香农](https://zhida.zhihu.com/search?content_id=241989631&content_type=Article&match_order=1&q=%E5%85%8B%E5%8A%B3%E5%BE%B7%C2%B7%E9%A6%99%E5%86%9C&zhida_source=entity)在他的论文A Mathematical Theory of Communication中定义并分析了这个度量，但是当时他并没有将其称为“互信息”。这个词后来由[罗伯特·法诺](https://zhida.zhihu.com/search?content_id=241989631&content_type=Article&match_order=1&q=%E7%BD%97%E4%BC%AF%E7%89%B9%C2%B7%E6%B3%95%E8%AF%BA&zhida_source=entity)创造。互信息也称为信息增益。

离散随机变量 X 和 Y 的互信息可以计算为：

![](https://pic1.zhimg.com/v2-aa6cb53fcfa5923fb2c3230ceb365460_1440w.jpg)

其中 p(x, y) 是 X 和 Y 的联合概率质量函数，而 p(x)和 p(y)分别是 X 和 Y 的边缘概率质量函数。

在连续随机变量的情形下，求和被替换成了二重定积分：

![](https://pic2.zhimg.com/v2-88cf44daba4442536c9c61ddcae0c0c7_1440w.jpg)

其中 p(x, y) 是 X 和 Y 的联合概率质量函数，而 p(x)和 p(y)分别是 X 和 Y 的边缘概率质量函数。

如果对数以 2 为基底，互信息的单位是bit。

**互信息(Mutual Information)与相关系数**
--------------------------------

我们来看下面的示例来说明互信息是一种广义的相关性度量：互信息对于相同的变量应该很高，正如第一个面板所示。对于即使在x和y之间没有直接相关性的变量，但当x的知识高度指示y时，互信息也应该很高。在第二个面板的说明中，数据被分箱，以便确定x所在的箱可以确定y将落入哪个箱。这说明了互信息作为一种广义的相关性度量。为了更好的比较，所有互信息量都被归一化为\[0,1\]。

```text
import numpy as np
from sklearn.feature_selection import mutual_info_regression,r_regression
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import normalized_mutual_info_score


def sub_plt(ind, data, mi,corr):
    plt.subplot(ind)
    plt.plot(data[0], data[1], ".", markersize=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title("Corr:{}  MI:{}".format(np.round(corr[0], 3),np.round(mi, 3)), fontsize=13)


plt.figure(figsize=[10, 3])

# identical variables
x = np.arange(0.001, 1, 1 / 1000)
y = np.flipud(x)
nmi=normalized_mutual_info_score(np.round(x,1),np.round(y,1))
x=x.reshape(-1, 1)
corr=r_regression(x,y)
sub_plt(141, [x, y],nmi,corr)

# shuffled identical variables
inds = np.arange(0.001, 1, 0.1)
x = [np.random.normal(loc=id, scale=0.015, size=[30]) for id in inds]
x = np.asarray(x).flatten()
s_inds = np.random.permutation(inds)
y = [np.random.normal(loc=id, scale=0.015, size=[30]) for id in s_inds]
y = np.asarray(y).flatten()
nmi=normalized_mutual_info_score(np.round(x,3),np.round(y,3))
x=x.reshape(-1, 1)
corr=r_regression(x,y)
sub_plt(142, [x, y],nmi,corr)

# noisy identical variables
x = np.random.rand(1000)
y = x + (np.random.randn(1000) * 0.02)
nmi=normalized_mutual_info_score(np.round(x,1),np.round(y,1))
x=x.reshape(-1, 1)
corr=r_regression(x,y)
sub_plt(143, [x, y],nmi,corr)

# random variables
x = np.random.rand(1000)
y = np.random.rand(1000)
nmi=normalized_mutual_info_score(np.round(x,1),np.round(y,1))
x=x.reshape(-1, 1)
corr=r_regression(x,y)
sub_plt(144, [x, y],nmi,corr)

plt.tight_layout()
plt.show()
```

![](https://picx.zhimg.com/v2-5e42b7c21f948175f5d46231ee7d24eb_1440w.jpg)

**互信息(Mutual Information)的计算**
------------------------------

互信息(Mutual Information)的计算我们以sklearn几个函数做简单示例一下。

1.  `mutual_info_score` 函数用于计算两个离散随机变量之间的互信息。它可以用于衡量两个离散随机变量之间的相关性或依赖性，其值的范围通常在0到正无穷大之间。在实际应用中，它常用于特征选择、聚类评估和分类模型的评估等任务。
2.  `normalized_mutual_info_score` 函数也用于计算两个离散随机变量之间的互信息，但它还进行了归一化处理，将互信息值归一化到\[0,1\]范围内。这个归一化的版本更适用于比较不同数据集或模型之间的互信息，因为它消除了数据集大小和熵的影响，使得互信息的值更具有可比性。

```text
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score

a = ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
x = ['X', 'X', 'X', 'Y', 'Z', 'Z', 'Y', 'Y', 'Z', 'Z']

mi = mutual_info_score(a,x)
print('mutual_info_score:',mi)

nmi = normalized_mutual_info_score(a,x)
print('normalized_mutual_info_score:',nmi)
```

![](https://pic3.zhimg.com/v2-9014c2b84bab5bf158c7c3a33982da9a_1440w.jpg)

`mutual_info_classif` 和 `mutual_info_regression` 是 Scikit-learn 中用于特征选择的函数，它们分别用于分类和回归任务。

1.  `mutual_info_classif`：这个函数用于分类任务。它可以计算特征与目标变量之间的互信息，用于评估特征与分类标签之间的相关性或依赖性。它适用于离散的分类问题，例如文本分类、图像分类等。
2.  `mutual_info_regression`：与 `mutual_info_classif` 相对应，这个函数用于回归任务。它可以计算特征与目标变量之间的互信息，用于评估特征与回归目标之间的相关性或依赖性。它适用于连续的回归问题，例如房价预测、股票价格预测等。

这两个函数都可以用于特征选择，通过计算特征与目标变量之间的互信息来确定哪些特征对目标变量的预测或分类最有用。在特征选择过程中，通常会选择互信息较高的特征作为最终的特征集。

```text
from sklearn.datasets import make_regression
from sklearn.feature_selection import mutual_info_regression
X, y = make_regression(
    n_samples=50, n_features=3, n_informative=1, noise=1e-4, random_state=42
)
mutual_info_regression(X, y)
```

![](https://pic1.zhimg.com/v2-5bb69f140ee088391e0838853d81bf3c_1440w.jpg)

```text
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif
X, y = make_classification(
    n_samples=100, n_features=4, n_informative=2, n_clusters_per_class=1,
    shuffle=False, random_state=42
)
mutual_info_classif(X, y)
```

![](https://pic3.zhimg.com/v2-1c74b925bd4d17b11e6e83cf0702f336_1440w.jpg)

  

互信息(Mutual Information)相较于相关系数的优点在于其能够捕捉更广泛的关系类型，包括线性和非线性关系。传统的相关系数受限于线性关系，而互信息则不受此限制，因此在分析复杂系统时更具有优势。互信息是一个通用且强大的工具，在数据科学、网络安全和基因学等多个领域都具有广泛的应用前景。

本文转自 <https://zhuanlan.zhihu.com/p/692365794>，如有侵权，请联系删除。