# **CS222 Project Proposal: Network Reformation and Fusion through CDRP Model Decomposition and Assembly**
我们参考以下这篇文章中提出的critical data routing path (CDRP)提出了一个较完整的项目流程。
```
@inproceedings{wang2018interpret,
  title={Interpret neural networks by identifying critical data routing paths},
  author={Wang, Yulong and Su, Hang and Zhang, Bo and Hu, Xiaolin},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={8906--8914},
  year={2018}
}
```
我们的主要目标是：
- 将大网络分解成多个小网络
- 各个小网络负责不同的二分类（或者多分类）子问题
- 给定目标类别集合，通过拼装小网络，组成一组复杂度足够低的网络能对该集合内的类别正确分类

如果按接下来描述的CDRP来解决前两个问题，会导致第三个问题变得trivial，因此我们考虑对多个不同任务（在同一性质不同数据集上训练的）网络进行以上操作，将其融合成一个模型。
这样一来，既保证了核心算法问题的复杂度，又使其成果有一定应用价值。

## **CDRP Model Decomposition and Clustering**
CDRP指的是在原来网络中，某一个sample输入后，每一组卷积层会有几个channel的激活响应比较高，这些卷积核组成了CDRP。
我们可以通过只保留每一层的这些channel来减小网络大小（总的层数不变，但是每一层的channel数减少了）同时保证这个sample仍然分类正确。
我们称减小了的网络维CDRP Model。

在此基础上，原论文提出可以用每个sample对应的CDRP（激活的卷积核）来给它编码，可以作为原网络下这个sample的特征表示$v$。
我们可以**对所有sample训练这个$v$**，然后进行聚类。
可以看出，属于同一个cluster的sample可以通过同一个CDRP Model进行预测并且保证一定的正确率。

同时如果我们假设原网络是一个完美的模型，那么不应该有两个class被分到了同一个cluster中，否则它们在原网络中会激活相同的CDRP，导致原本的分类结果就是错误的，这与完美模型假设冲突。
（这似乎也可以用来分析网络错误的原因）

这个聚类过程解决了我们原来提出的如何确定subclass和对应的discriminator（不需要train from scratch了）的问题。

## **Network Fusion**
但是如果只在一个网络中进行以上过程，最后得到的拼装问题只是不断添加原网络中的卷积核，直到满足要求为止。
这是一个非常trivial的问题，不需要任何复杂的算法。

但是，我们可以考虑将多个不同的网络进行decomposition和clustering再对小网络进行组装。
这里不同的网络指的是，针对同一类型task（都是classification或都是detection），在不同数据集上训练的模型。

假设网络$M$在数据集$A$上训练，网络$N$在数据集$B$上训练。
- 如果我们把$B - A$中的样本在$M$中计算CDRP并聚类，它们有可能：
  - 在特征空间上十分分散毫无规律，说明$M$无法处理这种sample；
  - 属于原来$A$在$M$上聚出来的某个cluster，说明这些sample所属类别和$A$中部分数据有相似的特征，有可能复用$N$中有关的CDRP模型；
  - 单独形成一个cluster，说明除了用$N$的某个CDRP模型，这个cluster对应的$M$中的某个CDRP模型也可以对其分类。
- 对于$A \cap B$的样本，它们本身就对应了$N$和$M$中的两份CDRP模型，有进行选择的余地。

同一个网络中的CDRP模型大部分会共用重合的卷积核，所以应该不用担心模型会过度复杂~~吧？~~。

这样一来，求解最优覆盖的算法设计需要足够general，同时得到的结果也可以有以下可能的用处：
- 将现有的在不同数据集上训练的不同功能的网络进行重构，得到可以处理与两个数据集中数据都有关的任务的模型；
- 实现fine grained network fusion，在模型的子结构上进行融合，而不是直接合并多个完整模型的输出结果；
- ~~似乎偏离了~~原来的目标：减小网络复杂度

## Problems
但是，我们现在有这么几个可能的问题：
- 原来论文里的方法是对每个sample都训练一个CDRP（卷积核的选择），我们也打算采用相同的方式来抽取用于聚类的feature，这个训练的开销有没有可能太大：
  - 单个sample的训练复杂度会不会太高
  - 对所有sample的训练会不会太低效
- 这个方法只能在每一个卷积层的channel数量上做删减，但是不能一次删掉一整个卷积层，及模型的深度应该是不变的。虽然子模型会共用一部分留下的channel，但是这么做会不会导致最终的模型（多个CDRP子模型合起来）复杂度太高
