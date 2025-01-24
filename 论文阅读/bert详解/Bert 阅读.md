---
created: 2025-01-12T14:34
updated: 2025-01-24T00:08
---
## Abstract

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al, 2018a; Radford et al, 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications.

我们提出了一种新的语言表征模型：BERT，e.g. Bidirectional Encoder Representations from Transformers(即来自 Transformers 的双向Encoder表征)。和最新提出来的语言模型不同，BERT的设计初衷是从没有标签的文本中，根据所有层左右的上下文同时获得预训练的深度双向表征。**这样，预训练的BERT模型可以通过仅仅一个额外的输出层，就可以在现在一大批任务上创建SOTA模型，比如问题回答和语言推断，而且不需要针对特定任务进行架构的修改。**

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).

 **BERT的概念非常直观而且经验显示了其强大的性能。它在11个自然语言处理任务上，取得了新的SOTA结果**，包括将GLUE分数提高到了80.5%（7.7%的绝对提高），MultiNLI精度提高到86.7%（4.6%绝对提高），SQuAD v1.1问题回答测试F1提高到了93.2（1.5的绝对提高），SQuAD v2.0测试F1提高到了83.1（5.1的绝对提高）

```ad-abstract
- BERT 
- 来自 Transformers 的双向Encoder表征
- **设计初衷**是从没有标签的文本中，根据所有层左右的上下文同时获得预训练的深度双向表征。
- 优势：**预训练的BERT模型可以通过仅仅一个额外的输出层，就可以在现在一大批任务上创建SOTA模型，比如问题回答和语言推断，而且不需要针对特定任务进行架构的修改。**
	- 概念简单，性能强大，在自然语言处理任务中取得了SOTA结果。

- 其他
- GLUE分数：[[【HugBert04】GLUE：BERT类模型的通用评估基准]]
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250112152252.png)
-  写论文的提示：
	- 基于前人的论文，改进在哪里，好在哪里
	- including pushing the GLUE score to 80.5% (7.7% point absolute improvement) 绝对精度和相对精度，从30%->37%效果肯定不如80%->87%
```

## 1 Introduction

Language model pre-training has been shown to be effective for improving many natural language processing tasks (Dai and Le, 2015; Peters et al., 2018a; Radford et al., 2018; Howard and Ruder, 2018). These include sentence-level tasks such as natural language inference (Bowman et al., 2015; Williams et al., 2018) and paraphrasing(解释，释义) (Dolan and Brockett, 2005), which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition and question answering, where models are required to produce fine-grained(细粒的) output at the token level (Tjong Kim Sang and De Meulder, 2003; Rajpurkar et al., 2016).

**语言模型预训练已被证明对提升许多自然语言处理任务是有效的**（戴和乐，2015；彼得斯等人，2018a；拉德福德等人，2018；霍华德和鲁德，2018）。**这些任务包括句子级别的任务**，如自然语言推理（鲍曼等人，2015；威廉姆斯等人，2018）和释义（多兰和布罗克特，2005），其目的是通过整体分析句子来预测句子之间的关系；还包括**Token级别的任务**，如命名实体识别和问答，在这些任务中，模型需要在Token级别产生精细的输出（宗金桑和德穆尔德，2003；拉杰普尔卡等人，2016）。

There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The feature-based approach, such as ELMo (Peters et al., 2018a), uses task-specific architectures that include the pre-trained representations as additional features. The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT) (Radford et al., 2018), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all pretrained parameters. The two approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations.

**将预训练的语言表征应用于下游任务有两种现有策略**：**基于特征的方法和微调方法。** 像 ELMo（彼得斯等人，2018a）这样的基于特征的方法，使用特定任务的架构，把预训练的表征作为额外特征。而微调方法，例如生成式预训练Transformer（OpenAI GPT）（拉德福德等人，2018），**引入极少的特定任务参数，并通过简单地对所有预训练参数进行微调来针对下游任务进行训练。** 这两种方法在预训练期间具有相同的目标函数，它们都使用单向语言模型来学习通用语言表征。

```ad-attention
- (1)**语言模型预训练已被证明对提升许多自然语言处理任务是有效的(bert让预训练方法出圈了)**
- 任务包括**句子级别的任务**，如自然语言推理和解释(目的是通过整体分析句子来预测句子之间的关系)
- **Token级任务**：命名实体识别和问答，需要模型生成细粒度(fine-grained)的输出
- (2)**将预训练的语言表征应用于下游任务有两种现有策略**：**基于特征的方法和微调方法。**
- **基于特征的方法**,使用特定任务的架构，把预训练的表征作为额外特征。如ELMo
- **微调方法**，**引入极少的特定任务参数，并通过简单地对所有预训练参数进行微调来针对下游任务进行训练。**例如生成式预训练Transformer（OpenAI GPT）
- **预训练的共同目标函数**——单向语言模型(unidirectional language models)学习通用语言特征。
```

We argue that current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training. For example, in OpenAI GPT, the authors use a left-toright architecture, where every token can only attend to previous tokens in the self-attention layers of the Transformer (Vaswani et al., 2017). Such restrictions are sub-optimal for sentence-level tasks, and could be very harmful when applying finetuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions.

我们认为，**当前的技术限制了预训练表征的能力，尤其是在微调方法方面**。**主要的局限性在于标准语言模型是单向的**，这限制了在预训练期间可以使用的架构选择。例如，在 OpenAI GPT 中，作者使用**从左到右的架构**，在这种架构下，每个token在 Transformer 的自注意力层中只能关注前面的标记（Vaswani 等人，2017）。这种限制对于句子级任务来说并非最优，并且在将基于微调的方法应用于token级任务（如问答）时可能会非常有害，**因为在问答任务中，整合来自两个方向的上下文至关重要。**

In this paper, we improve the fine-tuning based approaches by proposing BERT: Bidirectional Encoder Representations from Transformers. BERT alleviates the previously mentioned unidirectionality constraint by **using a “masked language model” (MLM)** pre-training objective, inspired by the Cloze task (Taylor, 1953). The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-toright language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pretrain a deep bidirectional Transformer. In addition to the masked language model, we also use a “next sentence prediction” task that jointly pretrains text-pair representations. The contributions of our paper are as follows: 

在本文中，我们通过提出 BERT（来自 Transformers 的双向编码器表征）改进了基于微调的方法。**BERT 受完形填空任务（泰勒，1953）的启发，采用“掩码语言模型”（MLM）预训练目标，缓解了前面提到的单向性限制。** **掩码语言模型从输入中随机掩码一些标记，目标是仅基于其上下文预测被掩码单词的原始词汇表 ID**。与从左到右的语言模型预训练不同，<mark style="background: #FF0000;">MLM 目标使表征能够融合左右上下文，这使我们能够预训练一个深度双向 Transformer。除了掩码语言模型，我们还使用“下一句预测”任务来联合预训练文本对表征。</mark>我们论文的贡献如下：

• We demonstrate the importance of bidirectional pre-training for language representations. Unlike Radford et al. (2018), which uses unidirectional language models for pre-training, BERT uses masked language models to enable pretrained deep bidirectional representations. This is also in contrast to Peters et al. (2018a), which uses a shallow concatenation of independently trained left-to-right and right-to-left LMs. . 
• We show that pre-trained representations reduce the need for many heavily-engineered taskspecific architectures. BERT is the first finetuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many task-specific architectures. 
• BERT advances the state of the art for eleven NLP tasks. The code and pre-trained models are available at https://github.com/ google-research/bert.

- 我们证明了双向预训练对语言表征的重要性。与拉德福德等人（2018）使用单向语言模型进行预训练不同，BERT 使用掩码语言模型来实现预训练的深度双向表征。这也与彼得斯等人（2018a）形成对比，后者使用独立训练的从左到右和从右到左语言模型的浅层连接。 
- 我们表明预训练的表征减少了对许多经过大量工程设计的特定任务架构的需求。BERT 是第一个基于微调的表征模型，在大量句子级和标记级任务上达到了最先进的性能，优于许多特定任务架构。 
- BERT 在 11 个自然语言处理任务上推动了技术的发展。代码和预训练模型可在 https://github.com/google-research/bert 上获取。

```ad-attention
- (1) **当前的技术限制了预训练表征的能力，尤其是在微调方法方面**。
	- **主要的局限性在于标准语言模型是单向的**
	- 每个token只能关注前面的token,**对句子级的任务不好，也对token级的任务很有害(更需要整合两个的方向的上下文)**
- (2) Bert**改进了微调**的方法，受**完形填空任务(Cloze task)启发**，采用**掩码语言模型(masked language model)MLM预训练目标**，**缓解了单向限制性**。
	- **掩码语言模型**从输入中随机掩码一些token记，**目标是仅基于其上下文预测被掩码单词的原始词汇 ID。**
	- MLM 目标使表征能够融合左右上下文,因此预训练出了一个深度双向 Transformer
- (3) 论文其他贡献：
	1. **证明了双向预训练对语言表征的重要性**，
		- BERT 使用掩码语言模型来实现预训练的深度双向表征。
	2. **表明预训练的表征减少了对许多经过大量工程设计的特定任务架构的需求**。
	3. BERT 在 11 个自然语言处理任务上推动了技术的发展。
```


## 2 Related Work

There is a long history of pre-training general language representations, and we briefly review the most widely-used approaches in this section.

预训练通用语言表征有着悠久的历史，在本节中我们简要回顾最常用的方法。

### 2.1 Unsupervised Feature-based Approaches
2.1 基于无监督特征的方法 

Learning widely applicable representations of words has been an active area of research for decades, including non-neural (Brown et al., 1992; Ando and Zhang, 2005; Blitzer et al., 2006) and neural (Mikolov et al., 2013; Pennington et al., 2014) methods. Pre-trained word embeddings are an integral part of modern NLP systems, offering significant improvements over embeddings learned from scratch (Turian et al., 2010). To pretrain word embedding vectors, left-to-right language modeling objectives have been used (Mnih and Hinton, 2009), as well as objectives to discriminate correct from incorrect words in left and right context (Mikolov et al., 2013).

几十年来，**学习可广泛应用的单词表示方法一直是一个活跃的研究领域，包括非神经网络方法**（[Brown 等人，1992](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xbrown-etal:1992:_class)；[Ando 和 Zhang，2005](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xando-zhang:2005)；[Blitzer 等人，2006](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xblitzer-mcdonald-pereira:2006:_domain)）和**神经网络方法**（[Mikolov 等人，2013](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xmikolov-etal:2013)； [Pennington 等人，2014](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xpennington-socher-manning:2014:_glove)）。**预训练的词嵌入是现代 NLP 系统不可或缺的一部分，与从零开始学习的嵌入相比有显着改进**（[Turian 等人，2010](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xturian-ratinov-bengio:2010:_word_repres)）。 为了**预训练词嵌入向量**，已使用从左到右的语言建模目标（[Mnih 和 Hinton，2009](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xminh09)），以及在左右上下文中区分正确单词和错误单词的目标（[Mikolov 等人，2013](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xmikolov-etal:2013)）。 

These approaches have been generalized to coarser granularities, such as sentence embeddings (Kiros et al., 2015; Logeswaran and Lee, 2018) or paragraph embeddings (Le and Mikolov, 2014). To train sentence representations, prior work has used objectives to rank candidate next sentences (Jernite et al., 2017; Logeswaran and Lee, 2018), left-to-right generation of next sentence words given a representation of the previous sentence (Kiros et al., 2015), or denoising autoencoder derived objectives (Hill et al., 2016).

**这些方法已经推广到更粗的粒度，如句子嵌入**（基罗斯等人，2015；洛格斯瓦兰和李，2018）**或段落嵌入**（乐和米科洛夫，2014）。为了**训练句子表示**，以前的工作使用对候选的下一句进行评分排序（[Jernite 等人，2017](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#XDBLP:journals/corr/JerniteBS17)；[Logeswaran 和 Lee，2018](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xlogeswaran2018an)）的目标，给定前一个句子的表示从左到右生成下一个句子的单词（[Kiros 等人，2015](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xkiros-etal:2015:_skip)）或去噪自动编码器派生的目标（[Hill 等人，2016](https://www.yiyibooks.cn/__trs__/nlp/bert/main.html#Xhill16)）。

ELMo and its predecessor (Peters et al., 2017, 2018a) generalize traditional word embedding research along a different dimension. They extract context-sensitive features from a left-to-right and a right-to-left language model. The contextual representation of each token is the concatenation of the left-to-right and right-to-left representations. When integrating contextual word embeddings with existing task-specific architectures, ELMo advances the state of the art for several major NLP benchmarks (Peters et al., 2018a) including question answering (Rajpurkar et al., 2016), sentiment analysis (Socher et al., 2013), and named entity recognition (Tjong Kim Sang and De Meulder, 2003). Melamud et al. (2016) proposed learning contextual representations through a task to predict a single word from both left and right context using LSTMs. Similar to ELMo, their model is feature-based and not deeply bidirectional. Fedus et al. (2018) shows that the cloze task can be used to improve the robustness of text generation models.

 **ELMo 及其前身（彼得斯等人，2017，2018a）从不同维度推广了传统的词嵌入研究。** 它们从从左到右和从右到左的语言模型中提取上下文敏感特征。**每个token的上下文表征是从左到右和从右到左表征的连接**。**当将上下文词嵌入与现有的特定任务架构相结合时，ELMo 在几个主要的自然语言处理基准测试（彼得斯等人，2018a）上推动了技术的发展**，**包括问答**（拉杰普尔卡等人，2016）、**情感分析**（索舍尔等人，2013）和**命名实体识别**（宗金桑和德穆尔德，2013）。梅拉穆德等人（2016）提出通过**使用长短期记忆网络从左右上下文预测单个单词来学习上下文表征**。与 ELMo 类似，他们的模型是基于特征的，而不是深度双向的。费杜斯等人（2018）表明**完形填空任务可用于提高文本生成模型的稳健性。**

```ad-attention
- (1)**学习可广泛应用的单词表示方法一直是一个活跃的研究领域，包括非神经网络方法和神经网络方法**
	- 预训练的词嵌入是现代 NLP 系统不可或缺的一部分，与从零开始学习的嵌入相比有显着改进
	- **预训练词嵌入向量使用的目标有**
	- 从左到右的语言建模目标、在左右上下文中区分正确单词和错误单词的目标
- (2) **这些方法已经推广到更粗的粒度，如句子嵌入，段落嵌入**
	- **训练句子表示使用的目标有**
	- 对候选的下一句进行评分排序、给定前一个句子的表示从左到右生成下一个句子的单词 或去噪自动编码器派生的目标
- (3) ELMo 及其前身从不同维度推广了传统的词嵌入研究。
	- 从从左到右和从右到左的语言模型中**提取上下文敏感特征**。
	- **每个token的上下文表示是从左到右和从右到左表征的连接**
	- 将上下文词嵌入与现有的特定任务架构相结合时，ELMo在几个NLP基准测试(问答，情感分析，命名实体识别等)上表现优秀
- (4) 梅拉穆德等人
	- **使用长短期记忆网络从左右上下文预测单个单词来学习上下文表征**
- (5) 费杜斯等人（2018）
	- **完形填空任务可用于提高文本生成模型的稳健性。**
```

### 2.2 Unsupervised Fine-tuning Approaches

2.2 无监督微调方法

As with the feature-based approaches, the first works in this direction only pre-trained word embedding parameters from unlabeled text (Collobert and Weston, 2008).

 与基于特征的方法一样，这个方向上的**早期工作仅从无标签文本中预训练词嵌入参数**（科洛伯特和韦斯顿，2008）。 

More recently, sentence or document encoders which produce contextual token representations have been pre-trained from unlabeled text and fine-tuned for a supervised downstream task (Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018). The advantage of these approaches is that few parameters need to be learned from scratch. At least partly due to this advantage, OpenAI GPT (Radford et al., 2018) achieved previously state-of-the-art results on many sentence level tasks from the GLUE benchmark (Wang et al., 2018a). Left-to-right language model ing and auto-encoder objectives have been used for pre-training such models (Howard and Ruder, 2018; Radford et al., 2018; Dai and Le, 2015).

最近，**从无标签文本中预训练，并针对有监督的下游任务进行微调，的产生上下文词符表示的句子或文档编码器已经出现**（戴和乐，2015；霍华德和鲁德，2018；拉德福德等人，2018）。这些方法的优点是几乎不需要从头学习参数。至少部分由于这一优势，OpenAI GPT（拉德福德等人，2018）在 GLUE 基准测试（王等人，2018a）的许多句子级任务上取得了先前的最先进结果。从左到右的语言建模和自动编码器目标已被用于预训练此类模型（霍华德和鲁德，2018；拉德福德等人，2018；戴和乐，2015）。

```ad-attention
- (1)**基于特征的方法一样,早期工作仅从无标签文本中预训练词嵌入参数**
- (2) **从无标签文本中预训练，并针对有监督的下游任务进行微调，的产生上下文词符表示的句子或文档编码器已经出现**
	- 优点是**几乎不需要从头学习参数**
- 基于上述优点。OpenAI GPT（拉德福德等人，2018）在 GLUE 基准测试的**许多句子级任务**上取得了以前的最先进结果。
- 从左到右的语言建模和自动编码器目标(**训练句子表示的目标**)已被用于预训练此类模型
```


### 2.3 Transfer Learning from Supervised Data

2.3 基于监督数据的迁移学习

There has also been work showing effective transfer from supervised tasks with large datasets, such as natural language inference (Conneau et al., 2017) and machine translation (McCann et al., 2017). Computer vision research has also demonstrated the importance of transfer learning from large pre-trained models, where an effective recipe is to fine-tune models pre-trained with ImageNet (Deng et al., 2009; Yosinski et al., 2014).

 也有研究表明**从大规模数据集的监督任务中进行有效迁移是可行的**，比如自然语言推理（Conneau 等人，2017）和机器翻译（McCann 等人，2017）。计算机视觉研究也已经证明了从大型预训练模型进行迁移学习的重要性，其中一种有效的方法是对在 ImageNet 上预训练的模型进行微调（Deng 等人，2009；Yosinski 等人，2014）。

```ad-attention
- **从大规模数据集的监督任务中进行有效迁移是可行的,如自然语言推理和机器翻译。**
- **CV证明了从大型预训练模型进行迁移学习的重要性**，一种有效的方法是对在 ImageNet 上预训练的模型进行微调
- 沐神：**通过无监督预训练的模型比有监督预训练的模型要好**
```

## 3 BERT

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250112191611.png)
Figure 1: Overall(总体的) pre-training and fine-tuning procedures for BERT. Apart from(除了..之外) output layers, the same architectures are used in both pre-training and fine-tuning. The same pre-trained model parameters are used to initialize models for different down-stream tasks. During fine-tuning, all parameters are fine-tuned. [CLS] is a special symbol added in front of every input example, and [SEP] is a special separator token (e.g. separating questions/answers).
图 1：BERT 的整体预训练和微调流程。**除了输出层，预训练和微调使用相同的架构。相同的预训练模型参数用于初始化不同下游任务的模型。在微调过程中，所有参数都会被微调。**[CLS]是在每个输入示例前添加的特殊符号，[SEP]是一种特殊的分隔标记（例如用于分隔问题/答案）。

We introduce BERT and its detailed implementation in this section. There are two steps in our framework: pre-training and fine-tuning. During pre-training, the model is trained on unlabeled data over different pre-training tasks. For finetuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models, even though they are initialized with the same pre-trained parameters. The question-answering example in Figure 1 will serve as a running example for this section.

在本节中，我们介绍 BERT 及其详细实现。**我们的框架有两个步骤：预训练和微调**。**在预训练期间**，模型通过不同的预训练任务在无标签数据上进行训练。**对于微调**，首先使用预训练参数初始化 BERT 模型，然后使用来自下游任务的有标签数据对所有参数进行微调。每个下游任务都有单独的微调模型，尽管它们是用相同的预训练参数初始化的。图 1 中的问答示例将作为本节的一个运行示例。

A distinctive feature of BERT is its unified architecture across different tasks. There is minimal difference between the pre-trained architecture and the final downstream architecture.

BERT 的一个**显著特点是其在不同任务中的统一架构。预训练架构和最终的下游任务架构之间差异极小。**

```ad-attention
- (1) **BERT 及其详细实现**
- 框架的两个步骤:预训练和微调
- **预训练**: **模型通过不同的预训练任务在无标签数据上进行训练。**
- **微调**: 首先使用**预训练参数初始化 BERT 模型**，然后**使用来自下游任务的有标签数据对所有参数进行微调。**
- **每个下游任务都有单独的微调模型**，尽管它们是用相同的预训练参数初始化的。
- (2) BERT
- **显著特点是其在不同任务中的统一架构。预训练架构和最终的下游任务架构之间差异极小。**
- **写论文的时候，简单介绍一下用到的方法。**
```

**Model Architecture** BERT’s model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library.1 Because the use of Transformers has become common and our implementation is almost identical to the original, we will omit an exhaustive background description of the model architecture and refer readers to Vaswani et al. (2017) as well as excellent guides such as “The Annotated Transformer.”2

**模型架构** 
**BERT的模型架构是一个多层双向Transformer编码器**，它基于Vaswani等人（2017）所描述的原始实现，并在tensor2tensor库中发布。由于Transformer的使用已经很普遍，并且我们的实施与原始实施几乎相同，我们将省略对模型架构的详尽背景描述，并建议读者参考Vaswani等人（2017）以及诸如《带注释的Transformer》等优秀指南。 
**脚注：**
3: 在所有情况下，我们将前馈/滤波器大小设置为4H，即当H = 768时为3072，当H = 1024时为4096。
4: 我们注意到，在文献中，双向Transformer通常被称为“Transformer编码器”，而仅使用左上下文的版本被称为“Transformer解码器”，因为它可用于文本生成。

```ad-note
这里文章使用的就是transformer的架构，沐神说这里在介绍架构的时候使用的完全一样可以简单介绍，但是需要在前面related work多介绍一下这篇文章，方便读者阅读。

但像一下基础的参数还是需要介绍一下，不然看不懂，比如L,H,A
```

In this work, we denote the number of layers (i.e., Transformer blocks) as L , the hidden size as H , and the number of self-attention heads as $A. ^{3}$ We primarily report results on two model sizes: $BERT_{BASE}$ ( $L=12$ , $H=768$ , $A=12$ , Total Param $eters =110 M$ ) and $BERT_{LARGE}$( $L=24$ , $H=1024$ , $A=16$ ,Total $Parameters =340 M$ . $BERT _{BASE }$ was chosen to have the same model size as OpenAI GPT for comparison purposes. Critically, however, the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left.4

在这项工作中，我们将层数（即 Transformer 块）记为 \(L\)，隐藏层大小记为 \(H\)，自注意力头的数量记为 \(A\)。我们**主要报告两种模型尺寸的结果**：$BERT_{BASE}$（\(L = 12\)，\(H = 768\)，\(A = 12\)，总参数 = 110M）和 $BERT_{LARGE}$（\(L = 24\)，\(H = 1024\)，\(A = 16\)，总参数 = 340M）。**选择$BERT_{BASE}$是为了使其与 OpenAI GPT 具有相同的模型大小以便进行比较**。然而关键的是，**BERT 的 Transformer 使用双向自注意力，而 GPT 的 Transformer 使用受限的自注意力，即每个标记只能关注其左侧的上下文**。

```ad-note
- H = 768 指的是embbeing的维度，也就是一个token用768长度的向量表示
- 超参数换算成可学习的大小，相当于回顾transformer架构
	- 可学习参数来源于2块，一个是嵌入层，另一部分来源于transformer块(注意力机制和MLP)。
	- 嵌入层：输入是字典的大小 30k,输出是隐含层单元的个数也就是768
	- 自注意力机制本身不需要学习参数，但对多头注意力机制，他会把所有进入的K(Key),V(Value),Q(Query)分别做一层投影，每次投影的维度等于64,头的个数A乘以64等于H
> attention is all you need
在这项工作中，我们使用了$h = 8$个并行的注意力层，也就是头部(heads)。对于每个头部，我们使用了 $d_k = d_v = d_{model} /h ​=64$。由于每个头部的维度减小，总的**计算成本与具有完整维度**的单头注意力相似。
```

**Input/Output Representations** 
To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., ⟨Question, Answer ⟩) in one token sequence. Throughout this work, a “sentence” can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A “sequence” refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.

**输入/输出表示** 
为了**使 BERT 能够处理各种下游任务**，**我们的输入表示能够在一个token序列中明确地表示单个句子和一对句子（例如，⟨问题，答案⟩）**。在本文中，**“句子”可以是连续文本的任意片段，而不一定是实际的语言句子。“序列”是指输入到 BERT 的标记序列，它可以是单个句子或组合在一起的两个句子。**

We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary. The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B. As shown in Figure 1, we denote input embedding as E , the final hidden vector of the special [CLS] token as $C \in \mathbb{R}^{H}$ and the final hidden vector for the $i^{th }$ input token as $T_{i} \in \mathbb{R}^{H}$ 
For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings. A visualization of this construction can be seen in Figure 2.

我们**使用包含30,000个词元词汇表的WordPiece嵌入**（吴等人，2016）。**每个序列的首个词元始终是一个特殊的分类词元（[CLS]）**。**与该词元对应的最终隐藏状态会被用作分类任务的聚合序列表示**。句子对会被合并到一个单一序列中。我们通过**两种方式来区分句子**。首先，**我们用一个特殊词元（[SEP]）将它们分隔开**。其次，我**们给每个词元添加一个可学习的嵌入，以表明它属于句子A还是句子B**。如图1所示，我们将**输入嵌入记为\(E\)**，特殊的[CLS]词元的最终隐藏向量记为$(C\in\mathbb{R}^{H})$ ，第\(i\)个输入词元的最终隐藏向量记为$T_{i}\in\mathbb{R}^{H}$。 **对于给定的词元，其输入表示是通过将相应的词元嵌入、句段嵌入和位置嵌入相加而构建的**。这种构建方式的可视化效果可参见图2。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250112202323.png)
Figure 2: BERT input representation. The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings.
图2：BERT输入表示。输入嵌入是词元嵌入、句段嵌入以及位置嵌入之和。

```ad-attention
- **模型架构** 
- **BERT的模型架构是一个多层双向Transformer编码器**
- 参数：层数（即 Transformer 块）记为 (L)，隐藏层大小记为 \(H)，自注意力头的数量记为 (A)
- 主要模型：$BERT_{BASE}$（\(L = 12\)，\(H = 768\)，\(A = 12\)，总参数 = 110M）和 $BERT_{LARGE}$（\(L = 24\)，\(H = 1024\)，\(A = 16\)，总参数 = 340M
	- 选择$BERT_{BASE}$是为了使其与 OpenAI GPT 具有相同的模型大小以便进行比较**
	- 都是基于Trandformer的，一个是双向自注意力，一个是受限的自注意力(只关注左侧的上下文)
- **输入/输出表示** 
- **输入表示**能够在一个token序列中明确地表示单个句子和一对句子，为了使BERT能处理各种下游任务。
	- "句子"可以是连续文本的任意片段,不一定是真的语言句子
	- **“序列(sequence)”是指输入到 BERT 的token 序列，它可以是单个句子或组合在一起的两个句子。**
- 使用的是包含30,000个词元词汇表的WordPiece嵌入
	- **每个序列的首个词元始终是一个特殊的分类词元（[CLS]）**。**与该词元对应的最终隐藏状态会被用作分类任务的聚合序列表示**。
	- 句子对会被合并到单一的序列中，通过2种方式区分
		- **特殊词元（[SEP]）/ 给每个词元添加一个可学习的嵌入，以表明它属于句子A还是句子B**
- 参数含义：
	- 我们将**输入嵌入记为\(E\)**，
	- 特殊的[CLS]词元的最终隐藏向量记为$(C\in\mathbb{R}^{H})$ ，
	- 第\(i\)个输入词元的最终隐藏向量记为$T_{i}\in\mathbb{R}^{H}$。 
- **对于给定的词元，其输入表示是通过将相应的词元嵌入、句段嵌入和位置嵌入相加而构建的，如figure 2**
```

### 3.1 Pre-training BERT

Unlike Peters et al. (2018a) and Radford et al. (2018), we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two unsupervised tasks, described in this section. This step is presented in the left part of Figure 1.

与彼得斯等人（2018a）以及拉德福德等人（2018）不同，我们**不会使用传统的从左到右或从右到左的语言模型来对BERT进行预训练。相反，我们使用本节所描述的两个无监督任务对BERT进行预训练。这一步骤展示在图1的左侧部分。**

**Task #1: Masked LM** Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-toright and a right-to-left model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself”, and the model could trivially predict the target word in a multi-layered context.

**任务  # 1：掩码语言模型**
直观来看，我们有理由**相信深度双向模型肯定要比单向（从左到右）模型或者从左到右与从右到左模型的浅层拼接更强大**。遗憾的是，**标准的条件语言模型只能进行从左到右或从右到左的训练，因为双向条件设定会使每个单词间接“看到自身”**，这样一来，模型就能轻而易举地在多层语境中预测目标单词了。

In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a “masked LM” (MLM), although it is often referred to as a Cloze task in the literature (Taylor, 1953). In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders (Vincent et al., 2008), we only predict the masked words rather than reconstructing the entire input.

为了训练深度双向表征，我们简单地**随机掩码一定比例的输入词元**，然后**预测这些被掩码的词元**。我们将这一过程称为“**掩码语言模型**”（MLM），尽管在文献中它常被称作完形填空任务（泰勒，1953）。在此情形下，**与掩码词元对应的最终隐藏向量会被送入一个基于词汇表的输出softmax层**，就如同在标准语言模型中那样。在我们所有的实验中，我们会**随机掩码每个序列中15%的WordPiece词元**。与去噪自编码器（文森特等人，2008）不同的是，我们**只预测被掩码的单词，而不是重构整个输入**。

Although this allows us to obtain a bidirectional pre-trained model, a downside is that we are creating a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning. To mitigate this, we do not always replace “masked” words with the actual [MASK] token. The training data generator chooses 15% of the token positions at random for prediction. If the i-th token is chosen, we replace the $\bar{\imath}$ -th token with (1) the [MASK] token 80% of the time (2) a random token 10% of the time (3) the unchanged i -th token 10% of the time. Then, $T_{i}$ will be used to predict the original token with cross entropy loss. We compare variations of this procedure in Appendix C.2.

尽管这能让我们获得一个双向预训练模型，但不利的一面在于，我们造成了**预训练和微调之间的不匹配，因为[MASK]标记在微调阶段并不会出现**。为了缓解这一问题，我们**并非总是用实际的[MASK]标记来替换“被掩码”的单词。训练数据生成器会随机选取15%的词元位置来进行预测**。如果第\(i\)个词元被选中，我们会按以下方式替换第\(i\)个词元：（1）80%的概率将其替换为[MASK]标记；（2）10%的概率替换为一个随机词元；（3）10%的概率保持第\(i\)个词元不变。然后，将使用\($T_{i}$\)通过交叉熵损失来预测原始词元。我们在附录C.2中对这一流程的不同变体进行了比较。

**Task #2: Next Sentence Prediction (NSP)**
Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext). As we show in Figure 1, c is used for next sentence prediction (NSP).5 Despite its simplicity, we demonstrate in Section 5.1 that pre-training towards this task is very beneficial to both QA and NLI. 6

**任务 # 2：下一句预测（NSP）** 
许多重要的下游任务，**比如问答（QA）和自然语言推理（NLI），都是基于对两个句子之间关系的理解**，而语言建模并不能直接捕捉到这种关系。为了**训练一个能够理解句子间关系的模型**，我们**针对一个二值化的下一句预测任务进行预训练，该任务可以轻易地从任何单语语料库中生成**。 具体来说，在**为每个预训练示例选择句子A和句子B时**，50%的情况下，句子B是紧跟在句子A之后的实际下一句（**标记为“IsNext**”），而另外50%的情况下，句子B是来自语料库的随机句子（标记为“**NotNext**”）。如图1所示，**\(c\)用于下一句预测（NSP）**。尽管这个任务很简单，但我们会在第5.1节中证明**针对该任务进行预训练对问答和自然语言推理这两项任务都是非常有益的。**

The NSP task is closely related to representation learning objectives used in Jernite et al. (2017) and Logeswaran and Lee (2018). However, in prior work, only sentence embeddings are transferred to down-stream tasks, where BERT transfers all parameters to initialize end-task model parameters. 

**下一句预测（NSP）任务**与杰尼特等人（2017）以及洛格斯瓦兰和李（2018）所使用的**表征学习目标密切相关**。然而，**在先前的研究工作中，只有句子嵌入会被迁移到下游任务中，而BERT则是将所有参数进行迁移，用以初始化最终任务的模型参数。**

**Pre-training data** The pre-training procedure largely follows the existing literature on language model pre-training. For the pre-training corpus we use the BooksCorpus (800M words) (Zhu et al., 2015) and English Wikipedia (2,500M words). For Wikipedia we extract only the text passages and ignore lists, tables, and headers. It is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark (Chelba et al., 2013) in order to extract long contiguous sequences.

预训练数据 
**预训练过程在很大程度上遵循了现有的语言模型预训练相关文献**。对于预训练**语料库**，我们使用了**BooksCorpus**（8亿词）（朱等人，2015）以及**英文维基百科**（25亿词）。对于维基百科，我们**只提取其中的文本段落**，而忽略列表、表格以及标题部分。**为了提取出较长的连续序列，使用文档级语料库**而非像“十亿单词基准”（切尔巴等人，2013）**那样打乱的句子级语料库是至关重要的。**
```ad-attention
- 3.1 Pre-training BERT
- (1)**不会使用传统的从左到右或从右到左的语言模型来对BERT进行预训练,两个无监督任务对BERT进行预训练**
- **Task #1: Masked LM(掩码语言模型)**
- (2)相信深度双向模型肯定要比单向（从左到右）模型或者从左到右与从右到左模型的浅层拼接更强大
	- 问题：**双向条件设定会使每个单词间接“看到自身”**
- (3)缓解问题
	- 简单地**随机掩码一定比例的输入词元**，然后**预测这些被掩码的词元**——“**掩码语言模型**”（MLM）也叫完型填空任务(Cloze task)
	- **与掩码词元对应的最终隐藏向量会被送入一个基于词汇表的输出softmax层,就会被分类到具体的预测单词**
	- 本文的比例是15%，只是预测被掩码的单词，不重构整个输入。
- (4)新的问题：[MASK]标记在微调阶段并不会出现，导致预训练和微调不匹配
	- 缓解问题——方法approaches:
	- 如果第\(i\)个词元被选中，我们会按以下方式替换第\(i\)个词元：
	- （1）80%的概率将其替换为[MASK]标记；
	- （2）10%的概率替换为一个随机词元；
	- （3）10%的概率保持第\(i\)个词元不变。
	- 然后，将使用\(T_{i}\)通过交叉熵损失来预测原始词元。
- **Task #2: 下一句预测 Next Sentence Prediction (NSP)**
- **目的**：需要解决问答（QA）和自然语言推理（NLI），是基于对两个句子之间关系的理解，语言建模无法处理这种关系。
- **方法**：**针对一个二值化的下一句预测任务进行预训练**(该任务可以轻易地从任何单语语料库中生成),用来理解句子间的关系
	- 在**为每个预训练示例选择句子A和句子B时**，50%的情况下，句子B是紧跟在句子A之后的实际下一句（**标记为“IsNext**”），
	- 而另外50%的情况下，句子B是来自语料库的随机句子（标记为“**NotNext**”）
- 应用：**\(c\)用于下一句预测（NSP）**
	- 尽管简单，但是对于QA和NLI很有用，5.1节会证明！
- **预训练数据Pre-training data**
- **过程**：
	- 预训练过程在很大程度上遵循了现有的语言模型预训练相关文献
- **语料库**
	- **BooksCorpus**以及**英文维基百科**
- **处理**：
	- 维基百科，我们**只提取其中的文本段落，忽略表格，标题等**
	- **要提取较长的连续序列，使用文档级语料库，不能是打乱的句子**
```

### 3.2 Fine-tuning BERT

Fine-tuning is straightforward since the selfattention mechanism in the Transformer allows BERT to model many downstream tasks— whether they involve single text or text pairs—by swapping out the appropriate inputs and outputs. For applications involving text pairs, a common pattern is to independently encode text pairs before applying bidirectional cross attention, such as Parikh et al. (2016); Seo et al. (2017). BERT instead uses the self-attention mechanism to unify these two stages, as encoding a concatenated text pair with self-attention effectively includes bidirectional cross attention between two sentences.

微调过程很简单，**因为Transformer中的自注意力机制使得BERT能够通过替换相应的输入和输出，对许多下游任务（无论这些任务涉及的是单个文本还是文本对）进行建模。** 对于涉及**文本对的应用**，一种常见的模式是**在应用双向交叉注意力之前对文本对分别进行编码**，比如帕里克等人（2016）以及徐等人（2017）的做法。而**BERT则利用自注意力机制将这两个阶段统一起来**，因为使用自注意力对拼接后的文本对进行编码实际上就包含了两个句子之间的双向交叉注意力。

For each task, we simply plug in the taskspecific inputs and outputs into BERT and finetune all the parameters end-to-end. At the input, sentence A and sentence B from pre-training are analogous to (1) sentence pairs in paraphrasing, (2) hypothesis-premise pairs in entailment, (3) question-passage pairs in question answering, and (4) a degenerate text-∅pair in text classification or sequence tagging. At the output, the token representations are fed into an output layer for tokenlevel tasks, such as sequence tagging or question answering, and the [CLS] representation is fed into an output layer for classification, such as entailment or sentiment analysis. 

**对于每项任务，我们只需将特定于任务的输入和输出接入BERT，然后对所有参数进行端到端的微调**。在输入方面，预训练中的句子A和句子B类似于以下几种情况：（1）释义任务中的句子对；（2）蕴含任务中的假设 - 前提对；（3）问答任务中的问题 - 文段对；（4）文本分类或序列标注任务中单一文本与空值构成的文本对。在输出方面，**词元表征( token representations)会被送入输出层以处理词元级别的任务，比如序列标注或问答任务**；而**[CLS]表征则会被送入输出层用于分类任务**，例如蕴含或情感分析任务。

Compared to pre-training, fine-tuning is relatively inexpensive. All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model.7 We describe the task-specific details in the corresponding subsections of Section 4. More details can be found in Appendix A.5.

 **与预训练相比，微调的成本相对较低**。从完全相同的预训练模型开始，本文中所有的结果在单个云TPU上最多1小时就能复现，在GPU上则需要几个小时。我们会在第4节相应的小节中描述特定任务的详细情况。更多细节可在附录A.5中找到。

```ad-attention
- 3.2 Fine-tuning BERT
- **过程**：简单，原因是**Transformer中的自注意力机制使得BERT能够通过替换相应的输入和输出，对许多下游任务（无论这些任务涉及的是单个文本还是文本对）进行建模。**
	- 涉及文本对的应用：
		- 常见的模式是**在应用双向交叉注意力之前对文本对分别进行编码**
		- **BERT则利用自注意力机制将这两个阶段统一起来，因为self-attention对拼接后的文本对进行编码时间就包含了2个句子之间的双向交叉注意力。**
- **对于每项任务，我们只需将特定于任务的输入和输出接入BERT，然后对所有参数进行端到端的微调**
	- 输入方面，句子A和B的情况有
		- （1）释义任务中的句子对；
		- （2）蕴含任务中的假设 - 前提对；
		- （3）问答任务中的问题 - 文段对；
		- （4）文本分类或序列标注任务中单一文本与空值构成的文本对。
	- 输出方面：
		- (1)**词元表征( token representations)会被送入输出层以处理词元级别的任务，比如序列标注或问答任务**；
		- (2)**[CLS]表征(representation)则会被送入输出层用于分类任务**，例如蕴含或情感分析任务。
- **成本**： **与预训练相比，微调的成本相对较低**。
	- a single Cloud TPU : 最多1h
	- a GPU : a few hours
- 其他: [[CPU、GPU、TPU、NPU等到底是什么？]]
```

## 4 Experiments

In this section, we present BERT fine-tuning results on 11 NLP tasks.
在本节中，我们将展示BERT在11个自然语言处理（NLP）任务上的微调结果。 

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250112222428.png)
Table 1: GLUE Test results, scored by the evaluation server (https://gluebenchmark.com/leaderboard). The number below each task denotes the number of training examples. The “Average” column is slightly different than the official GLUE score, since we exclude the problematic WNLI set.8 BERT and OpenAI GPT are single model, single task. F1 scores are reported for QQP and MRPC, Spearman correlations are reported for STS-B, and accuracy scores are reported for the other tasks. We exclude entries that use BERT as one of their components.
表1：GLUE测试结果，由评估服务器（https://gluebenchmark.com/leaderboard）评分。每个任务下方的数字表示训练样本的数量。“平均”栏与官方GLUE分数略有不同，因为我们排除了存在问题的WNLI数据集。BERT和OpenAI GPT都是单模型、单任务的。针对QQP（文本语义相似度）和MRPC（微软研究释义语料库）任务报告的是F1分数，针对STS-B（语义文本相似度基准）任务报告的是斯皮尔曼相关系数，针对其他任务报告的则是准确率分数。我们排除了将
BERT作为其组件之一的条目。
**具体NLP任务含义可以看**：[[【HugBert04】GLUE：BERT类模型的通用评估基准]]

### 4.1 GLUE

The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018a) is a collection of diverse natural language understanding tasks. Detailed descriptions of GLUE datasets are included in Appendix B.1.

 通用语言理解评估（GLUE）基准（Wang等人，2018a）是一组多样化的自然语言理解任务。GLUE数据集的详细描述包含在附录B.1中。 

To fine-tune on GLUE, we represent the input sequence (for single sentence or sentence pairs) as described in Section 3, and use the final hidden vector $C \in \mathbb{R}^{H}$ corresponding to the first input token ([CLS]) as the aggregate representation. The only new parameters introduced during fine-tuning are classification layer weights $W E$ $\mathbb{R}^{K ×H}$ , where K is the number of labels. We compute a standard classification loss with c and W , i.e., log(softmax( $(C W^{T})$ .

 为了在GLUE上进行微调，我们按照第3节所述来表示输入序列（针对单句或句对），并使用与首个输入词元（[CLS]）对应的最终隐藏向量\($C\in\mathbb{R}^{H}$\)作为聚合表示。在微调过程中引入的唯一新参数是分类层权重\($W\in\mathbb{R}^{K\times H}$\)，其中\(K\)是标签的数量。我们利用\(C\)和\(W\)计算标准的分类损失，即\($\log(\text{softmax}(C W^{T}))$\)。

We use a batch size of 32 and fine-tune for 3 epochs over the data for all GLUE tasks. For each task, we selected the best fine-tuning learning rate (among 5e-5, 4e-5, 3e-5, and 2e-5) on the Dev set. Additionally, for $BERT _{LARGE }$ we found that finetuning was sometimes unstable on small datasets, so we ran several random restarts and selected the best model on the Dev set. With random restarts, we use the same pre-trained checkpoint but perform different fine-tuning data shuffling and classifier layer initialization.9
**我们在所有GLUE任务的数据上使用批量大小为32的数据进行3轮微调。对于每项任务，我们会在开发集（Dev set）上选择最佳的微调学习率（从\(5e-5\)、\(4e-5\)、\(3e-5\)和\(2e-5\)中选取）。** 此外，**对于\(BERT_{LARGE}\)**，我们发现其在小型数据集上进行微调时有时不太稳定，所以我们进行了**多次随机重启**，并在开发集上选择最佳模型。在随机重启时，我们使用相同的预训练检查点，但对微调数据进行不同的打乱处理以及对分类器层进行不同的初始化。 

Results are presented in Table 1. Both $BERT BASE$ and $BERT _{LARGE }$ outperform all systems on all tasks by a substantial(巨大的) margin(幅度), obtaining 4.5% and 7.0% respective(分别的，各自的) average accuracy improvement over the prior state of the art. Note that $BERT BASE$ and OpenAI GPT are nearly identical in terms of model architecture apart from the attention masking. For the largest and most widely reported GLUE task, MNLI, BERT obtains a 4.6% absolute accuracy improvement. On the official GLUE leaderboard10, $BERT _{LARGE }$ obtains a score of 80.5, compared to OpenAI GPT, which obtains 72.8 as of the date of writing.
We find that $BERT _{LARGE }$ significantly outperforms $BERT BASE$ across all tasks, especially those with very little training data. The effect of model size is explored more thoroughly in Section 5.2.

结果展示在表1中。**\(BERT_{BASE}\)和\(BERT_{LARGE}\)在所有任务上都大幅超越了所有其他系统**，相较于之前的最优水平，平均准确率分别提高了\(4.5\%\)和\(7.0\%\)。需要注意的是，除了注意力掩码方面有所不同外，\(BERT_{BASE}\)和OpenAI GPT在模型架构上几乎是一样的。对于GLUE中规模最大且被广泛报道的任务——多体裁自然语言推理（MNLI），BERT实现了\(4.6\%\)的绝对准确率提升。在官方的GLUE排行榜上，截至撰写本文时，\(BERT_{LARGE}\)的得分为80.5，而OpenAI GPT的得分为72.8。 **我们发现\(BERT_{LARGE}\)在所有任务上都显著优于\(BERT_{BASE}\)，尤其是在那些训练数据非常少的任务上。** 模型大小的影响在第5.2节中会进行更深入的探讨。

- 语言理解评估（GLUE）基准
- 处理流程：
	- 输入序列+[CLS]\(开始的词元)得到隐藏层表示\($C\in\mathbb{R}^{H}$\),在微调过程中引入的唯一新参数是分类层权重\($W\in\mathbb{R}^{K\times H}$\)，其中\(K\)是标签的数量。我们利用\(C\)和\(W\)计算标准的分类损失，即\($\log(\text{softmax}(C W^{T}))$\)。
- 训练流程：
	- 在所有GLUE任务的数据上**使用批量大小(batch_size)为32的数据进行3轮微调。**
	- **每项任务**，我们会在开发集（Dev set）上**选择最佳的微调学习率**（从\(5e-5\)、\(4e-5\)、\(3e-5\)和\(2e-5\)中选取）。
	- **\(BERT_{LARGE}\)**在小数据集微调不太稳定，进行多次随机重启。
- 结果：
	- **\(BERT_{BASE}\)和\(BERT_{LARGE}\)在所有任务上都大幅超越了所有其他系统**。
	- **发现\(BERT_{LARGE}\)在所有任务上都显著优于\(BERT_{BASE}\)，尤其是在那些训练数据非常少的任务上。**

### 4.2 SQuAD v1.1

The Stanford Question Answering Dataset (SQuAD v1.1) is a collection of 100k crowdsourced(众包的) question/answer pairs (Rajpurkar et al., 2016). Given a question and a passage from Wikipedia containing the answer, the task is to predict the answer text span in the passage.
斯坦福问答数据集（SQuAD v1.1）包含10万个众包的问题/答案对（拉杰普尔卡等人，2016）。给定一个问题以及维基百科中包含答案的文段，任务就是预测文段中答案文本的跨度范围。

> 众包的含义,从一广泛群体，特别是在线社区，获取所需想法，服务或内容贡献的实践。

As shown in Figure 1, in the question answering task, we represent the input question and passage as a single packed sequence, with the question using the A embedding and the passage using the B embedding. We only introduce a start vector $S \in \mathbb{R}^{H}$ and an end vector $E \in \mathbb{R}^{H}$ during fine-tuning. The probability of word i being the start of the answer span is computed as a dot product between $T_{i}$ and S followed by a softmax over all of the words in the paragraph: $P_{i}=\frac{e^{S \cdot T_{i}}}{\sum_{j} e^{S \cdot T_{j}}}$ The analogous formula is used for the end of the answer span. The score of a candidate span from position i to position j is defined as $S \cdot T_{i}+E \cdot T_{j}$ , and the maximum scoring span where $j ≥i$ is used as a prediction. The training objective is the sum of the log-likelihoods of the correct start and end positions. We fine-tune for 3 epochs with a learning rate of 5e-5 and a batch size of 32.

 如图1所示，在问答任务中，**我们将输入的问题和文段表示为一个合并后的单一序列，问题使用A嵌入，文段使用B嵌入。** 在微调阶段，我们仅引入一个起始向量\($S\in\mathbb{R}^{H}$\)和一个结束向量\($E\in\mathbb{R}^{H}$\)。单词\(i\)是 答案跨度起始位置的概率 通过计算\($T_{i}$\)与\(S\)的点积，然后对文段中所有单词进行softmax运算得出：\($P_{i}=\frac{e^{S\cdot T_{i}}}{\sum_{j}e^{S\cdot T_{j}}}$\)。对于答案跨度的结束位置，使用类似的公式计算。从位置\(i\)到位置\(j\)的候选跨度得分定义为\($S\cdot T_{i}+E\cdot T_{j}$\)，并将\($j\geq i$\)时得分最高的跨度作为预测结果。**训练目标是正确起始位置和结束位置的对数似然之和。我们以\($5e-5$\)的学习率、批量大小为32进行3轮微调。** 

> 特殊的[CLS]词元的最终隐藏向量记为$(C\in\mathbb{R}^{H})$ ，第\(i\)个输入词元的最终隐藏向量记为$T_{i}\in\mathbb{R}^{H}$。 

Table 2 shows top leaderboard entries as well as results from top published systems (Seo et al., 2017; Clark and Gardner, 2018; Peters et al., 2018a; Hu et al., 2018). The top results from the SQuAD leaderboard do not have up-to-date public system descriptions available,11 and are allowed to use any public data when training their systems. We therefore use modest data augmentation in our system by first fine-tuning on TriviaQA (Joshi et al., 2017) befor fine-tuning on SQuAD.
Our best performing system outperforms the top leaderboard system by +1.5 F1 in ensembling and +1.3 F1 as a single system. In fact, our single BERT model outperforms the top ensemble system in terms of F1 score. Without TriviaQA fine tuning data, we only lose 0.1-0.4 F1, still outper forming all existing systems by a wide margin.12

表2展示了排行榜上排名靠前的条目以及已发表的顶尖系统（徐等人，2017；克拉克和加德纳，2018；彼得斯等人，2018a；胡等人，2018）的结果。SQuAD排行榜上排名靠前的结果并没有最新的公开系统描述可用，而且这些系统在训练时被允许使用任何公开数据。因此，我们在系统中**采用了适度的数据增强方法，即在针对SQuAD进行微调之前，先对TriviaQA（乔希等人，2017）进行微调。** 我们表现最佳的系统在集成时的F1值比排行榜上排名靠前的系统高出\(+1.5\)，作为单一系统时F1值高出\(+1.3\)。实际上，**就F1分数而言，我们的单一BERT模型表现优于排名靠前的集成系统。如果不使用TriviaQA微调数据，我们的F1值仅会降低\(0.1 - 0.4\)，但仍然大幅优于所有现有系统。**

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250112222514.png)
Table 2: SQuAD 1.1 results. The BERT ensemble is 7x systems which use different pre-training checkpoints and fine-tuning seeds.
表2：斯坦福问答数据集（SQuAD）1.1版的结果。BERT集成模型是由7个使用了不同预训练检查点和微调随机种子的系统构成。

- 斯坦福问答数据集（SQuAD v1.1）
- 包含10万个众包的问题/答案对。给定一个问题以及维基百科中包含答案的文段，任务就是预测文段中答案文本的跨度范围。
	- 在问答任务中，**我们将输入的问题和文段表示为一个合并后的单一序列，问题使用A嵌入，文段使用B嵌入。** 
	- 在微调阶段，我们仅引入一个起始向量\($S\in\mathbb{R}^{H}$\)和一个结束向量\($E\in\mathbb{R}^{H}$\)。单词\(i\)是 答案跨度**起始位置的概率** 通过计算\($T_{i}$\)与\(S\)的点积，然后对文段中所有单词进行softmax运算得出：\($P_{i}=\frac{e^{S\cdot T_{i}}}{\sum_{j}e^{S\cdot T_{j}}}$\)。
	- 对于**答案跨度的结束位置**，使用类似的公式计算。从位置\(i\)到位置\(j\)的候选跨度得分定义为\($S\cdot T_{i}+E\cdot T_{j}$\)，并将\($j\geq i$\)时得分最高的跨度作为预测结果。**训练目标是正确起始位置和结束位置的对数似然之和。我们以\($5e-5$\)的学习率、批量大小为32进行3轮微调。** 
- 成果
	- 我们在系统中**采用了适度的数据增强方法，即在针对SQuAD进行微调之前，先对TriviaQA（乔希等人，2017）进行微调。** 
	- **就F1分数而言，我们的单一BERT模型表现优于排名靠前的集成系统。如果不使用TriviaQA微调数据，我们的F1值仅会降低\(0.1 - 0.4\)，但仍然大幅优于所有现有系统。**

### 4.3 SQuAD v2.0

The SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic.

We use a simple approach to extend the SQuAD v1.1 BERT model for this task. We treat questions that do not have an answer as having an answer span with start and end at the [CLS] token. The probability space for the start and end answer span positions is extended to include the position of the [CLS] token. For prediction, we compare the score of the no-answer span: $s_{null }=$ $S \cdot C+E \cdot C$ to the score of the best non-null span $\hat{s_{i, j}}=max _{j ≥i} S \cdot T_{i}+E \cdot T_{j}$ . We predict a non-null answer when $s_{i, j}>s_{null }+\tau$ , where the threshold τ is selected on the dev set to maximize F1. We did not use TriviaQA data for this model. We fine-tuned for 2 epochs with a learning rate of 5e-5 and a batch size of 48.

The results compared to prior leaderboard entries and top published work (Sun et al., 2018; Wang et al., 2018b) are shown in Table 3, excluding systems that use BERT as one of their components. We observe a +5.1 F1 improvement over the previous best system.

SQuAD 2.0任务扩展了SQuAD 1.1的问题定义，**它考虑到了在所提供的文段中可能不存在简短答案的情况，这使得该问题更加贴近现实。** 

我们采用一种简单的方法，针对这项任务对SQuAD v1.1的BERT模型进行扩展。我们将没有答案的问题视作其答案跨度的起始和结束位置都在[CLS]词元处。起始和结束答案跨度位置的概率空间被扩展至包含[CLS]词元的位置。在进行预测时，我们将无答案跨度的得分\($s_{null}=S\cdot C + E\cdot C$\)与最佳非空跨度的得分\($\hat{s_{i,j}} = \max_{j\geq i} S\cdot T_{i} + E\cdot T_{j}$\)进行比较。当\($s_{i,j} > s_{null} + \tau$\)时（其中阈值\($\tau$\)是在开发集上选取的，目的是使F1值最大化），我们预测存在非空答案。我们在构建这个模型时没有使用TriviaQA数据。我们以\(5e-5\)的学习率、批量大小为48进行了2轮微调。 
与之前排行榜上的条目以及已发表的顶尖成果（孙等人，2018；王等人，2018b）相比的结果展示在表3中（不包含将BERT作为其组件之一的系统）。我们观察到相较于之前表现最佳的系统，F1值提高了\(+5.1\)。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250112222546.png)
Table 3: SQuAD 2.0 results. We exclude entries that use BERT as one of their components.
表3：SQuAD 2.0的结果。我们排除了将BERT作为其组件之一的条目。

- SQuAD 2.0
- 任务：扩展了SQuAD 1.1的问题定义，**它考虑到了在所提供的文段中可能不存在简短答案的情况，这使得该问题更加贴近现实。**
- 训练过程：
	- 将没有答案的问题视作其答案跨度的起始和结束位置都在[CLS]词元处。起始和结束答案跨度位置的概率空间被扩展至包含[CLS]词元的位置。
	- 在进行预测时，我们将无答案跨度的得分\($s_{null}=S\cdot C + E\cdot C$\)与最佳非空跨度的得分\($\hat{s_{i,j}} = \max_{j\geq i} S\cdot T_{i} + E\cdot T_{j}$\)进行比较。当\($s_{i,j} > s_{null} + \tau$\)时（其中阈值\($\tau$\)是在开发集上选取的，目的是使F1值最大化），我们预测存在非空答案。
	- 我们在构建这个模型时没有使用TriviaQA数据。**我们以\(5e-5\)的学习率、批量大小为48进行了2轮微调。** 

### 4.4 SWAG

The Situations With Adversarial Generations (SWAG) dataset contains 113k sentence-pair completion examples that evaluate grounded commonsense inference (Zellers et al., 2018). Given a sentence, the task is to choose the most plausible continuation among four choices.
“对抗生成情境”（SWAG）数据集包含11.3万个句子对补全示例，用于评估基于常识的推理能力（泽勒斯等人，2018）。给定一个句子，任务是从四个选项中选出最合理的后续句子。 

When fine-tuning on the SWAG dataset, we construct four input sequences, each containing the concatenation of the given sentence (sentence A) and a possible continuation (sentence B). The only task-specific parameters introduced is a vector whose dot product with the [CLS] token representation c denotes a score for each choice which is normalized with a softmax layer.
在针对SWAG数据集进行微调时，我们构建四个输入序列，每个序列都包含给定句子（句子A）与一个可能的后续句子（句子B）的拼接内容。唯一引入的特定于该任务的参数是一个向量，该向量与[CLS]词元表征\(c\)的点积表示每个选项的得分，然后通过一个softmax层进行归一化处理。

We fine-tune the model for 3 epochs with a learning rate of 2e-5 and a batch size of 16. Results are presented in Table 4. BERTLARGE outperforms the authors’ baseline ESIM+ELMo system by +27.1% and OpenAI GPT by 8.3%.

我们以\(2e-5\)的学习率、批量大小为16对模型进行3轮微调。结果展示在表4中。\(BERT_{LARGE}\)比作者提出的基准系统（ESIM + ELMo系统）性能高出\(27.1\%\)，比OpenAI GPT高出\(8.3\%\)。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250112222608.png)
Table 4: SWAG Dev and Test accuracies. †Human performance is measured with 100 samples, as reported in the SWAG paper.
表4：“对抗生成情境”（SWAG）数据集开发集和测试集的准确率。†人类表现是通过100个样本进行衡量的，正如SWAG论文中所报告的那样。

- “对抗生成情境”（SWAG）数据集包含11.3万个句子对补全示例，用于评估基于常识的推理能力，即**给定一个句子，任务是从四个选项中选出最合理的后续句子。** 
- 训练过程：
	- 进行微调时，我们构建**四个输入序列，每个序列都包含给定句子（句子A）与一个可能的后续句子（句子B）的拼接内容**。
	- 唯一引入的特定于该任务的参数是一个向量，**该向量与[CLS]词元表征\(c\)的点积表示每个选项的得分，然后通过一个softmax层进行归一化处理。**

## 5 Ablation Studies
消融(切除)研究

```ad-tip
消融实验（Ablation Study）是一种在研究中用于评估模型或系统中不同组件重要性的实验方法。 在自然语言处理等领域，以 BERT 模型的消融实验为例，其具体操作如下：

- **实验设计思路**：在保持其他条件尽可能相同的情况下，依次移除或修改模型的某些特定部分，如 BERT 中的某个预训练任务、某种网络结构或特定的损失函数组件等，然后观察模型在相同任务上的性能变化。
    
- **评估指标**：通常会采用如准确率、F1 值、困惑度等在相应任务中常用的评估指标来量化模型性能。例如在文本分类任务中，观察模型在去除某组件后的准确率下降情况；在语言建模任务中，关注困惑度的变化。
    
- **作用和意义**：通过消融实验，可以清晰地了解到模型各个组成部分对整体性能的贡献程度。比如，如果去除 BERT 的下一句预测任务后，模型在某些自然语言推理任务上的准确率大幅下降，就说明该任务对于模型在这些任务上的表现起到关键作用；若某个特定的网络层被移除后性能变化不大，则表明该层相对不那么重要，这有助于研究人员在模型设计和优化时决定保留或改进哪些部分，从而更高效地提升模型性能或简化模型结构。
```

In this section, we perform ablation experiments over a number of facets of BERT in order to better understand their relative importance. Additional ablation studies can be found in Appendix C.
在本节中，我们对BERT的多个方面进行消融实验，以便更好地了解它们的相对重要性。更多消融研究内容可在附录C中找到。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250115234754.png)
Table 5: Ablation over the pre-training tasks using the $BERT _{BASE }$ architecture. “No NSP” is trained without the next sentence prediction task. “LTR & No NSP” is trained as a left-to-right LM without the next sentence prediction, like OpenAI GPT. “+ BiLSTM” adds a randomly initialized BiLSTM on top of the “LTR + No $NSP$ model during fine-tuning.

表5：基于$BERT_{BASE}$架构对预训练任务的消融实验。“无NSP”表示在训练中不使用下一句预测任务。“从左到右且无NSP”像OpenAI GPT一样，作为从左到右的语言模型进行训练，且不使用下一句预测任务。“+双向长短期记忆网络（BiLSTM）”表示在微调期间，在“从左到右 + 无NSP”模型之上添加一个随机初始化的双向长短期记忆网络。

### 5.1 Effect of Pre-training Tasks

We demonstrate the importance of the deep bidirectionality of BERT by evaluating two pretraining objectives using exactly the same pretraining data, fine-tuning scheme, and hyperparameters as $BERT _{BASE }$
我们通过评估两种预训练目标，来展示BERT深度双向性的重要性。这两种预训练目标使用与$BERT_{BASE}$完全相同的预训练数据、微调方案和超参数。

**No NSP:** A bidirectional model which is trained using the “masked LM” (MLM) but without the “next sentence prediction” (NSP) task.
**LTR & No NSP:** A left-context-only model which is trained using a standard Left-to-Right (LTR) LM, rather than an MLM. The left-only constraint was also applied at fine-tuning, because removing it introduced a pre-train/fine-tune mismatch that degraded downstream performance. Additionally, this model was pre-trained without the NSP task. This is directly comparable to OpenAI GPT, but using our larger training dataset, our input representation, and our fine-tuning scheme.

**无 NSP：** 一个使用“掩码语言模型”（MLM）训练但没有“下一句预测”（NSP）任务的双向模型。

**LTR & 无 NSP：** 一个仅使用左上下文的模型，它使用标准的从左到右（LTR）语言模型而非 MLM 进行训练。**在微调时也应用了仅左上下文的限制，因为移除该限制会引入预训练/微调不匹配的问题，从而降低下游性能。**此外，该模型在预训练时没有 NSP 任务。这与 OpenAI GPT 直接可比，但使用了我们更大的训练数据集、输入表示和微调方案。

We first examine the impact brought by the NSP task. In Table 5, we show that removing NSP hurts performance significantly on QNLI, MNLI, and SQuAD 1.1. Next, we evaluate the impact of training bidirectional representations by comparing “No NSP” to “LTR & No NSP”. The LTR model performs worse than the MLM model on all tasks, with large drops on MRPC and SQuAD.

我们首先研究了下一句预测（NSP）任务所带来的影响。在表 5 中，**我们表明移除 NSP 会显著降低在 QNLI、MNLI 和 SQuAD 1.1 上的性能。**

接下来，我们通过比较“无 NSP”和“从左到右且无 NSP”来评估训练双向表示的影响。**从左到右（LTR）模型在所有任务上的表现都比掩码语言模型（MLM）差，在 MRPC 和 SQuAD 上的性能下降幅度很大。**

For SQuAD it is intuitively clear that a LTR model will perform poorly at token predictions, since the token-level hidden states have no rightside context. In order to make a good faith attempt at strengthening the LTR system, we added a randomly initialized BiLSTM on top. This does significantly improve results on SQuAD, but the results are still far worse than those of the pretrained bidirectional models. The BiLSTM hurts performance on the GLUE tasks. 

**对于 SQuAD 任务，直观上很明显，从左到右（LTR）模型在词元预测方面表现会很差，因为词元级隐藏状态没有右侧上下文。** 为了切实尝试加强 LTR 系统，我们在其顶部添加了一个随机初始化的双向长短期记忆网络（BiLSTM）。这确实显著提高了 SQuAD 上的结果，但结果仍然远不如预训练的双向模型。而且 BiLSTM 还损害了在 GLUE 任务上的性能。

We recognize that it would also be possible to train separate LTR and RTL models and represent each token as the concatenation of the two models, as ELMo does. However: (a) this is twice as expensive as a single bidirectional model; (b) this is non-intuitive for tasks like QA, since the RTL model would not be able to condition the answer on the question; (c) this it is strictly less powerful than a deep bidirectional model, since it can use both left and right context at every layer.

我们认识到，也可以像 ELMo 那样**分别训练 LTR 和 RTL（从右到左）模型，并将每个词元表示为这两个模型的连接。**然而：（a）这比单个双向模型的**成本要高**一倍；（b）对于问答（QA）等任务来说不直观，因为 RTL(右侧到左侧) 模型无法根据问题来确定答案；（c）这**严格来说不如深度双向模型强大**，因为深度双向模型在每一层都能同时使用左右上下文。

```ad-note
本文通过设置“No NSP”（仅用掩码语言模型训练的双向模型）和“LTR & No NSP”（仅用从左到右语言模型训练且无下一句预测任务的模型）两种模型，并采用与 BERT_BASE 相同的预训练数据、微调方案和超参数进行实验，研究了 BERT 深度双向性的重要性。

结果显示：

- 移除下一句预测任务会显著降低在 QNLI、MNLI 和 SQuAD 1.1 等任务上的性能，且从左到右模型在所有任务上表现均比掩码语言模型差，尤其在 MRPC 和 SQuAD 上下降幅度大。
    
- 在 SQuAD 任务中为加强从左到右模型添加随机初始化的双向长短期记忆网络虽有效果但仍远不如预训练双向模型且损害了 GLUE 任务性能。
    
- 此外还探讨了像 ELMo 那样分别训练左右向模型并连接的方式，指出其存在成本高、对问答任务不直观且不如深度双向模型强大等问题。
```
### 5.2 Effect of Model Size

In this section, we explore the effect of model size on fine-tuning task accuracy. We trained a number of BERT models with a differing number of **layers, hidden units, and attention heads,** while otherwise using the same hyperparameters and training procedure as described previously.
在本节中，我们探究模型规模对微调任务准确率的影响。我们训练了多个BERT模型，这些模型的层数、隐藏单元数量以及注意力头的数量各不相同，而其他方面则采用与前文所述相同的超参数和训练流程。

Results on selected GLUE tasks are shown in Table 6. In this table, we report the average Dev Set accuracy from 5 random restarts of fine-tuning. We can see that larger models lead to a strict accuracy improvement across all four datasets, even for MRPC which only has 3,600 labeled training examples, and is substantially different from the pre-training tasks. It is also perhaps surprising that we are able to achieve such significant improvements on top of models which are already quite large relative to the existing literature. For example, the largest Transformer explored in Vaswani et al.(2017) is( $L=6$ , $H=1024$ , $A=16$ ) with 100M parameters for the encoder, and the largest Transformer we have found in the literature is $L=64$ , $H=512$ , $A=2$ ) with 235M parameters (AI-Rfou et al., 2018).By contrast, $BERT _{BASE }$ contains 110M parameters and $BERT _{LARGE }$ contains 340M parameters.

所选 GLUE 任务的结果展示在表 6 中。在该表中，我们报告了**微调 5 次随机重启后的dev set平均准确率**。可以看出，更大的模型在所有四个数据集上都带来了显著的准确率提升，即使对于仅有 3600 个标注训练样本且与预训练任务有很大差异的 MRPC 数据集也是如此。同样令人惊讶的是，**在相对于现有文献已经相当大的模型基础上，我们还能实现如此显著的改进**。例如，Vaswani 等人（2017）所研究的最大的 Transformer 是（L = 6，H = 1024，A = 16），其编码器有 1 亿个参数，而我们在文献中找到的最大的 Transformer 是（L = 64，H = 512，A = 2），有 2.35 亿个参数（AI - Rfou 等人，2018）。相比之下，BERT_{BASE} 包含 1.1 亿个参数，BERT_{LARGE} 包含 3.4 亿个参数。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250115235517.png)

Table 6: Ablation over BERT model size. # L = the number of layers; # H = hidden size; # A = number of attention heads. “LM (ppl)” is the masked LM perplexity of held-out training data.
表6：BERT模型规模的消融实验。#L = 层数；#H = 隐藏层大小；#A = 注意力头的数量。“LM (ppl)” 是留出的训练数据的掩码语言模型困惑度。

It has long been known that increasing the model size will lead to continual improvements on large-scale tasks such as machine translation and language modeling, which is demonstrated by the LM perplexity of held-out training data shown in Table 6. However, we believe that this is the first work to demonstrate convincingly that scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained. Peters et al. (2018b) presented mixed results on the downstream task impact of increasing the pre-trained bi-LM size from two to four layers and Melamud et al. (2016) mentioned in passing that increasing hidden dimension size from 200 to 600 helped, but increasing further to 1,000 did not bring further improvements. Both of these prior works used a featurebased approach — we hypothesize that when the model is fine-tuned directly on the downstream tasks and uses only a very small number of randomly initialized additional parameters, the taskspecific models can benefit from the larger, more expressive pre-trained representations even when downstream task data is very small.

长期以来人们都知道，增加模型规模会在诸如机器翻译和语言建模等大规模任务上带来持续的改进，表 6 中留出的训练数据(held-out training data)的语言模型困惑度就证明了这一点。然而，我们认为这是第一项令人信服地证明了**只要模型经过充分的预训练，将模型扩展到极大规模也能在非常小规模的任务上带来大幅改进的工作。**Peters 等人（2018b）在将预训练的双向语言模型从两层增加到四层对下游任务的影响方面呈现出了复杂的结果，Melamud 等人（2016）顺便提到将隐藏维度从 200 增加到 600 有帮助，但进一步增加到 1000 并没有带来进一步的改进。这两项先前的工作都使用了**基于特征的方法**——**我们假设当模型直接在下游任务上进行微调并且只使用极少量随机初始化的额外参数时，即使下游任务数据非常少，特定任务模型也能从更大、更具表现力的预训练表示中受益。**

```ad-note
本文主要探究了 BERT 模型规模对微调任务准确性的影响。通过训练不同层数、隐藏单元数量和注意力头数量的 BERT 模型，使用相同超参数和训练流程，在 GLUE 任务上进行实验。

- 结果显示更大的模型在所有测试数据集上都能提升准确率，甚至在样本量少且与预训练任务差异大的 MRPC 数据集上也如此，且在已有较大规模模型基础上继续扩大规模仍有显著改进，这由留出训练数据的语言模型困惑度证明。
    
- 此前研究在增加模型规模对下游任务影响上结果不一，本文认为只要模型充分预训练，扩展到极大规模对小规模任务也有很大提升，**还假设在直接微调且使用少量随机初始化额外参数时，即使下游任务数据少，特定任务模型也能从更大更具表现力的预训练表示中受益，且之前相关工作多采用基于特征的方法。**
```

### 5.3 Feature-based Approach with BERT
All of the BERT results presented so far have used the fine-tuning approach, where a simple classification layer is added to the pre-trained model, and all parameters are jointly fine-tuned on a downstream task. However, the feature-based approach, where fixed features are extracted from the pretrained model, has certain advantages. First, not all tasks can be easily represented by a Transformer encoder architecture, and therefore require a task-specific model architecture to be added. Second, there are major computational benefits to pre-compute an expensive representation of the training data once and then run many experiments with cheaper models on top of this representation.

到目前为止所展示的所有 BERT 结果都使用了微调方法，即向预训练模型添加一个简单的分类层，并在下游任务上联合微调所有参数。然而，基于特征的方法（**从预训练模型中提取固定特征**）具有一定的优势。首先，并非所有任务都能轻易地由 Transformer 编码器架构表示，因此需要添加特定任务的模型架构。其次，**预先计算一次训练数据的昂贵表示，然后在此表示之上使用更简单的模型进行许多实验，在计算上有很大的益处**。

In this section, we compare the two approaches by applying BERT to the CoNLL-2003 Named Entity Recognition (NER) task (Tjong Kim Sang and De Meulder, 2003). In the input to BERT, we use a case-preserving WordPiece model, and we include the maximal document context provided by the data. Following standard practice, we formulate this as a tagging task but do not use a CRF layer in the output. We use the representation of the first sub-token as the input to the token-level classifier over the NER label set.

在本节中，我们通过将 BERT 应用于 CoNLL - 2003 命名实体识别（NER）任务（Tjong Kim Sang 和 De Meulder，2003）来比较这两种方法。在 BERT 的输入中，我们使用保留大小写的 WordPiece 模型，并包含数据提供的最大文档上下文。按照标准做法，我们将此任务表述为标记任务，但**在输出中不使用条件随机场（CRF）层。** 我们使用**第一个子词的表示作为命名实体识别标签集上的词元级分类器的输入。**

To ablate the fine-tuning approach, we apply the feature-based approach by extracting the activations from one or more layers without fine-tuning any parameters of BERT. These contextual embeddings are used as input to a randomly initialized two-layer 768-dimensional BiLSTM before the classification layer. Results are presented in Table 7 $BERT _{LARGE }$ performs competitively with state-of-the-art methods. The best performing method concatenates the token representations from the top four hidden layers of the pre-trained Transformer, which is only 0.3 F1 behind fine-tuning the entire model. This demonstrates that BERT is effective for both finetuning and feature-based approaches.

**为了去除微调方法的影响，我们采用基于特征的方法，从一个或多个层中提取激活值，而不微调 BERT 的任何参数。** 这些上下文嵌入在分类层之前被用作随机初始化的两层 768 维双向长短期记忆网络（BiLSTM）的输入。 结果展示在表 7 中。BERT_{LARGE} 与最先进的方法相比具有竞争力。表现最佳的方法是将预训练 Transformer 顶部四个隐藏层的词元表示进行连接，其 F1 值仅比微调整个模型低 0.3。**这表明 BERT 对于微调方法和基于特征的方法都是有效的。**

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250115235538.png)
Table 7: CoNLL-2003 Named Entity Recognition results. Hyperparameters were selected using the Dev set. The reported Dev and Test scores are averaged over 5 random restarts using those hyperparameters.
表 7：CoNLL - 2003 命名实体识别结果。超参数是使用开发集选择的。**所报告的开发集和测试集分数是使用这些超参数进行 5 次随机重启后的平均值。**

```ad-note
- 该部分主要探讨了BERT的两种应用方法：微调方法与基于特征的方法，并通过CoNLL - 2003命名实体识别（NER）任务进行对比。
    
- **微调方法是在预训练模型上加简单分类层并联合微调所有参数；基于特征的方法则是从预训练模型提取固定特征**，它有两大优势，一是能适配Transformer编码器难以表示的任务，二是计算上更具效益。
    
- 在NER(命名实体识别)任务实验中，输入采用保留大小写的WordPiece模型及最大文档上下文，将任务设为标记任务且输出不用CRF层，以首个子词表示作为词元级分类器输入。为突出基于特征方法的效果，
    
    - 实验时从BERT的一层或多层提取激活值，不经微调直接输入随机初始化的两层768维BiLSTM，再接入分类层。
        
    - 结果表明，BERT_{LARGE}表现与当前最优方法相当，其中将预训练Transformer顶部四层隐藏层词元表示拼接的方法效果最佳，其F1值仅比微调整个模型低0.3，
        
- **证明BERT对微调与基于特征这两种方法均有效**。表7展示了实验结果，分数为使用开发集选定超参数后，经5次随机重启的平均开发集和测试集分数。
```

## 6 Conclusion

Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems. In particular, these results enable even low-resource tasks to benefit from deep unidirectional architectures. Our major contribution is further generalizing these findings to deep bidirectional architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.

由于语言模型的迁移学习而在近期取得的实证性改进表明，丰富的无监督预训练是许多语言理解系统不可或缺的一部分。特别是，**这些结果使得即使是低资源任务也能从深度单向架构中受益。** 我们的**主要贡献是进一步将这些发现推广到深度双向架构，使相同的预训练模型能够成功应对广泛的自然语言处理任务。**