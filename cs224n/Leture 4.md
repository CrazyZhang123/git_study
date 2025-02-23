#### Verb Phrase (VP) attachment ambiguity 动词短语（VP）附着歧义

![image-20250208143433136](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250208143525089.png)

Mutilated body washes up on Rio beach to be used for Olympics beach volleyball 

残缺的尸体被冲到里约海滩，（该尸体）将被用于奥运会沙滩排球项目（此处语义不通，是因歧义导致）

 解释 1. **示例分析**：对于句子“Mutilated body washes up on Rio beach to be used for Olympics beach volleyball”，动词短语“to be used for Olympics beach volleyball”的附着对象存在歧义。    

- 从字面看，一种理解是“残缺的尸体被冲到里约海滩，（尸体）将被用于奥运会沙滩排球项目”，但这种理解从常识上显然不合理。    
- 更合理的推测是“to be used for Olympics beach volleyball”本应修饰“Rio beach”，即“被冲到将用于奥运会沙滩排球项目的里约海滩上”，但由于句子结构安排，使得动词短语的附着关系出现歧义，造成理解上的混乱。这体现了在自然语言处理中准确判断动词短语附着关系对正确理解句子语义的重要性。 

#### Dependency paths help extract semantic interpretation 依存路径有助于提取语义解释

![img](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250208144412889.png)

- simple practical example: extracting protein - protein interactions  简单实际示例：提取蛋白质 - 蛋白质相互作用

KaiC ←nsubj interacts nmod:with → SasA 

KaiC ←nsubj interacts nmod:with → SasA conj:and → KaiA 

KaiC ←nsubj interacts nmod:with → SasA conj:and → KaiB 

[Erkan et al. EMNLP 07, Fundel et al. 2007, etc.] 

翻译  

​												“demonstrated（证明）”  

​								主语（nsubj）                         补足语（ccomp） “

​					results（结果）”        “mark（引导词）”   “interacts（相互作用）”   与……的关系（nmod:with）

​					限定词（det）          “that（那）”               状语（advmod）                              “SasA（蛋白质名称）”    

​														主语（nsubj）   格标记（case）  并列连词（conj:and）    

​				“the（这个）”        KaiC  “rhythmically（有节奏地）”    “with（和）” “KaiA（蛋白质名称）” “and（和）” “KaiB（蛋白质名称）” 

KaiC ←主语（nsubj） 相互作用（interacts） 与……的关系（nmod:with） → SasA 

KaiC ←主语（nsubj） 相互作用（interacts） 与……的关系（nmod:with） → SasA 并列连词（conj:and） → KaiA 

KaiC ←主语（nsubj） 相互作用（interacts） 与……的关系（nmod:with） → SasA 并列连词（conj:and） → KaiB 

[埃尔坎等人，EMNLP 2007年会议，芬德尔等人，2007年，等等] 

解释 

1. **核心观点**：表明依存路径在提取语义解释方面有帮助，以提取蛋白质 - 蛋白质相互作用为例进行说明。 
2. **依存关系分析**：    
   - 句子“The results demonstrated that KaiC interacts rhythmically with SasA, KaiA and KaiB”通过依存句法分析展示各词间关系。“results”是“demonstrated”的主语（nsubj），“that”引导从句，“KaiC”是“interacts”的主语（nsubj） ，“SasA” “KaiA” “KaiB” 通过“with”和“and”等与“interacts”建立关系。    
   -  从依存路径角度，如“KaiC ←nsubj interacts nmod:with → SasA” 等路径明确呈现出蛋白质KaiC与SasA、KaiA、KaiB之间存在相互作用关系。这说明通过分析依存路径，能够清晰地从句子中提取出蛋白质 - 蛋白质相互作用这类语义信息，在自然语言处理应用于生物医学等领域时，有助于从文本中挖掘相关知识。 



#### Dependency Grammar and Dependency Structure Dependency 依存语法与依存结构

![image-20250208145802997](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250208145803152.png)

syntax postulates that syntactic structure consists of relations between lexical items, normally binary asymmetric relations (“arrows”) called dependencies 

An arrow connects a head (governor, superior, regent) with a dependent (modifier, inferior, subordinate) 

Usually, dependencies form a tree (a connected, acyclic, single - root graph) ### 中文翻译  依存句法假定句法结构由词汇项之间的关系组成，通常是被称为依存关系的二元不对称关系（“箭头”） 一个箭头将一个中心词（支配词、上级词、主导词）与一个依存词（修饰词、下级词、从属词）连接起来 通常，依存关系形成一棵树（一个连通的、无环的、单根图） ### 解释 1. **依存语法基本概念**：依存语法认为句法结构是由词汇项之间的关系构成的，这些关系一般是二元不对称的，用“箭头”来表示，被称作依存关系。 2. **中心词与依存词关系**：在依存关系中，箭头连接的两端分别是中心词和依存词。中心词也叫支配词、上级词或主导词，依存词也叫修饰词、下级词或从属词，表明依存词对中心词存在修饰、限定等关系。 3. **依存结构特点**：通常情况下，这些依存关系会构成一棵树状结构，它具有连通性（所有节点都相互连接）、无环性（不存在回路）以及单根性（只有一个根节点） 。例如在给出的句子“Bills were submitted by Brownback on ports and immigration”的依存结构分析中，“submitted”是根节点（中心词），“Bills”是它的主语（nsubj:pass），“Brownback”是施动者（obl)等，各词汇项通过依存关系构成了一个清晰的树状结构来反映句子的句法关系。 































功能配置20241122149



