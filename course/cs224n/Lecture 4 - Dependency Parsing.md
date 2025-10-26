---
created: 2025-02-01T00:21
updated: 2025-02-06T23:57
---
第4讲：依赖解析

## Lecture Plan 

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250201150056.png)
Syntactic Structure and Dependency parsing
句法结构与依存句法分析
1. Syntactic Structure: Consistency and Dependency (25 mins)  句法结构：一致性与依存关系
2. Dependency Grammar and Treebanks (15 mins)  依存语法与树库
3. Transition - based dependency parsing (15 mins) 基于转移的依存句法分析
4. Neural dependency parsing (20 mins) 神经依存句法分析

Reminders/comments: 
In Assignment 3, out on Tuesday, you build a neural dependency parser using PyTorch 
Start installing and learning PyTorch (Ass 3 has scaffolding)
Come to the PyTorch tutorial, Friday 10am (under the Zoom tab, not a Webinar)
Final project discussions – come meet with us; focus of Thursday class in week 4 

周二将发布作业3，你需要使用PyTorch构建一个神经依存句法分析器 
开始安装并学习PyTorch（作业3有框架） 
周五上午10点参加PyTorch教程（在Zoom选项卡下，不是网络研讨会） 
最终项目讨论 - 来与我们见面；第4周周四课程的重点 


### 1. Two views of linguistic structure: Constituency = phrase structure grammar = context - free grammars (CFGs)  成分结构 = 短语结构语法 = 上下文无关语法（CFGs）
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206223730.png)

- Phrase structure organizes words into nested constituents  短语结构将单词组织成嵌套的成分
- **Starting unit: words**  起始单位：单词
	- the, cat, cuddly, by, door 
	- Det    N     Adj      P     N  （a/the/this/that/every限定词 名词 形容词 介词 名词）
- **Words combine into phrases** 单词组合成短语
	- the cuddly cat（可爱的猫）, by the door（在门边）
- **Phrases can combine into bigger phrases**  短语可以组合成更大的短语
	- the cuddly cat by the door 
	
1. **语言结构观点**：介绍了语言结构的一种观点，即成分结构，它等同于短语结构语法和上下文无关语法（CFGs）。上下文无关语法是形式语言理论中的概念，用于描述自然语言或编程语言的语法结构。
2. **短语结构构建**：短语结构的构建从单词开始，如“the”（限定词Det）、“cat”（名词N）、“cuddly”（形容词Adj）、“by”（介词P）、“door”（名词N）。这些单词首先组合成短语，例如“the cuddly cat”（“the”和“cuddly”修饰“cat” ）和“by the door”。然后，==这些短语可以进一步组合成更大的短语，像“the cuddly cat by the door” 。这种从单词到短语再到更大短语的嵌套组合方式，体现了短语结构将单词组织成具有层次结构的成分的特点，有助于分析句子的语法构成和语义关系 。==

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206225005.png)
这里介绍了一些CFG(上下文无关文法)，从右边的 **lexicon(词典)**，到中间的**Grammer(文法)**，随着老师的讲解不断再扩充，从NP(名词短语)，PP(介词短语), VP(动词短语),S(句子)，以及他们的文法句子，\*和编译原理一样，代表0或多次，这就是一个简单的文法用于生成S，这里就是举例说明了一下CFG。

### Two views of linguistic structure: Dependency structure 依存结构
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206230311.png)

 - Dependency structure shows which words depend on (modify, attach to, or are arguments of) which other words. 依存结构展示了哪些单词依赖于（修饰、附着于或作为其论元）哪些其他单词。 

Look    **in the large crate**    in the kitchen by the door
看看厨房门边的大箱子
**Look in**     the large crate    in the kitchen by the door
小心厨房门边的大箱子

解释 
1. **依存结构定义**：依存结构是语言结构的一种视图，它关注单词之间的依赖关系。在自然语言中，单词并非孤立存在，而是存在着各种语义和语法上的联系。例如，形容词通常修饰名词，介词短语常常对动词或名词进行补充说明等。 
2. **示例分析**：以“Look in the large crate in the kitchen by the door”这句话为例，从依存结构的角度来看，“in”（介词P）和“the large crate”存在依赖关系，“in”表示位置关系，附着于“the large crate” ；“the large crate” 中“large”修饰“crate” ；“in the kitchen”和“in the large crate” 相关，进一步说明“the large crate” 的位置；“by the door” 又对“the large crate” 进行补充说明。通过这种方式，依存结构清晰地呈现了句子中单词之间的依赖、修饰等关系，有助于深入理解句子的语义和语法结构。

#### Why do we need sentence structure? 为什么需要句子结构
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206230656.png)
- Humans communicate complex ideas by composing words together into bigger units to convey complex meanings  人类通过将单词组合成更大的单位来传达复杂的意义，从而交流复杂的想法。
- Listeners need to work out what modifies \[attaches to] what 听众需要弄清楚什么修饰（附着于）什么。
- A model needs to understand sentence structure in order to be able to interpret language correctly  ==一个模型需要理解句子结构，以便能够正确地解释语言==。

1. **人类交流层面**：人类语言要表达复杂思想，不能仅靠单个单词，而是要将单词组合成短语、句子等更大的语言单位。例如，“我在美丽的公园散步”，“美丽的”修饰“公园”，“在公园”是表示地点的短语，它们共同构成完整的表达，传递复杂语义。
2. **听众理解层面**：当人们听到话语时，需要分析句子结构来明确单词间的修饰或依附关系。比如“红色的苹果”，听众要明白“红色的”是对“苹果”的修饰限定，才能准确理解语义。 
3. **模型应用层面**：在自然语言处理等领域，模型若想正确理解和处理语言，就必须理解句子结构。比如机器翻译模型，只有准确分析源语言句子结构，才能在目标语言中生成恰当、符合语法和语义的翻译。

### Prepositional phrase attachment ambiguity 介词短语附着歧义
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206231202.png)
 San Jose cops(警察) kill man with knife **圣何塞警察用刀杀死男子 / 圣何塞警察杀死带刀的男子**
 Ex - college football player, 23, shot 9 times allegedly charged police at fiancee’s home 23岁前大学橄榄球运动员，据称在未婚妻家中向警察开枪，中枪9次 
 解释 
  **示例分析**：以新闻标题“San Jose cops kill man with knife”为例，介词短语“with knife”存在附着歧义。它既可以理解为警察用刀杀死男子（“with knife”修饰“kill”这个动作，说明杀人的工具），也可以理解为警察杀死了带着刀的男子（“with knife”修饰“man”，描述男子的状态） 。这种歧义在自然语言中较为常见，会给语言的理解和处理带来困难。在自然语言处理中，模型需要准确判断介词短语的附着对象，才能正确理解句子的语义。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206231529.png)
Scientists count whales from space
科学家在太空中数鲸鱼。
**“from space” 是指科学家在太空这个位置来数鲸鱼，而不是鲸鱼在太空，所以海洋中鲸鱼的图片理解正确，太空里鲸鱼的图片理解错误。**

#### PP attachment ambiguities multiply 介词短语附着歧义增多
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206232213.png)
 - A key parsing decision is how we ‘attach’ various constituents **一个关键的句法分析决策是我们如何“附着”各种成分**。
	 - PPs, adverbial or participial phrases, infinitives, coordinations, 介词短语（PPs）、状语或分词短语、不定式、并列结构等
 - The board approved(批准) \[its acquisition(收购)\]
	 - \[by Royal Trustco Ltd.\] 
	 - \[of Toronto\] 
	 - \[for $27 a share\] 
	 - \[at its monthly meeting]
 -  董事会在其月度会议上批准了（由皇家信托公司（位于多伦多）以每股27美元）对其的收购。 
正确的修饰关系
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206232925.png)

 解释 
 1. **示例分析**：以句子“The board approved \[its acquisition\] \[by Royal Trustco Ltd.\] \[of Toronto\] \[for $27 a share\] \[at its monthly meeting]\.”为例，其中多个介词短语存在附着歧义。比如“by Royal Trustco Ltd.” 既可以理解为修饰“acquisition”（表示收购的主体），“of Toronto” 可理解为修饰“Royal Trustco Ltd.”（表示公司所在地点） ；“for $27 a share” 修饰“acquisition”（表示收购的价格） ；“at its monthly meeting” 修饰“approved”（表示批准这一动作发生的场合）。==但这些修饰关系的确定在不同语境下可能会有所不同，这就体现了介词短语附着歧义会随着句子中成分的增多而变得复杂，在自然语言处理中准确分析这些关系对于正确理解句子语义至关重要。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808224558421.png)

- Catalan numbers:  $C_n = (2n)!/\[(n + 1)!n!]$ 
- An exponentially growing series, which arises in many tree - like contexts: - E.g., the number of possible triangulations of a polygon with \( n + 2 \) sides - Turns up in triangulation of probabilistic graphical models (CS228)… 
- **卡特兰数**：  $C_n = (2n)!/\[(n + 1)!n!]$ 
- 一类指数增长的数列，出现在诸多类似树的情境中 
- 例如，一个有 \( n + 2 \) 条边的多边形的可能三角剖分数量 - 也会出现在概率图模型（CS228课程涉及）的三角剖分里…… 
**解释** 
- **卡特兰数公式**：卡特兰数是组合数学中重要数列，\( C_n \) 计算公式为 \( (2n) \) 的阶乘除以 \( (n + 1) \) 的阶乘与 \( n \) 的阶乘的乘积 ，像 \( C_0 = 1 \)，\( C_1 = 1 \)，\( C_2 = 2 \) 等，可用于解决很多组合计数问题。  ==这里实际想说不带()阶乘会引发歧义。==
- **数列特性与应用场景**：它属于指数增长数列，常和树状结构相关场景联系，比如多边形三角剖分（把多边形通过连接对角线分成多个三角形的方式数量，边数为 \( n + 2 \) 时可用卡特兰数计数 ），在概率图模型（如CS228课程里会涉及用其分析图结构三角剖分相关概率、结构计数等）等领域也有应用，体现其在组合结构计数及相关理论、应用学科里的价值 。
#### Coordination scope ambiguity 并列范围歧义
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206233220.png)
 - \[Shuttle **veteran**(老手)] and \[longtime NASA **executive**(高管) Fred Gregory] appointed to board 
 - \[航天飞机老手] 和 \[美国国家航空航天局（NASA）长期高管弗雷德·格雷戈里] 被任命为董事会成员
 - \[Shuttle veteran and longtime NASA executive] Fred Gregory appointed to board 
 -  \[航天飞机老手兼美国国家航空航天局（NASA）长期高管] 弗雷德·格雷戈里被任命为董事会成员
 - 解释 1. **概念**：“Coordination scope ambiguity”指的是并列范围歧义，即在句子中由于并列结构的范围不明确而导致的语义理解上的歧义。 
 1. **示例分析** - 对于“[Shuttle veteran] and [longtime NASA executive Fred Gregory] appointed to board”，从结构上理解是“航天飞机老手”和“美国国家航空航天局长期高管弗雷德·格雷戈里”两个人被任命为董事会成员，这里“and”连接的是两个不同的个体。 - 而“[Shuttle veteran and longtime NASA executive] Fred Gregory appointed to board”，则是将“航天飞机老手兼美国国家航空航天局长期高管”作为对“弗雷德·格雷戈里”的修饰，意思是弗雷德·格雷戈里这一个人被任命为董事会成员 。这种因并列范围界定不同产生的歧义在自然语言处理中需要准确分析句子结构和语义来消除，以确保对句子的正确理解。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206233545.png)
这句话“Doctor: No heart, cognitive issues”存在结构歧义。 
- 从红色标注来看，可理解为“医生表示：**没有心脏方面的问题，但是有认知方面的问题**” ，即把“heart”和“cognitive”分开理解，分别指心脏和认知。
- 从蓝色标注来看，可理解为“医生表示：**没有心脏 - 认知方面的问题**” ，即将“heart”和“cognitive”看作一个整体修饰“issues” ，表示不存在与心脏和认知相关的问题。这体现了自然语言中因结构理解不同产生的歧义现象。

#### Adjectival/Adverbial Modifier Ambiguity 形容词/状语修饰语歧义
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206233908.png)
Students get first hand job experience 
- 学生获得第一手工作经验 
解释 
###  **常规歧义：语义组合的两种解读**

- **解读一**：**"第一份工作经历"**
    - 将 "first hand job experience" 拆解为 "first（第一份） + hand job experience（手工类工作经验）"，但实际更常见的组合是 "first-hand job experience"（直接的工作经验）。
    - 若强行拆分，"hand job" 可能被误读为与“手工工作”相关（如木工、维修），但此用法较少见。
        
- **解读二**：**"直接的工作经验"**
    
    - "First-hand"（亲身的/直接的）作为固定搭配，修饰 "job experience"，强调学生通过实践获得的经验，而非理论学习。
---
### 2. **隐藏歧义：俚语陷阱**
- **潜在风险**：  
    在英语中，"hand job" 是一个俚语，含义与性相关（指用手进行的性行为）。如果读者将 "first hand job experience" 连读为 "first hand-job experience"，可能产生令人尴尬的误解。

#### Verb Phrase (VP) attachment ambiguity 动词短语（VP）附着歧义

![image-20250208143433136](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250208143525089.png)

Mutilated body washes up on Rio beach to be used for Olympics beach volleyball 

残缺的尸体被冲到里约海滩，（该尸体）将被用于奥运会沙滩排球项目（此处语义不通，是因歧义导致）

 解释 1. **示例分析**：对于句子“Mutilated body washes up on Rio beach to be used for Olympics beach volleyball”，动词短语“to be used for Olympics beach volleyball”的附着对象存在歧义。    

- 从字面看，一种理解是“残缺的尸体被冲到里约海滩，（尸体）将被用于奥运会沙滩排球项目”，但这种理解从常识上显然不合理。    
- **更合理的推测是“to be used for Olympics beach volleyball”本应修饰“Rio beach” ，即“被冲到将用于奥运会沙滩排球项目的里约海滩上”**，但由于句子结构安排，使得动词短语的附着关系出现歧义，造成理解上的混乱。这体现了在自然语言处理中准确判断动词短语附着关系对正确理解句子语义的重要性。 

#### Dependency paths help extract semantic interpretation 依存路径有助于提取语义解释

![img](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250208144412889.png)

- simple practical example: extracting protein - protein interactions  **简单实际示例：提取蛋白质 - 蛋白质相互作用**

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



#### 2 Dependency Grammar and Dependency Structure Dependency 依存语法与依存结构

![image-20250208145802997](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250208145803152.png)

- Dependency syntax postulates that syntactic structure consists of relations between lexical items, normally binary asymmetric relations (“arrows”) called dependencies 
**依存句法假定，句法结构由词汇项之间的关系构成，通常是被称为 “依存关系” 的二元非对称关系（“箭头”）** 。
- An arrow connects a head (governor, superior, regent) with a dependent (modifier, inferior, subordinate) 
==一个箭头连接一个中心词（主导词、上位词、支配词 ，如 “head” 相关释义 ）和一个依存词（修饰词、从属词、下位词 ，如 “dependent” 相关释义 ）== 。
- Usually, dependencies form a tree (a connected, acyclic, single - root graph) 
- 通常，依存关系构成一棵树（一个连通、无环、单根的图 ） 。

图中示例里，
- 像 “submitted（提交）” 与 “Bills（议案等，结合语境 ）” 有 “nsubj:pass（被动主谓关系 ）” 依存关系；
- “submitted” 和 “were” 是 “aux（助动词 ）” 依存关系 ；
- “Bills” 与 “ports” 是 “nmod（名词修饰 ）” 关系 ，“ports” 和 “on” 是 “case（格标记 ）” 关系 ，“ports” 与 “and” 是 “cc（并列连词 ）” 关系 ，“ports” 和 “immigration” 是 “conj（并列 ）” 关系 ；
- “submitted” 与 “Brownback” 是 “obl（ oblique，间接宾格等关系 ）” 关系 ；“Brownback” 与 “by” 是 “case” 关系 ，“Brownback” 和 “Senator” 是 “flat（平层，常指名称等平级组合 ）” 关系 ，“Brownback” 与 “Republican” 是 “appos（同位语 ）” 关系 ；
- “Republican” 与 “Kansas” 是 “nmod” 关系 ；“Kansas” 与 “Stanford” 是 “case” 关系 ，“Stanford” 与 “of” 是 “case” 关系 等 ，因图里示例是具体语句的依存分析，需结合语法分析场景理解 。

**解释** 
1. **依存语法基本概念**：依存语法认为句法结构是由词汇项之间的关系构成的，这些关系一般是二元不对称的，用“箭头”来表示，被称作依存关系。
2. **中心词与依存词关系**：在依存关系中，箭头连接的两端分别是中心词和依存词。中心词也叫支配词、上级词或主导词，依存词也叫修饰词、下级词或从属词，表明依存词对中心词存在修饰、限定等关系。 
3. **依存结构特点**：通常情况下，这些依存关系会构成一棵树状结构，它具有连通性（所有节点都相互连接）、无环性（不存在回路）以及单根性（只有一个根节点） 。例如在给出的句子“Bills were submitted by Brownback on ports and immigration”的依存结构分析中，“submitted”是根节点（中心词），“Bills”是它的主语（nsubj:pass），“Brownback”是施动者（obl)等，各词汇项通过依存关系构成了一个清晰的树状结构来反映句子的句法关系。 

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808230343413.png)
Panini对梵文(**Sanskrit**: 梵语)进行了研究，说明Dependency Grammar依存语法具有悠久的历史，丰富的词形变化，和句子结构。

#### Dependency Grammar/Parsing History 依存语法/剖析历史
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808230657770.png)
- The idea of dependency structure goes back a long way
	- To Pānini’s grammar (c. 5th century BCE) 
	- Basic approach of 1st millennium Arabic grammarians 
- **Constituency/context-free grammar is a new-fangled invention** 
	- 20th century invention (R.S. Wells, 1947; then Chomsky 1953, etc.) 
- Modern dependency work is often sourced to Lucien Tesnière (1959) 
	- Was dominant approach in “East” in 20th Century (Russia, China,...) 
	- Good for free - er word order, inflected languages like Russian (or Latin!) 
- Used in some of the earliest parsers in NLP, even in the US: 
	- David Hays, one of the founders of U.S. computational linguistics, built early (first?) dependency parser (Hays 1962) and published on dependency grammar in Language
**翻译**  
- **依存结构的理念由来已久** 
	- 可追溯至巴尼尼语法（约公元前5世纪） 
	- 公元1千年时阿拉伯语法学家的基本研究方法 
- **组块/上下文无关语法是较新出现的概念** 
	- 20世纪的发明（R.S. 威尔斯，1947年；而后诺姆·乔姆斯基在1953年等进一步发展 ） 
	- 现代依存语法研究通常追溯到 Lucien Tesnière（1959年） 
		- 20世纪在“东方”（俄罗斯、中国…… ）是主流研究方法 
- ==适用于语序较灵活、有词形变化的语言，如俄语（或拉丁语 ）== 
	- 即便在美国，也被用于自然语言处理中一些最早的剖析器： 
	- 大卫·海斯，美国计算语言学创始人之一，开发了早期（首台？ ）依存剖析器（海斯，1962年 ），并在《语言》期刊上发表了关于依存语法的研究 
**解释** 这段内容梳理了依存语法相关的历史脉络： 
- **起源古老**：依存结构思想很早就有，公元前5世纪印度巴尼尼语法就有相关影子，公元1千年阿拉伯语法学家也用类似思路，说明其在语法研究里根基深厚，从古代语法分析实践中发展而来 。 
- **与其他语法对比**：组块/上下文无关语法是20世纪才兴起的，突出依存语法在时间维度上的“前辈” 地位，体现语法理论发展的不同阶段 ，像威尔斯、乔姆斯基对后者的推动，反映现代形式语法发展轨迹 。 
- **现代传承与地域应用**：现代依存语法常以 Tesnière 1959年成果为源头，且20世纪在俄罗斯、中国等东方国家是主流，因这些地区语言（如俄语有丰富词形变化、语序相对灵活 ）特点，适配依存语法分析优势（关注词汇依存，不严格依赖固定语序 ），拉丁语同理 。 
- **在自然语言处理（NLP）里的应用**：美国计算语言学发展中，海斯早在1962年就开发依存剖析器、发表相关研究，说明依存语法从理论到实际NLP工具构建的落地，是自然语言处理早期技术探索的一部分，体现其跨地域、跨理论实践的价值 ，也见证语法理论和计算机技术结合的进程 。

#### Dependency Grammar and Dependency Structure 依存语法与依存结构
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808231433174.png)
 ROOT Discussion of the outstanding issues was completed . 
- Some people draw the arrows one way; some the other way! 
	- Tesnière had them point from head to dependent – we follow that convention 
- We usually add a fake ROOT so every word is a dependent of precisely 1 other node 
翻译 
-  ROOT 对突出问题的讨论已完成 。 
- 有些人箭头画成一个方向；有些人画成另一个方向！ 
	- 特斯尼埃（Tesnière）**让箭头从中心词指向依存词**——我们遵循这个惯例 
- ==我们通常添加一个虚拟的 ROOT，这样每个词恰好是另一个节点的依存词== 
**解释** - **标题与示例语句**：“Dependency Grammar and Dependency Structure” 是关于依存语法和依存结构的内容，下方语句 “ROOT Discussion of the outstanding issues was completed.” 是用于展示依存结构分析的示例，“ROOT” 是为构建依存树添加的虚拟根节点 。 
- **箭头方向**：说明在依存语法表示依存关系时，箭头方向存在不同画法习惯，但遵循特斯尼埃的做法，即从中心词（head，在句法结构中起支配作用的词 ）指向依存词（dependent，受中心词支配的词 ） ，这是一种约定俗成的绘图规范，方便统一分析和交流 。
- **虚拟 ROOT 作用**：添加虚拟 “ROOT” 节点，目的是让语句中每个词都能成为恰好一个其他节点的依存词，构建出连通、单根（以 ROOT 为根 ）的依存树结构 ，符合依存结构常呈现为树状（连通、无环、单根 ）的特点，保证句法分析结构的完整性和规范性 ，让整个语句词汇的依存关系能清晰、系统地用树结构呈现 ，便于分析词汇间的支配 - 从属关系 。

#### The rise of annotated data & Universal Dependencies treebanks 标注数据的兴起与通用依存树库
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808231855038.png)

 - **语料库相关**： 
	 - Brown corpus (1967; PoS tagged 1979) ：布朗语料库（1967年创建；1979年完成词性标注 ） ，是自然语言处理领域经典英语语料库，为语言分析等提供基础数据 。
	 - Lancaster-IBM Treebank (starting late 1980s) ：兰卡斯特 - 国际商业机器公司树库（20世纪80年代后期开始构建 ） ，用于句法等分析的语料资源 。 
	 - Marcus et al. 1993, The Penn Treebank, Computational Linguistics ：马库斯等人1993年（相关成果 ）、宾州树库、《计算语言学》（期刊 ） ，宾州树库是极具影响力的句法树库，对句法分析研究意义重大 。 - Universal Dependencies: http://universaldependencies.org/ ：通用依存（项目 ），网址为 http://universaldependencies.org/ ，致力于提供跨语言统一依存标注标准和树库资源 。 
 - **示例树库内容**：图中展示了带有依存关系标注的语句分析，像 “Miramar was a famous goat trainer or something” 、“Why is the city called Miramar?” 、“Do you think there are any koreans in Miramar?” 等语句的依存结构，通过标注 “PRON（代词 ）”“VERB（动词 ）”“PROPN（专有名词 ）”“DET（限定词 ）”“ADJ（形容词 ）”“NOUN（名词 ）”“CCONJ（并列连词 ）”“PUNCT（标点 ）” 等词性及依存关系（如 “nsubj” 主语、“cop” 系动词、“det” 限定词关系等 ），呈现语句的句法依存树结构 ，用于展示通用依存树库的标注方式和分析样例 ，辅助理解标注数据和依存树库的实际形态 。

### The rise of annotated data 注释数据的兴起
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808232400186.png)
- Starting off, building a treebank seems a lot slower and less useful than writing a grammar (by hand) 
一开始，**构建树库似乎比（手动）编写语法要慢得多，而且用处也小**。
- But a treebank gives us many things 
	- Reusability of the labor  ==劳动成果的可复用性==
		- Many parsers, part-of-speech taggers, etc. can be built on it 
			许多句法分析器、词性标注器等都可以基于它来构建
		- Valuable resource for linguistics 
			对语言学研究而言是宝贵资源
	- Broad coverage, not just a few intuitions 
		==广泛的覆盖性，而非仅依赖零散的直觉==
	- Frequencies and distributional information 
		==（语言单位的）频率与分布信息==
	- A way to evaluate NLP systems
	    ==评估自然语言处理（NLP）系统的一种方式==
**解释** 
这段内容聚焦**标注数据（以树库为代表）的价值**，分两部分理解： 
1. **初期认知对比**： 开篇提到“构建树库初期看起来不如手动写语法高效、有用”，反映早期对树库价值的直观感受——手动写语法直接制定规则，树库需耗费大量人力标注数据，初期“投入产出比”看似低。
2. **树库的核心价值**： 用“but”转折，详细阐述树库的实际意义： 
	- **可复用性**：树库是标注好的语言数据资源，能成为句法分析器、词性标注器等工具的“基石”，避免重复标注劳动；也为语言学研究（如句法规律挖掘、语言演变分析）提供真实语料支撑。 
	- **覆盖性**：基于大规模真实语料构建，能覆盖多样语言现象（如复杂句式、罕见用法），不像“手动写语法”易受个人经验（直觉）局限，更贴近真实语言使用。 
	- **频率与分布信息**：树库记录词汇、结构的出现频率和语境分布，为自然语言处理（NLP）模型训练（如统计语言模型）、语言规律总结（如哪些搭配更常见）提供数据依据。 
	- **评估作用**：可作为“基准测试集”，将NLP系统的输出（如句法分析结果）与树库标注对比，判断系统性能优劣，推动技术迭代。 简言之，树库通过“数据标注”沉淀语言规律，从“长期价值”上弥补了初期构建的成本，是推动NLP发展和语言学研究的关键基础设施。

#### Dependency Conditioning Preferences 依存分析的制约偏好
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808232709612.png)
 What are the sources of information for dependency parsing? 
 依存分析的信息来源有哪些？
 1. Bilexical affinities 　　The dependency [discussion → issues] is plausible 
	1. **双词亲和性** ：依存关系 [discussion（讨论）→ issues（问题）] 是合理的 
 2. Dependency distance 　 Most dependencies are between nearby words 
	 1. **依存距离** ：大多数依存关系存在于相邻词语之间
 3. Intervening material 　 Dependencies rarely span intervening verbs or punctuation 
	 1. **插入内容** ：依存关系很少会跨越插入的动词或标点符号
 4. Valency of heads 　　　How many dependents on which side are usual for a head? 
	 **中心词配价** ：对于一个中心词来说，通常在哪些位置有多少个依存词？
 ROOT Discussion of the outstanding issues was completed .
 ROOT 对突出问题的讨论已完成 。 
 **解释** 这段内容围绕**依存句法分析（dependency parsing）的信息依据**展开，是自然语言处理中句法分析领域的知识： 
 - **标题与问题**：“Dependency Conditioning Preferences” 指依存分析时，影响依存关系判断的一些偏好（即倾向依据哪些信息确定依存关系 ），开篇问题 “What are the sources of information for dependency parsing?” 引出依存分析的信息来源探讨 。 
 - **四个信息来源**： -
	 - **双词亲和性**：指词语之间本身存在的语义、句法关联，像 “discussion（讨论 ）” 和 “issues（问题 ）” 语义上常搭配，所以认为它们之间构建依存关系是合理的，即词语固有==搭配习惯==会影响依存分析 。 
	 - **依存距离**：说明在实际语言中，大多数依存关系发生在位置相近的词语间，距离远的词语形成依存关系的概率低，分析时会优先考虑相邻或近邻词语的依存可能 。 
	 - **插入内容**：若语句中有插入的动词、标点等，依存关系一般不会跨越它们，也就是这些插入成分会分割依存关系，分析时需考虑其对依存连接的阻断作用 。 
	 - **中心词配价**：==“配价（valency ）” 源于语言学概念，指中心词（head ）在句法结构中能支配的依存成分的数量和位置等属性 ，比如某个动词作中心词时，通常能带几个宾语、状语等，位置在左边还是右边，了解这些常规配价情况有助于分析依存关系 。== 

#### Dependency Parsing 依存句法分析
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808233518592.png)
- A sentence is parsed by choosing for each word what other word (including ROOT) it is a dependent of  **分析一个句子时，要为每个词选定它所依存的另一个词（包括 ROOT 节点 ）**
- Usually some constraints:   通常有这些==约束==
	- Only one word is a dependent of ROOT  ==只有一个词依存于 ROOT 节点==
	- Don’t want cycles A → B, B → A  ==不允许出现循环（如 A 依存于 B，同时 B 依存于 A ）==
- This makes the dependencies a tree  ==这使得依存关系构成一棵树==
- Final issue is whether arrows can cross (be non-projective) or not ==最后一个问题是箭头是否可以交叉（即构成“非投射性”结构 ）==
- （下方示例：“ROOT I ’ll give a talk tomorrow on neural networks” 及依存箭头图示 ） 
**解释** 这段内容是**自然语言处理中“依存句法分析（Dependency Parsing）”的基础理论**，核心讲依存分析的规则和特性：
1. **基本操作**： 依存分析的核心是给句子中每个词找“父节点”（即它依存的词 ），甚至引入虚拟的 “ROOT” 节点作为整句依存树的根，让所有词最终关联到 ROOT 。
2. **约束条件**： 
	- **单 ROOT 依存**：保证整棵依存树有唯一“根”，让结构清晰、无歧义。 
	- **无循环**：禁止 A←→B 这类循环依存，否则句法结构会混乱，无法形成“树”（树的本质是无环连通图 ）。 
3. **结构特性**： 满足上述约束后，依存关系会自然构成**树结构**（连通、无环、单根 ），这是依存分析的典型形态。 
4. **进阶问题**： ==提到“箭头是否交叉（非投射性，non-projective ）”，是依存分析的深入话题——严格的“投射性”要求依存箭头不交叉，符合常规句法直觉；但部分语言（如带长距离依存的语种 ）会出现交叉，因此需要讨论“非投射性”是否合理，这也是句法分析理论的拓展点==。 下方的例句和箭头图示，是用直观方式展示依存树的构建（如 “I ’ll give a talk tomorrow on neural networks” 的依存关系 ），辅助理解上述规则如何落地。 简言之，这段内容把“依存句法分析”的核心逻辑（找依存、守约束、成树结构 ）讲清楚了，是理解自然语言句法分析的基础框架。

#### Projectivity 投射性
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808234134372.png)
 - Definition of a projective parse: There are no crossing dependency arcs when the words are laid out in their linear order, with all arcs above the words 
	 **投射性剖析的定义**：当词语按线性顺序排列、且所有弧（依存关系线 ）置于词语上方时，==不存在交叉的依存弧==
 - Dependencies corresponding to a CFG tree must be projective - I.e., by forming  ==与上下文无关文法（CFG）树对应的依存关系必须是投射性的==
	 - dependencies by taking 1 child of each category as head 
	==即，通过将每个语法范畴的一个子节点作为中心词来构建依存关系==
 - Most syntactic structure is projective like this, but dependency theory normally does allow non-projective structures to account for displaced constituents 、
	 **大多数句法结构都是这样的投射性结构**，但依存语法理论通常也允许非投射性结构，以解释移位的句法成分
	 - You can’t easily get the semantics of certain constructions right without these nonprojective dependencies ==要是没有这些非投射性依存关系，某些句法结构的语义就很难准确解读==
 （还有示例句子 “Who did Bill buy the coffee from yesterday?” 及依存关系图 ）（示例句子：“比尔昨天从谁那里买的咖啡？” 及对应的依存关系图 ）、
**解释** 
  - **标题与核心概念**：“Projectivity” 是句法分析里的“投射性”概念，关乎依存句法分析中依存关系的结构形态 。 
  - **投射性剖析定义**：说明在投射性剖析中，词语线性排列时，依存关系的弧不能交叉，且弧画在词语上方（是一种可视化约定 ），保证句法结构呈现清晰、无交叉干扰的形态 ，符合常规对句法结构“层次分明”的认知 。 -
  - **与 CFG 树关联**：上下文无关文法（CFG）构建的句法树，其对应的依存关系得是投射性的 ，构建方式是每个语法范畴（如短语、词类等语法分类 ）选一个子节点当中心词（head ）来连依存关系，体现了不同语法理论（CFG 和依存语法 ）在结构上的关联与约束 。 
  - **非投射性的必要性**：现实语言中多数句法结构是投射性的，但也有成分移位情况（如疑问句里疑问词移位 ），此时非投射性结构能合理呈现这种移位后的依存关系 ，若不允许非投射性，这类语句（像示例 “Who did Bill buy the coffee from yesterday?” 中 “Who” 移位 ）的语义就难以通过依存关系准确解析 ，体现理论对实际语言现象的适配性 。 
  - **示例作用**：下方示例句子及依存图，直观展示投射性（或涉及非投射性，因有疑问词移位 ）的依存结构 ，辅助理解投射性和非投射性在实际语句分析中的体现 ，让抽象理论和语言实例结合 。 整体是依存语法中关于结构形态（投射性 vs 非投射性 ）的基础理论内容，解释了句法分析里结构约束和对语言现实的适配方式 。

#### 3. Methods of Dependency Parsing  依存句法分析方法
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808234608402.png)1. **Dynamic programming**  动态规划法 
	Eisner (1996) gives a clever algorithm with complexity \( O(n^3) \), by producing parse items with heads at the ends rather than in the middle 
	Eisner（1996）提出一种巧妙算法，时间复杂度为 \( O(n^3) \) ，其特点是生成的剖析项中，中心词（head）位于两端而非中间 
2. **Graph algorithms** 图算法 
	You create a Minimum Spanning Tree for a sentence McDonald et al.’s (2005) MSTParser scores dependencies independently using an ML classifier (he uses MIRA, for online learning, but it can be something else) 
	为**句子构建最小生成树**（Minimum Spanning Tree） McDonald 等人（2005）的 MSTParser 用机器学习分类器独立为依存关系打分（他用 MIRA 做在线学习，也可换其他方法 ） 
	**Neural graph-based parser:** Dozat and Manning (2017) et seq. – very successful! 
	基于神经图的剖析器：Dozat 和 Manning（2017 等后续研究）——非常成功！
3. **Constraint Satisfaction**  约束满足法 
	Edges are eliminated that don’t satisfy hard constraints. Karlsson (1990), etc. 
	移除不满足硬性约束的边。代表如 Karlsson（1990）等的研究 。
4. “**Transition-based parsing” or “deterministic dependency parsing”**  “转移基剖析” 或 “确定性依存剖析”
	Greedy choice of attachments guided by good machine learning classifiers E.g., 
	MaltParser (Nivre et al. 2008). Has proven highly effective. 
	由优质机器学习分类器引导，贪心选择依存连接 例如，MaltParser（Nivre 等人 2008）。已被证明十分有效 。
**解释**
**自然语言处理中“依存句法分析（Dependency Parsing）”的主流算法分类**，介绍了4类核心方法： 
5. **动态规划法**： 借“动态规划”思想拆解句法分析问题，Eisner 算法通过调整“中心词位置”（放两端 ）优化计算，\( O(n^3) \) 复杂度适合中等规模句子分析，是经典依存分析算法。 
6. **图算法**： 把句子当“图”，用“最小生成树（MST）”找最优依存结构。MSTParser 结合机器学习（如 MIRA 在线学习 ）给依存关系打分，后续又发展出神经图模型（Dozat & Manning ），利用神经网络优化图结构学习，效果突出。 
7. **约束满足法**： 靠“硬性约束”（如句法规则、语义限制 ）过滤不合理依存边，简化分析空间，Karlsson 等的研究是这类方法的早期探索，体现“规则 + 剪枝”思路。 
8. **转移基/确定性剖析**： 模拟“逐步决策”过程，用机器学习分类器做“贪心选择”（每步选最可能的依存连接 ），MaltParser 是典型工具，因高效、易实现，在工业级 NLP 任务中常用。 这些方法覆盖了“传统算法（动态规划、图、约束 ）”到“神经模型（神经图剖析器 ）”的演进，也体现“规则驱动”到“数据驱动（机器学习 ）”的趋势，是理解依存分析技术路线的关键框架。

#### Greedy transition-based parsing \[Nivre 2003] 贪心转移基剖析 [尼弗 2003]
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808235301627.png)
- **一种简单形式的贪心判别式依存剖析器**
- 该剖析器执行一系列自底向上的操作
    - 大致类似移进 - 归约剖析器里的 “移进” 或 “归约” 操作，但 “归约” 操作经过特化，用于创建中心词在左侧或右侧的依存关系
- 该剖析器包含：
    - 一个栈 σ，栈顶朝右
        - 以 ROOT 符号为初始内容
    - 一个缓冲区 β，缓冲区顶端朝左
        - 以输入语句为初始内容
    - 一组依存弧 A
        - 初始为空
    - 一组操作

#### Basic transition-based dependency parser 基本的基于转移的依存句法分析器
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250808235706178.png)
初始状态：栈 $\sigma = [\text{ROOT}]$，==缓冲区== $\beta = w_1,\ ...,\ w_n$，==依存弧集合== $A = \emptyset$

1. 移进（Shift）： 状态 $\sigma,\ w_i|\beta,\ A$ 变为 $\sigma|w_i,\ \beta,\ A$（==把缓冲区首词 $w_i$ 移进栈== ）
2. 左弧（Left-Arc$_r$）： 状态 $\sigma|w_i|w_j,\ \beta,\ A$ 变为 $\sigma|w_j,\ \beta,\ A \cup \{r(w_j, w_i)\}$（==栈顶两词 $w_i, w_j$，构建 $w_j$ 为中心词、$w_i$ 为依存词的左弧，更新栈和依存弧 ）==
3. 右弧（Right-Arc$_r$）： 状态 $\sigma|w_i|w_j,\ \beta,\ A$ 变为 $\sigma|w_i,\ \beta,\ A \cup \{r(w_i, w_j)\}$（栈==顶两词 $w_i, w_j$，构建 $w_i$ 为中心词、$w_j$ 为依存词的右弧，更新栈和依存弧 ==）

==结束状态：栈 $\sigma = [w]$，缓冲区 $\beta = \emptyset$（所有词处理完毕 ）==

**3个步骤前进/左弧/右弧都可以选择，只能选一个。**

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250809000308115.png)
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250809000731035.png)

#### MaltParser [Nivre and Hall 2005]
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250809000948595.png)
 - We have left to explain how we choose the next action 🤔  我们还得解释如何选择下一步操作 
	 - Answer: Stand back, I know machine learning!  别急，我懂机器学习！
 - Each action is predicted by a discriminative classifier (e.g., softmax classifier) over each legal move  - ==每个操作由判别式分类器（如 softmax 分类器 ）在合法操作中预测得出==
	 - Max of 3 untyped choices; max of \( |R| \times 2 + 1 \) when typed  
	   无类型时最多3种选择；带类型时最多 $|R| \times 2 + 1$ 种（$|R|$是关系类型数量 ）
	 - Features: top of stack word, POS; first in buffer word, POS; etc. 
	 特征：栈顶词及其词性；缓冲区首词及其词性；等等
 - There is NO search (in the simplest form)  **（最简形式下）无需搜索**
	 - But you can profitably do a beam search if you wish (slower but better): You keep \( k \) good parse prefixes at each time step  但如果愿意，**也可高效执行束搜索（**速度慢但效果好 ）：每一步保留 \( k \) 个优质剖析前缀
 - The model’s accuracy is fractionally below the state of the art in dependency parsing, but  该模型==准确率略低于依存句法分析的当前最优水平==，但
 - It provides **very fast linear time parsing**, with high accuracy – great for parsing the web 
 它能实现非常快速的线性时间剖析，且准确率高——很适合网页文本剖析
**解释**
- **工具与提出者**：“MaltParser” 是自然语言处理中用于依存句法分析的工具，由 Nivre 和 Hall 在2005年提出 ，在依存分析领域应用广泛 。 
- **操作选择机制**：用机器学习（判别式分类器 ）解决“下一步操作选什么”的问题 ，通过提取栈顶词、缓冲区词及其词性等特征，让分类器（如 softmax ）在合法操作里预测该选的动作 ，操作数量因是否带关系类型有不同上限，体现对复杂句法关系的适配 。 
- **搜索策略**：最简版无需搜索，靠贪心选操作；也可扩展束搜索，保留多个较优剖析路径（用 \( k \) 个前缀 ），平衡速度和效果 ，束搜索虽慢但能提升剖析质量 。 
- **性能特点**：准确率虽不是顶尖，但胜在速度快（线性时间复杂度 ）、准确率尚可 ，适合网页等大规模文本的依存剖析场景 ，满足实际应用中对效率和效果的双重需求 ，解释了其在工业级或大规模数据处理中的价值 。 整体介绍了 MaltParser 的核心工作机制（机器学习选操作 ）、可选策略（搜索方式 ）和性能优势，是理解该工具在依存分析生态中定位的关键内容 。

#### Conventional Feature Representation   传统特征表示  

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250809001626640.png)

**翻译**
栈（Stack）  
ROOT  has.VBZ  good.JJ  
He.PRP （nsubj 依存关系 ）  

缓冲区（Buffer）  
control.NN  ...  

二进制、稀疏表示  
维度 =$10^6–10^7$  
[0 0 0 1 0 0 1 0 ... 0 0 1 0]  

**特征模板**：通常是配置中1–3个元素的组合  

**指示特征**  
$s1$ 的词 = good 且$s1$ 的词性 = JJ  
$s2$ 的词 = has 且$s2$ 的词性 = VBZ 且$s1$ 的词 = good  
$s_2$ 左孩子的词性 = PRP 且$s_2$ 的词性 = VBZ 且$ s_1$ 的词性 = JJ  
$s_2$ 左孩子的词 = He 且$s_2$ 左孩子的标签 = nsubj 且 s_2$ 的词 = has  


### 解释  
这是**自然语言处理（NLP）中“依存句法分析”的传统特征工程**，核心讲“如何把句法状态（栈、缓冲区、依存关系 ）转化为模型可学习的特征”，分3层理解：  

1. **句法状态可视化**：  
   用“栈（Stack）”存待处理的句法成分（如 ROOT、has.VBZ、good.JJ ），“缓冲区（Buffer）”存未处理词（如 control.NN ），“nsubj” 是依存关系（He.PRP 是 has.VBZ 的主语 ），呈现依存分析的中间状态。  

2. **特征编码方式**：  
   把句法状态转成“二进制、稀疏”向量（dim 达$10^6–10^7$ ），用0/1标记特征是否存在。例如栈里的 good.JJ、has.VBZ 等成分，对应向量中特定位置为1，体现“稀疏性”（大部分位置是0 ）。  

3. **特征模板与指示特征**：  
   - **模板**：选“栈顶1–3个元素（如$s1, s2$ ）、左孩子（$lc(s_2)$ ）”等构建特征，让模型捕捉局部句法关系。  
   - **指示特征**：是具体的逻辑组合（如$s1.w = \text{good} \land s1.t = \text{JJ}$ ），用这些细粒度特征教模型识别“什么状态下该选什么依存操作”，是传统机器学习（如 SVM、Softmax ）做依存分析的核心“知识载体”。  

**这套方法是“传统 NLP”的典型思路——靠人工设计特征模板编码语言知识，再用分类器学习。** ==虽然后续被“神经模型（直接学词向量、句法表示 ）”部分替代，但理解它能明白“特征工程如何赋能句法分析”，是衔接传统方法与现代模型的桥梁。==

#### Evaluation of Dependency Parsing: (labeled) dependency accuracy 依存句法分析的评估：（带标签的）依存准确率
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250809002742305.png)
### 提取文字

Evaluation of Dependency Parsing: (labeled) dependency accuracy

Acc = \frac{\text{\# correct deps}}{\text{\# of deps}}

UAS = 4 / 5 = 80\% LAS = 2 / 5 = 40\%

Gold（标准依存 ） 
1 2 She nsubj 
2 0 saw root 
3 5 the det 
4 5 video nn 
5 2 lecture obj

Parsed（模型输出 ） 
1 2 She nsubj 
2 0 saw root 
3 4 the det 
4 5 video nsubj 
5 2 lecture ccomp

### 翻译

#### 依存句法分析的评估：（带标签的）依存准确率

$准确率 = \frac{\text{正确依存数}}{\text{总依存数}}$

**无标签依存准确率**（UAS） = 4 / 5 = 80\% **带标签依存准确率**（LAS） = 2 / 5 = 40\%

标准依存（Gold ） 
序号 **“中心词（head） + 依存词（dependent） + 关系标签”** 的树状结构
==标签 （）里面的==
1 2 她 主语（**nsubj** ） 
2 0 看见（saw ） 根节点（root ） 
3 5 这（the ） 限定词（det ）
4 5 视频（video ） 名词修饰（nn ） 
5 2 讲座（lecture ） 宾语（obj ）

模型输出（Parsed ） 
序号 **“中心词（head） + 依存词（dependent） + 关系标签”** 的树状结构
1 2 她(her) 主语（nsubj ） 
2 0 看见（saw ） 根节点（root ） 
3 4 视频（video ） 限定词（det ） 
4 5 视频（video ） 主语（nsubj ） 
5 2 讲座（lecture ） 并列补语（ccomp ）

依存关系（Dependency Relation）是句法分析里的核心概念，用来描述句子中**词语之间的支配与从属关系** ，通俗讲就是 “谁管谁、谁修饰谁、谁辅助谁” 的语法关联。
用图里的例子具体看
- “She” 和 “saw” 是 `nsubj`（主语关系 ）→ “saw” 是核心动词（管着 “She” ），“She” 是主语（被 “saw” 管 ），“nsubj” 就是这层关系的标签。
- “saw” 和 “lecture” 标准标注是 `obj`（宾语关系 ）→ “saw” 支配 “lecture”，“lecture” 是 “saw” 的宾语，“obj” 是标签。

本质上，依存关系把线性的句子，转化成 **“中心词（head） + 依存词（dependent） + 关系标签”** 的树状结构，让隐藏的语法层次（谁是核心、谁是附属 ）清晰化，是理解句子句法结构的基础 。
### 解释

这是**依存句法分析（Dependency Parsing）的核心评估指标演示**，分2类指标、2组数据对比：
1. **指标定义**:
    - **UAS（无标签依存准确率，Unlabeled Attachment Score ）**：只看“依存关系是否正确连接”（不管标签 ），公式是 “正确依存数 / 总依存数”。
    - **LAS（带标签依存准确率，Labeled Attachment Score ）**：既看“连接是否对”，也看“标签（如 nsubj、obj ）是否对”，要求更严格。
        
2. **示例计算**：
    
    - 总依存数是5（Gold 里5行 ）。
    - **UAS**：模型输出中，“She-nsubj”“saw-root”“the-det”“lecture-（连接到 saw ）”这4个依存的“连接对象”正确，所以 4/5 = 80\% 。
    - **LAS**：只有“Shensubj”“saw-root”的“连接 + 标签”全对，所以 2/5 = 40\% （比如 “the” 的连接对象错，“video”“lecture” 的标签错 ）。
3. **意义**： UAS 衡量“结构连接能力”，LAS 衡量“结构 + 标签精准度”，两者结合能全面评估依存 parser 的性能——UAS 高说明“找得到正确中心词”，LAS 高说明“连对且标签准”。
这是 NLP 中依存分析评估的“标准范式”，几乎所有依存 parser 都会用 UAS/LAS 报告结果，理解它就能看懂论文里的模型性能对比。