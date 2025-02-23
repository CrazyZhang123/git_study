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
	- Det    N     Adj      P     N  （限定词 名词 形容词 介词 名词）
- **Words combine into phrases** 单词组合成短语
	- the cuddly cat（可爱的猫）, by the door（在门边）
- **Phrases can combine into bigger phrases**  短语可以组合成更大的短语
	- the cuddly cat by the door 
	
1. **语言结构观点**：介绍了语言结构的一种观点，即成分结构，它等同于短语结构语法和上下文无关语法（CFGs）。上下文无关语法是形式语言理论中的概念，用于描述自然语言或编程语言的语法结构。
2. **短语结构构建**：短语结构的构建从单词开始，如“the”（限定词Det）、“cat”（名词N）、“cuddly”（形容词Adj）、“by”（介词P）、“door”（名词N）。这些单词首先组合成短语，例如“the cuddly cat”（“the”和“cuddly”修饰“cat” ）和“by the door”。然后，这些短语可以进一步组合成更大的短语，像“the cuddly cat by the door” 。这种从单词到短语再到更大短语的嵌套组合方式，体现了短语结构将单词组织成具有层次结构的成分的特点，有助于分析句子的语法构成和语义关系 。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206225005.png)
这里介绍了一些CFG(上下文无关文法)，从右边的 **lexicon(词典)**，到中间的**Grammer(文法)**，随着老师的讲解不断再扩充，从NP(名词短语)，PP(介词短语), VP(动词短语),S(句子)，以及他们的文法句子，\*和编译原理一样，代表0或多次，这就是一个简单的文法用于生成S，这里就是举例说明了一下CFG。

### Two views of linguistic structure: Dependency structure 依存结构
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206230311.png)

 - Dependency structure shows which words depend on (modify, attach to, or are arguments of) which other words. 依存结构展示了哪些单词依赖于（修饰、附着于或作为其论元）哪些其他单词。 

Look in the large crate in the kitchen by the door
看看厨房门边的大箱子

解释 
1. **依存结构定义**：依存结构是语言结构的一种视图，它关注单词之间的依赖关系。在自然语言中，单词并非孤立存在，而是存在着各种语义和语法上的联系。例如，形容词通常修饰名词，介词短语常常对动词或名词进行补充说明等。 
2. **示例分析**：以“Look in the large crate in the kitchen by the door”这句话为例，从依存结构的角度来看，“in”（介词P）和“the large crate”存在依赖关系，“in”表示位置关系，附着于“the large crate” ；“the large crate” 中“large”修饰“crate” ；“in the kitchen”和“in the large crate” 相关，进一步说明“the large crate” 的位置；“by the door” 又对“in the kitchen” 进行补充说明。通过这种方式，依存结构清晰地呈现了句子中单词之间的依赖、修饰等关系，有助于深入理解句子的语义和语法结构。

#### Why do we need sentence structure? 为什么需要句子结构
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206230656.png)
- Humans communicate complex ideas by composing words together into bigger units to convey complex meanings  人类通过将单词组合成更大的单位来传达复杂的意义，从而交流复杂的想法。
- Listeners need to work out what modifies [attaches to] what 听众需要弄清楚什么修饰（附着于）什么。
- A model needs to understand sentence structure in order to be able to interpret language correctly  一个模型需要理解句子结构，以便能够正确地解释语言。

1. **人类交流层面**：人类语言要表达复杂思想，不能仅靠单个单词，而是要将单词组合成短语、句子等更大的语言单位。例如，“我在美丽的公园散步”，“美丽的”修饰“公园”，“在公园”是表示地点的短语，它们共同构成完整的表达，传递复杂语义。
2. **听众理解层面**：当人们听到话语时，需要分析句子结构来明确单词间的修饰或依附关系。比如“红色的苹果”，听众要明白“红色的”是对“苹果”的修饰限定，才能准确理解语义。 
3. **模型应用层面**：在自然语言处理等领域，模型若想正确理解和处理语言，就必须理解句子结构。比如机器翻译模型，只有准确分析源语言句子结构，才能在目标语言中生成恰当、符合语法和语义的翻译。

### Prepositional phrase attachment ambiguity 介词短语附着歧义
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206231202.png)
 San Jose cops kill man with knife **圣何塞警察用刀杀死男子 / 圣何塞警察杀死带刀的男子**
 Ex - college football player, 23, shot 9 times allegedly charged police at fiancee’s home 23岁前大学橄榄球运动员，据称在未婚妻家中向警察开枪，中枪9次 
 解释 
  **示例分析**：以新闻标题“San Jose cops kill man with knife”为例，介词短语“with knife”存在附着歧义。它既可以理解为警察用刀杀死男子（“with knife”修饰“kill”这个动作，说明杀人的工具），也可以理解为警察杀死了带着刀的男子（“with knife”修饰“man”，描述男子的状态） 。这种歧义在自然语言中较为常见，会给语言的理解和处理带来困难。在自然语言处理中，模型需要准确判断介词短语的附着对象，才能正确理解句子的语义。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206231529.png)
Scientists count whales from space
科学家在太空中数鲸鱼。
**“from space” 是指科学家在太空这个位置来数鲸鱼，而不是鲸鱼在太空，所以海洋中鲸鱼的图片理解正确，太空里鲸鱼的图片理解错误。**

#### PP attachment ambiguities multiply 介词短语附着歧义增多
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206232213.png)
 - A key parsing decision is how we ‘attach’ various constituents 一个关键的句法分析决策是我们如何“附着”各种成分
 - PPs, adverbial or participial phrases, infinitives, coordinations, 介词短语（PPs）、状语或分词短语、不定式、并列结构等
 - The board approved \[its acquisition\] \[by Royal Trustco Ltd.\] \[of Toronto\] \[for $27 a share\] \[at its monthly meeting]\
 -  董事会在其月度会议上批准了（由皇家信托公司（位于多伦多）以每股27美元）对其的收购。 
正确的修饰关系
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206232925.png)

 解释 
 1. **示例分析**：以句子“The board approved \[its acquisition\] \[by Royal Trustco Ltd.\] \[of Toronto\] \[for $27 a share\] \[at its monthly meeting]\.”为例，其中多个介词短语存在附着歧义。比如“by Royal Trustco Ltd.” 既可以理解为修饰“acquisition”（表示收购的主体），“of Toronto” 可理解为修饰“Royal Trustco Ltd.”（表示公司所在地点） ；“for $27 a share” 修饰“acquisition”（表示收购的价格） ；“at its monthly meeting” 修饰“approved”（表示批准这一动作发生的场合）。但这些修饰关系的确定在不同语境下可能会有所不同，这就体现了介词短语附着歧义会随着句子中成分的增多而变得复杂，在自然语言处理中准确分析这些关系对于正确理解句子语义至关重要。

#### Coordination scope ambiguity 并列范围歧义
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206233220.png)
 - [Shuttle veteran] and [longtime NASA executive Fred Gregory] appointed to board 
 - [航天飞机老手] 和 [美国国家航空航天局（NASA）长期高管弗雷德·格雷戈里] 被任命为董事会成员
 - [Shuttle veteran and longtime NASA executive] Fred Gregory appointed to board 
 -  [航天飞机老手兼美国国家航空航天局（NASA）长期高管] 弗雷德·格雷戈里被任命为董事会成员
 - 解释 1. **概念**：“Coordination scope ambiguity”指的是并列范围歧义，即在句子中由于并列结构的范围不明确而导致的语义理解上的歧义。 
 1. **示例分析** - 对于“[Shuttle veteran] and [longtime NASA executive Fred Gregory] appointed to board”，从结构上理解是“航天飞机老手”和“美国国家航空航天局长期高管弗雷德·格雷戈里”两个人被任命为董事会成员，这里“and”连接的是两个不同的个体。 - 而“[Shuttle veteran and longtime NASA executive] Fred Gregory appointed to board”，则是将“航天飞机老手兼美国国家航空航天局长期高管”作为对“弗雷德·格雷戈里”的修饰，意思是弗雷德·格雷戈里这一个人被任命为董事会成员 。这种因并列范围界定不同产生的歧义在自然语言处理中需要准确分析句子结构和语义来消除，以确保对句子的正确理解。
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250206233545.png)
这句话“Doctor: No heart, cognitive issues”存在结构歧义。 
- 从红色标注来看，可理解为“医生表示：没有心脏方面的问题，但是有认知方面的问题” ，即把“heart”和“cognitive”分开理解，分别指心脏和认知。
- 从蓝色标注来看，可理解为“医生表示：没有心脏 - 认知方面的问题” ，即将“heart”和“cognitive”看作一个整体修饰“issues” ，表示不存在与心脏和认知相关的问题。这体现了自然语言中因结构理解不同产生的歧义现象。

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