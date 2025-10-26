---
created: 2024-10-10T22:53
updated: 2024-11-01T22:25
---
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241010230510.png)

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241101211602.png)
- 两个女人在吵架，同样的语言，不同人的理解不一样

人类的沟通能力使得人类比其他生物更有优势。

- 计算机和人工智能的关键问题：==如何让计算机能够理解人类传达的意思==

机器翻译 neural machine translation
- Trained on text data, neural machine translation is quite good!
NLP 自然语言处理
- GPT3: A first step on the path to universal models 通用模型的第一步
- 我们不用再去专门的设定具体的功能，比如检测垃圾邮件，色情信息，任何语言的信息，只是建立所有这些不同任务的独立监督分类器,我们刚刚建立了一个可以理解的模型。
==and just building all these separate supervised
classifiers for every different task，we've now just built up a  model  that  understands .

==所以它所能做的只是预测后面的单词,左边输入要生成什么，模型就会生成后面的文本，实际上是一次预测一个单词，然后生成的单词又作为输入去预测，循环往复。
So exactly what it does is it just predicts following words.

GPT3 功能
- 我们给出问题，GPT3可以给出合理的问答后续
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241101214712.png)
- 翻译sql
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241101222157.png)

#### the meaning of word
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20241101222333.png)
符号 和 想法或者事情 的对应
denotational semantics 指称语义