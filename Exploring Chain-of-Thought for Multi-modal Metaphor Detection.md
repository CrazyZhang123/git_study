Exploring Chain-of-Thought for Multi-modal Metaphor Detection

探索思维链在多模态隐喻检测中的应用

## Abstract

Metaphors are commonly found in advertising and internet memes. However, the free form of internet memes often leads to a lack of high-quality textual data. Metaphor detection demands a deep interpretation of both textual and visual elements, requiring extensive common-sense knowledge, which poses a challenge to language models. To address these challenges, we propose a compact(小型的；紧密的) framework called C4MMD, which utilizes a Chain-of-Thought(CoT) method for Multi-modal Metaphor Detection. Specifically(具体地，特别地), our approach designs a three-step process inspired by CoT that extracts and integrates knowledge from Multi-modal Large Language Models(MLLMs) into smaller ones. We also developed a modality fusion architecture(模态融合架构) to transform knowledge from large models into metaphor features, supplemented(补充；增补) by auxiliary(辅助的；备用的) tasks to improve model performance. Experimental results on the MET-MEME dataset demonstrate that our method not only effectively enhances the metaphor detection capabilities of small models but also outperforms existing models. To our knowledge, this is the first systematic study(系统研究) leveraging(利用) MLLMs in metaphor detection tasks. The code for our method is publicly available at https: //github.com/xyz189411yt/C4MMD.

隐喻在广告和网络模因中很常见。然而，网络模因的自由形式往往导致缺乏高质量的文本数据。隐喻检测需要对文本和视觉元素进行深入解读，这需要大量的常识性知识，这给语言模型带来了挑战。为了应对这些挑战，我们提出了一个名为C4MMD的紧凑框架，该框架利用思维链（CoT）方法进行多模态隐喻检测。具体来说，我们的方法受思维链启发，设计了一个三步流程，从多模态大语言模型（MLLMs）中提取知识并将其整合到较小的模型中。我们还开发了一种模态融合架构，将大模型中的知识转化为隐喻特征，并辅以辅助任务来提高模型性能。在MET-MEME数据集上的实验结果表明，我们的方法不仅有效地增强了小模型的隐喻检测能力，而且优于现有模型。据我们所知，这是首次在隐喻检测任务中系统地利用多模态大语言模型的研究。我们方法的代码可在https://github.com/xyz189411yt/C4MMD上公开获取。 

**解释:**

- “Metaphors” 是 “metaphor” 的复数形式，常见释义为 “隐喻；暗喻” ，是一种重要的修辞手法和语言现象。在语言表达中，隐喻通过将一个事物的特征或概念，含蓄地映射到另一个事物上，从而暗示两者之间的相似性，以此来更生动、形象地传达意义。
  - **文学创作**：在诗歌、小说等文学作品中，隐喻是塑造意境、表达情感的常用手法。例如，“Juliet is the sun.”（朱丽叶是太阳。）出自莎士比亚的《罗密欧与朱丽叶》，这里将朱丽叶比作太阳，并非说她具有太阳的物理属性，而是借此表达罗密欧对朱丽叶的爱慕，强调她在自己心中如同太阳般重要、美好且充满光芒。
  - **日常交流**：在日常对话和写作里，隐喻也被频繁使用，让表达更加简洁有力。比如，“Time is money.”（时间就是金钱。）人们用 “金钱” 来比喻 “时间”，强调时间的宝贵，提醒大家要珍惜时间，如同珍惜金钱一样。
- “网络模因（internet memes）” 指在互联网上迅速传播、被大量用户复制和模仿的文化现象，涵盖多种形式，传播和演变依赖互联网用户的参与和创造。

## 1 Introduction

Metaphors are highly prevalent(流行的；盛行的) in our everyday expressions and writings, which can have a range of impacts on downstream tasks in Natural Language Processing (NLP), such as semantic understanding (Neuman et al., 2013), sentiment analysis(Ghosh and Veale, 2016; Mohammad et al., 2016) and other tasks. In recent years, the rise of social media has sparked(引发; 触发;) interest in multi-modal metaphors. As a result, **several datasets for multimodal metaphors have been proposed** (Zhang et al., 2021, 2023a; Alnajjar et al., 2022).

隐喻在我们的日常表达和写作中极为普遍，它会对自然语言处理（NLP）中的一系列下游任务产生影响，比如语义理解（Neuman等人，2013年）、情感分析（Ghosh和Veale，2016年；Mohammad等人，2016年）以及其他任务。近年来，社交媒体的兴起引发了人们对多模态隐喻的兴趣。因此，一些用于多模态隐喻研究的数据集被提出（Zhang等人，2021年、2023年a；Alnajjar等人，2022年） 。 

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250220143259781.png" alt="image-20250220143000244" style="zoom:67%;" />

​														Figure 1: An example of multi-modal metaphor detection.

​														图1：多模态隐喻检测示例。

Current research on multi-modal metaphor detection is still in its early stages. The primary challenge lies in the complexity and variety of multimodal metaphors. Compared to single-modality detection, multi-modal metaphor detection not only spots metaphors in sentences but also categorizes them as image-dominated, text-dominated, or complementary. The second major challenge arises from the poor quality of textual content, mainly sourced from advertisements and memes on social media. Texts give the image more metaphorical features. Recent efforts use OCR (Optical Character Recognition) to extract texts in the image. However, only relying on OCR to convert them into parallel texts leads to the loss of texts’ positional information. Figure 1 presents a representative example, symbolizing how ’PUBG’ (a video game) acts like a trap preventing "me" from achieving my "life goals".

目前，**多模态隐喻检测的研究仍处于早期阶段。**主要挑战在于多模态隐喻的复杂性和多样性。**与单模态检测相比，多模态隐喻检测不仅要识别句子中的隐喻，还需要将它们分类为以图像为主导、以文本为主导或互补型**。第二个主要挑战源于**文本内容质量较差，这些文本主要来自社交媒体上的广告和模因。**文本赋予图像更多隐喻特征。最近，人们尝试使用光学字符识别（OCR）技术来提取图像中的文本。然而，**仅依靠OCR将其转换为并行文本会导致文本位置信息的丢失**。图1展示了一个典型例子，象征着《绝地求生》（一款电子游戏）如何像一个陷阱，阻止 “我 ”实现 “人生目标”。 

**解释:**

OCR 即 Optical Character Recognition，中文名为光学字符识别。它是一种能够将图像中的文字信息转换为可编辑文本的技术。下面从其工作原理、应用场景和局限性方面简单介绍：

- **工作原理**：OCR 技术的核心原理是通过对图像中字符的特征进行提取和分析，与预定义的字符模板或特征库进行匹配，从而识别出字符。在识别过程中，首先对包含文字的图像进行预处理，包括灰度化、降噪、倾斜校正等操作，以提高图像的清晰度和质量。接着，进行字符分割，将文本中的字符逐个分离出来。然后，对每个字符进行特征提取，例如笔画特征、轮廓特征等，并将这些特征与预先训练好的模型或字符库进行比对，最终确定每个字符的识别结果。

To overcome these challenges, we hope to gain insights(启发) from LLMs, utilizing their rich world knowledge and contextual understanding capabilities to obtain deeper meanings of both images and text. An intuitive(直观) and efficient approach is to use these LLMs to generate supplementary(补充的) information without fine-tuning them; we then only need to fine-tune a smaller model to establish(建立) connections between this information and metaphors. To reduce the illusion(幻觉) of MLLMs, inspired by CoT (Wei et al., 2022), we have designed a three-step method that progressively acquires the MLLM’s information in describing images, analyzing text, and integrating information from both modalities. The advantages of this strategy are as follows: First, it can provide downstream models with additional information for each modality. Second, the shallow-to-deep understanding sequence aligns closely with human logic, making it easier for the LLM to grasp deeper meanings. Furthermore, subsequent steps can correct misunderstandings from earlier steps, enhancing the model’s robustness.

为了克服这些挑战，我们希望从大语言模型（LLMs）中获取启发，利用它们丰富的世界知识和上下文理解能力来挖掘图像和文本的深层含义。**一种直观且高效的方法是，在不对这些大语言模型进行微调的情况下，利用它们生成补充信息**；然后，我们只需对一个较小的模型进行微调，以建立这些信息与隐喻之间的联系。为了**减少多模态大语言模型（MLLMs）产生的幻觉，受思维链**（CoT）（Wei等人，2022）的启发，**我们设计了一个三步法，逐步获取多模态大语言模型在描述图像、分析文本以及整合两种模态信息时的内容。**这种策略的优势如下：第一，它可以为下游模型提供每个模态的额外信息。第二，从浅到深的理解顺序与人类逻辑紧密契合，使大语言模型更容易把握深层含义。此外，后续步骤可以纠正前面步骤中的误解，增强模型的稳健性。 



Overall, we utilize a CoT-based method called C4MMD to summarize knowledge from MLLMs and enhance metaphor detection in smaller models by fine-tuning them to link this knowledge with metaphors. The basic idea is shown in Figure 1, we first input images and text into the MLLM and obtain information describing the image, text, and their fusion. Furthermore, we have designed a downstream modality fusion structure, which is intended to translate supplementary information into metaphorical features for more accurate classification. Specifically, we have designed two auxiliary tasks focused on determining the presence of metaphors within the image and text modalities.

总体而言，**我们利用一种名为C4MMD的基于思维链（CoT）的方法，从多模态大语言模型（MLLMs）中总结知识，并通过微调较小的模型，将这些知识与隐喻联系起来，从而提升其隐喻检测能力**。基本思路如图1所示，我们首先将图像和文本输入到多模态大语言模型中，获取描述图像、文本及其融合信息的内容。此外，我们设计了一种下游模态融合结构，旨在将补充信息转化为隐喻特征，以实现更准确的分类。具体来说，我们设计了**两个辅助任务**，专注于确定图像和文本模态中是否存在隐喻。 

### 总结

```
本文聚焦于多模态隐喻检测，分析了当前研究面临的挑战，并提出了创新的解决方法C4MMD，具体内容如下：
1. **研究背景**：隐喻在日常表达和写作中普遍存在，对自然语言处理下游任务影响广泛。社交媒体兴起引发多模态隐喻研究兴趣，相关数据集不断涌现。
2. **研究挑战**
    - **隐喻复杂多样**：多模态隐喻检测不仅要识别隐喻，还需区分主导类型（图像主导、文本主导或互补型），较单模态检测更具挑战。
    - **文本质量不佳**：文本多源于社交媒体广告和模因，质量差。用OCR提取图像文本会导致位置信息丢失。
3. **解决策略**
    - **利用大模型知识**：借助大语言模型（LLMs）丰富的知识和理解能力，在不微调LLMs的情况下生成补充信息，再微调小模型建立信息与隐喻的联系。
    - **设计三步法**：受思维链（CoT）启发，设计三步法逐步获取多模态大语言模型（MLLMs）描述图像、分析文本和整合信息的内容，减少MLLMs幻觉，为下游模型提供信息，符合人类逻辑且增强模型稳健性。
4. **C4MMD方法**
    - **总结知识提升能力**：利用基于CoT的C4MMD方法从MLLMs总结知识，微调小模型提升其隐喻检测能力。
    - **设计融合结构**：设计下游模态融合结构，将补充信息转化为隐喻特征以实现准确分类。
    - **设置辅助任务**：设计两个辅助任务，确定图像和文本模态中隐喻的存在情况。 
```



## 2 Related Work

Early metaphor detection tasks were confined to(仅限于) a single modality and employed methods based on rule constraints and metaphor dictionaries (Fass, 1991; Krishnakumaran and Zhu, 2007; Wilks et al., 2013). With the flourishing(繁荣的) development in the field of NLP, machine learning-based methods (Turney et al., 2011; Shutova et al., 2016) and neural network-based methods (Mao et al., 2019; Zayed et al., 2020) have successively(先后) emerged. Following the introduction of the Transformer (Vaswani et al., 2017), methods based on pre-trained models gradually supplanted(取代，替代) the former methods and became the current mainstream approach (Cabot et al., 2020; Li et al., 2021; Lin et al., 2021). Ge et al. (2023) have categorized current efforts into four main directions, namely additional data and feature methods (Shutova et al., 2016; Gong et al., 2020; Kehat and Pustejovsky, 2021), semantic methods (Mao et al., 2019; Choi et al., 2021; Su et al., 2021; Zhang and Liu, 2022; Li et al., 2023b; Tian et al., 2023a), context-based methods (Su et al., 2020; Song et al., 2021), and multitask methods (Chen et al., 2020; Le et al., 2020; Mao et al., 2023; Badathala et al., 2023; Zhang and Liu, 2023; Tian et al., 2023b), where semantic methods and multitask methods have become the primary focus of recent research.

**早期的隐喻检测任务局限于单模态，采用基于规则约束和隐喻词典的方法**（法斯，1991；克里希纳库马兰和朱，2007；威尔克斯等人，2013 ）。随着自然语言处理领域的蓬勃发展，基于机器学习的方法（特尼等人，2011；舒托娃等人，2016 ）和基于神经网络的方法（毛等人，2019；扎耶德等人，2020 ）相继出现。自Transformer（瓦斯瓦尼等人，2017 ）被引入后，基于预训练模型的方法逐渐取代了前者，成为当前的主流方法（卡博特等人，2020；李等人，2021；林等人，2021 ）。葛等人（2023 ）将当前的研究工作分为**四个主要方向，即额外数据和特征方法**（舒托娃等人，2016；龚等人，2020；凯哈特和普斯捷约夫斯基，2021 ）、**语义方法**（毛等人，2019；崔等人，2021；苏等人，2021；张和刘，2022；李等人，2023b；田等人，2023a ）、**基于上下文的方法**（苏等人，2020；宋等人，2021 ）以及**多任务方法**（陈等人，2020；勒等人，2020；毛等人，2023；巴达萨拉等人，2023；张和刘，2023；田等人，2023b ），其中**语义方法和多任务方法已成为近期研究的主要焦点。** 

As an emerging direction, numerous datasets across image and text modalities have emerged, primarily sourced from social media and advertisements, yielding(产生，提供) extensive(广泛的;大量的) multilingual(多语言) text-image modal data (Zhang et al., 2021; Xu et al., 2022; Zhang et al., 2023a). Unlike the aforementioned(上述的，前面提到的) approaches that extract information from different modalities and directly merge them, we leverage LLMs employing the CoT method to analyze features between modalities, aiding(帮助) downstream models in cross-modal fusion.

作为一个新兴方向，出现了大量跨图像和文本模态的数据集，这些数据集主要来源于社交媒体和广告，产生了广泛的多语言文本 - 图像模态数据（Zhang等人，2021年；Xu等人，2022年；Zhang等人，2023年a）。与上述从不同模态中提取信息并直接融合的方法不同，我们利用大语言模型（LLMs）并采用思维链（CoT）方法来分析模态之间的特征，帮助下游模型进行跨模态融合。 

### 总结

```
这段内容主要回顾了隐喻检测方法的发展历程，介绍了当前研究方向，并阐述了本文独特的研究思路，具体如下：
1. **隐喻检测方法发展历程**
    - **早期单模态方法**：早期隐喻检测局限于单模态，采用基于规则约束和隐喻词典的方法。
    - **方法的演进**：随着NLP领域发展，机器学习和神经网络方法先后出现。Transformer引入后，基于预训练模型的方法成为主流。
2. **当前研究方向分类**：葛等人（2023）将当前隐喻检测研究分为四个方向，即额外数据和特征方法、语义方法、基于上下文的方法和多任务方法，其中语义和多任务方法是近期研究焦点。
3. **新兴研究方向与本文思路**
    - **新兴方向**：跨图像和文本模态的数据集不断涌现，源于社交媒体和广告，提供了多语言文本 - 图像模态数据。
    - **本文思路**：区别于直接提取和融合不同模态信息的方法，本文利用大语言模型（LLMs）结合思维链（CoT）方法分析模态间特征，助力下游模型进行跨模态融合。 
```

## 3 Method

We propose a novel framework called C4MMD using MLLMs to enhance metaphor detection. We first introduce the task definition(3.1) and the complete model architecture((3.2). After that, we elaborate on(详细说明) knowledge acquisition(获得，得到) from MLLMs using the CoT method(3.3) and the implementation of the downstream fusion module(3.4). Finally, we provide a brief exposition(阐述解释) of the training methodology (3.5).

我们提出了一种名为C4MMD的全新框架，该框架利用多模态大语言模型（MLLMs）来增强隐喻检测能力。我们首先介绍**任务定义（3.1节）和完整的模型架构（3.2节）**。之后，详细阐述如何使用思维链（CoT）方法从多模态大语言模型中获取知识（3.3节），以及下游融合模块的实现（3.4节）。最后，对训练方法进行简要说明（3.5节）。 

### 3.1 Task Definition

Formally, the task of multi-modal metaphor detection falls under the typical category of multi-modal classification problems. Given a set of cross-modal sample pairs, the task aims to determine whether metaphorical features are present and provide a classification result. Our work focuses on the detection of metaphors in image-text pairs, thus the task is represented as:  																		$Y=F\left(x^{I}, x^{T}\right) (1)$

形式上，多模态隐喻检测任务属于典型的多模态分类问题。**给定一组跨模态样本对，该任务旨在确定是否存在隐喻特征，并给出分类结果**。我们的工作重点是检测图像 - 文本对中的隐喻，因此该任务可表示为： $Y = F(x^I, x^T) \ (1)$ 

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250221113050838.png" alt="image-20250221113050674" style="zoom: 67%;" />

Figure 2: An illustration of C4MMD using the MLLM for multi-modal metaphor detection.

图2：使用多模态大语言模型进行多模态隐喻检测的C4MMD示意图。 

> 这张图片是一张幽默的生日祝福图，使用了双关语（pun）来创造幽默效果。图片中的鱼嘴里叼着一张卡片，上面写着“BEST FISHES ON YOUR BIRTHDAY”。这里的“BEST FISHES”是一个双关语，既可以直接理解为“最好的鱼”，也可以理解为“最好的祝愿”（best wishes），因为“fishes”与“wishes”发音相似。这种幽默的表达方式用来祝福生日，显得既有趣又独特。

where $x^{I}$and $x^{T}$ respectively denote the features of the image and text modalities. Our objective is to utilize a more effective method F to ensure that the classification result $\hat{Y}$ more closely aligns with the true value Y ∶

其中$x^{I}$和$x^{T}$分别表示图像模态和文本模态的特征。我们的目标是利用一种更有效的方法\(F\)，确保分类结果\(\hat{Y}\)更接近真实值\(Y\)。 

### 3.2 Overview

As shown in Figure2, the architecture of C4MMD consists of two primary components: a knowledge summarization module and a downstream structure for multi-model fusion.

如图2所示，C4MMD架构主要由两个部分组成：知识总结模块和用于多模型融合的下游结构。 

In the knowledge summarization module, we provide an image-text pair to the MLLM and design a three-step template with CoT prompting. The first two templates instruct the MLLM to focus exclusively on a single modality—either text or image, ignoring the other to generate explanations and insights. In the third step, the MLLM combines insights from both modalities. Based on previous analyses, the model achieves a deeper understanding and a fuller integration of both modalities.