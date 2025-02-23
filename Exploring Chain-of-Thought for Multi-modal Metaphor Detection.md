[Exploring Chain-of-Thought for Multi-modal Metaphor Detection](https://aclanthology.org/2024.acl-long.6.pdf)

探索思维链在多模态隐喻检测中的应用

[TOC]

> ​	隐喻，也称简喻、暗喻，[修辞](https://baike.baidu.com/item/修辞/175591?fromModule=lemma_inlink)学术语。隐喻是一种比喻，用一种事物暗喻另一种事物。巧妙地使用隐喻，对[表现手法](https://baike.baidu.com/item/表现手法/91087?fromModule=lemma_inlink)的生动、简洁、加重等方面起重要作用，比[明喻](https://baike.baidu.com/item/明喻/2109148?fromModule=lemma_inlink)更加形象.
>
> ①本体和[喻体](https://baike.baidu.com/item/喻体/245275?fromModule=lemma_inlink)是[并列关系](https://baike.baidu.com/item/并列关系/6147750?fromModule=lemma_inlink)。例如：
>
> 从喷泉里喷出来的都是水，从血管里流出来的都是血。（从血管里流出来的都是血）
>
> ②本体和喻体是[修饰关系](https://baike.baidu.com/item/修饰关系/3751250?fromModule=lemma_inlink)。例如：
>
> 这里是花的海洋（“海洋”修饰“花”）
>
> ③本体和喻体是注释关系。例如：
>
> 我爱北京——祖国的心脏。（北京——祖国的心脏）
>
> ④本体和喻体是[复指](https://baike.baidu.com/item/复指/55229929?fromModule=lemma_inlink)关系。例如：
>
> 让我们对土地倾注更强烈的感情吧！因为大地母亲的镣铐解除了......（大地母亲复指关系）

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



Overall, we utilize a CoT-based method called C4MMD to summarize knowledge from MLLMs and enhance metaphor detection in smaller models by fine-tuning them to link this knowledge with metaphors. The basic idea is shown in Figure 1, we first input images and text into the MLLM and obtain information describing the image, text, and their fusion. Furthermore, we have designed a downstream modality fusion structure, which is intended to translate supplementary(补充的；额外的；) information into metaphorical features for more accurate classification. Specifically, we have designed two auxiliary tasks focused on determining the presence of metaphors within the image and text modalities.

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

Formally, the task of multi-modal metaphor detection falls under the typical category of multi-modal classification problems. Given a set of cross-modal sample pairs, the task aims to determine whether metaphorical features are present and provide a classification result. Our work focuses on the detection of metaphors in image-text pairs, thus the task is represented as
$$
Y=F\left(x^{I}, x^{T}\right) (1)
$$


形式上，多模态隐喻检测任务属于典型的多模态分类问题。**给定一组跨模态样本对，该任务旨在确定是否存在隐喻特征，并给出分类结果**。我们的工作重点是检测图像 - 文本对中的隐喻，因此该任务可表示为： $Y = F(x^I, x^T) \ (1)$ 

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250221113050838.png" alt="image-20250221113050674" style="zoom: 67%;" />

Figure 2: An illustration of C4MMD using the MLLM for multi-modal metaphor detection.

图2：使用多模态大语言模型进行多模态隐喻检测的C4MMD示意图。 

> 这张图片是一张幽默的生日祝福图，使用了双关语（pun）来创造幽默效果。图片中的鱼嘴里叼着一张卡片，上面写着“BEST FISHES ON YOUR BIRTHDAY”。这里的“BEST FISHES”是一个双关语，既可以直接理解为“最好的鱼”，也可以理解为“最好的祝愿”（best wishes），因为“fishes”与“wishes”发音相似。这种幽默的表达方式用来祝福生日，显得既有趣又独特。

where $x^{I}$and $x^{T}$ respectively denote the features of the image and text modalities. Our objective is to utilize a more effective method F to ensure that the classification result $\hat{Y}$ more closely aligns with the true value Y ∶

其中$x^{I}$和$x^{T}$分别表示图像模态和文本模态的特征。我们的目标是利用一种更有效的方法F，确保分类结果$\hat{Y}$更接近真实值Y。 

### 3.2 Overview

As shown in Figure2, the architecture of C4MMD consists of two primary components: a knowledge summarization module and a downstream structure for multi-model fusion.

如图2所示，C4MMD架构主要由两个部分组成：知识总结模块和用于多模型融合的下游结构。 

In the knowledge summarization module, we provide an image-text pair to the MLLM and design a three-step template with CoT prompting. The first two templates instruct(指示) the MLLM to focus exclusively(唯一地) on a single modality—either text or image, ignoring the other to generate explanations and insights(了解，洞察力). In the third step, the MLLM combines insights from both modalities. Based on previous analyses, the model achieves a deeper understanding and a fuller integration of both modalities.

在**知识总结模块**中，我们向多模态大语言模型（MLLM）提供一对图像-文本数据，并使用思维链（CoT）提示设计了一个三步模板。**前两个模板指示多模态大语言模型只专注于单一模态——要么是文本，要么是图像**，忽略另一模态，以生成解释和见解。在第三步中，**多模态大语言模型结合了来自两种模态的见解。基于先前的分析**，该模型实现了对两种模态更深入的理解和更全面的整合。 

After obtaining additional textual information for different modalities from the MLLM, we merge this with the original texts to form a textual input. Similarly, the input image is treated as the visual modality input. The model then processes these inputs through modality-specific encoders to derive(得到，获得) feature vectors.

在从多模态大语言模型（MLLM）中获取了不同模态的额外文本信息后，**我们将这些信息与原始文本进行合并，形成一个文本输入**。同样地，将**输入图像作为视觉模态输入**。然后，模型通过特定模态的编码器对这些输入进行处理，以得出**特征向量。** 

In the multi-model fusion module, we scale and combine vectors from different modalities and develop a fine-grained(细粒度的) classifier. Specifically(具体地，特别地), we integrate the **supplementary(补充的) image description vector** with the visual modality input vector as the image vector, combine the text analysis vector with the textual input vector as the text vector, and merge these to form a cross-modal vector. These three vectors are then used for classification purposes. The classifier uses the cross-modal vector to detect metaphors, the image vector to identify imagedominated content, and the text vector for textdominated content. This approach enhances the use of multi-modal features for precise metaphor detection.

在**多模型融合模块**中，我们**对来自不同模态的向量进行缩放并组合，同时开发了一个细粒度的分类器**。具体来说，我们将**补充的图像描述向量与视觉模态输入向量整合，形成图像向量**；将**文本分析向量与文本输入向量相结合，构成文本向量**；然后**将这两个向量合并，形成一个跨模态向量**。<font size=4 color="bule" >这三个向量随后被用于分类任务。分类器利用跨模态向量来检测隐喻，使用图像向量识别以图像为主导的内容，利用文本向量识别以文本为主导的内容。这种方法加强了对多模态特征的运用，从而实现精确的隐喻检测。</font>

### 3.3 Knowledge Summarization from MLLMs Using the CoT Method

To guide MLLM in generating higher-quality and more informative features, we employ CoT prompting. This method directs the MLLMs to extract deeper information across modalities. We then utilize this supplementary(补充的；额外的；) information to assist the smaller model in achieving better semantic understanding and modality fusion. In conclusion, we construct the three-step prompts as follows.

为了引导多模态大语言模型（MLLM）生成更高质量且信息更丰富的特征，我们采用了**思维链（CoT）提示法**。这种方法指导多模态大语言模型跨模态提取更深入的信息。然后，我们**利用这些补充信息来帮助较小的模型实现更好的语义理解和模态融合。**总之，我们构建的三步提示如下。 

<font size=5 color="black">**STEP1**.</font> Initially, to ensure that the model concentrates on comprehending objects, scenes, or other visual elements in the image(Represented by x^{I} ) without interference from textual features, we guide the model to understand and interpret the image information based on a template Question1: 

> Question1: Please temporarily ignore the text in the image and describe the content in the image. Try to be concise while ensuring the correctness of your answers. 

This step can be formulated as follows:  


**第一步：**首先，为了确保模型**专注于理解图像（用$x^{I}$表示）中的物体、场景或其他视觉元素**，而不受文本特征的干扰，我们依据模板问题1来引导模型理解和解读图像信息：

>  问题1：请暂时忽略图像中的文本内容，并描述图像中的内容。尽量简洁，同时确保你的回答正确。

 这一步可以用以下公式表示： 
$$
m^{I}=MLLM\left(x^{I}, Question1 \right) (2)
$$
<font size=5 > **STEP2**. </font>Next, to better comprehend the hidden meanings in the text(Represented by $x^{T}$ ) while excluding(排除,不包括) any interference(干扰) from image features, we guide the model to understand and interpret the textual information according to a template Question2:

> Question2:  Please analyze the meaning of the text. Note that there may be homophonic memes and puns, distinguish and explain them but do not over interpret while ensuring the correctness of the answer and be concise.

This step can be formulated as follows:  

第二步：接下来，为了在排除任何图像特征干扰的情况下，更好地理解文本（用$x^{T}$表示）中的隐含意义，我们依据模板问题2来引导模型理解和解读文本信息： 

> 问题2：请分析这段文本的含义。请注意，其中可能存在谐音梗和双关语，请识别并对其进行解释，但不要过度解读，同时要确保答案的正确性，并尽量简洁。 

 这一步可以用以下公式表示： 
$$
m^{T}=MLLM\left(x^{T}, Question2 \right) (3)
$$
<font size=5 >**STEP3**</font>. Ultimately, we aspire(渴望，有志) for the model to synthesize(合成；综合) the results from the previous two steps(Represented by m^{I} and m^{T} ) and further integrate the image and text features( x^{I} and x^{T} ), thereby(从而；由此) obtaining more profound(深邃的，理解深刻的) cross-modal interaction information. We encourage the model to fuse features from different modalities according to template Question3: 

>  Question3: Please combine the image, text, and their description information and try to understand the deep meaning of the combination of the image and text. No need to describe images and text, only answer implicit meanings. Ensure the accuracy of the answer and try to be concise as much as possible. 

This step can be formulated as follows:  

第三步：最后，我们期望模型能够综合前两步的结果（分别由m^{I}和m^{T}表示），并进一步整合图像和文本特征（x^{I}和x^{T}），从而获得更深刻的跨模态交互信息。我们依据模板问题3来促使模型融合来自不同模态的特征： 

>  问题3：请结合图像、文本以及它们的描述信息，尝试理解图像和文本组合的深层含义。无需描述图像和文本，仅回答其隐含意义。确保答案的准确性，并尽可能简洁。 

这一步可以用以下公式表示： 
$$
m^{Mix }=MLLM\left(x^{I}, x^{T}, m^{I}, m^{T}, Question3 \right)
$$

### 3.4 Multi-modal Fusion for Metaphor Detection

After obtaining additional modal information generated by the MLLM, we designed a modal fusion architecture to facilitate inter-modal integration and effectively leverage the extra information produced by the MLLM to enhance metaphor detection capabilities.

在获取了由多模态大语言模型（MLLM）生成的额外模态信息后，我们设计了一种**模态融合架构，以促进模态间的整合，并有效地利用多模态大语言模型所产生的额外信息，从而提升隐喻检测能力。** 

#### 3.4.1 Modality-Specific Encoding

We use an **image encoder and a text encoder to obtain vectorized encodings of the image x^{I} and text x^{T}** for subsequent inter-modal fusion. Considering the additional information generated by the MLLM is presented in text form, we treat it as extra visual m^{I} , textual m^{T} , and mixed m^{M i x} information. This information is concatenated with the original text and then processed through the text encoder for computation.  
$$
\begin{aligned} & V=ViT-Encoder\left(x^{I}\right), \\ & T=XLMR-Encoder\left(x^{T}, m^{T}, m^{I}, m^{Mix }\right) \end{aligned}
$$
where V is the output of the image encoder, and T is the output of the text encoder.

我们使用一个**图像编码器和一个文本编码器来获取图像x^{I}和文本x^{T}的向量化编码，以便用于后续的模态间融合。**考虑到多模态大语言模型（MLLM）生成的额外信息是以文本形式呈现的，我们将其视为额外的视觉信息m^{I}、文本信息m^{T}和混合信息m^{Mix}。这些信息与原始文本连接起来，然后通过文本编码器进行处理计算。
$$
\begin{aligned} & V=ViT-Encoder\left(x^{I}\right), \\ & T=XLMR-Encoder\left(x^{T}, m^{T}, m^{I}, m^{Mix }\right) \end{aligned}
$$
其中V是图像编码器的输出，T是文本编码器的输出。 

To enable the text encoder to distinguish between texts from different modalities during computation, we adopt a method similar to BERT’s segment encoding by adding extra learnable parameter vectors for the text from each modality. The vectorized encoding Emb_{i} of the i -th word x_{i}$(x_{i} \in \{x^{T}, m^{T}, m^{I}, m^{M i x}\})$ entering the text encoder can be represented as follows:  
$$
E m b_{i}=E_{T}\left(x_{i}\right)+E_{P} (i)+E_{S}\left(segment\left(x_{i}\right)\right)
$$
 where E_{T} , E_{P} and E\_{S} represent learnable matrices for token embeddings, positional encodings, and segment embeddings, respectively. The term $ segment(x_{i}) \in (0,1,2,3) $ refers to the segment encoding of the word x_{i} , this encoding is specifically represented by the following formula: 
$$
segment\left(x_{i}\right)=\left\{\begin{array}{ll}1, & if x_{i} \in m^{I} \\ 2, & if x_{i} \in\left\{x^{T}, m^{T}\right\} \\ 3, & if x_{i} \in m^{M i x} \\ 0, & otherwise \end{array} \quad\right.
$$
为了**使文本编码器在计算过程中能够区分来自不同模态的文本，我们采用了一种类似于BERT的片段编码方法，即为每个模态的文本添加额外的可学习参数向量**。进入文本编码器的第i个单词$x_{i}(x_{i} \in\{x^{T}, m^{T}, m^{I}, m^{Mix}\})$的向量化编码Emb_{i}可以表示如下：_
$$
Emb_{i}=E_{T}(x_{i}) + E_{P}(i) + E_{S}(segment(x_{i}))
$$
 其中$E_{T}、E_{P}和E_{S}$分别表示用于词块嵌入、位置编码和片段嵌入的可学习矩阵。术语$segment (x_{i}) \in(0,1,2,3)$指的是单词x_{i}的片段编码，这种编码具体由以下公式表示：
$$
segment\left(x_{i}\right)=\left\{\begin{array}{ll}1, & if x_{i} \in m^{I} \\ 2, & if x_{i} \in\left\{x^{T}, m^{T}\right\} \\ 3, & if x_{i} \in m^{M i x} \\ 0, & otherwise \end{array} \quad\right.
$$

#### 3.4.2 Modality Funsion 

模态融合

Before modal fusion, to ensure the vector dimensions from both encoders are consistent, in the textual modality, we compute the average of all word vectors mean (T) as the vector representation of the entire sentence. For the visual modality, we take the vector of the CLS token V_{C L S} as the representation of the entire image. Then, we use a linear layer with a GeLU activation function (Hendrycks and Gimpel, 2016) to map it to the same feature space as the textual modality. The formula is represented as follows:  

在进行模态融合之前，为**确保两个编码器输出的向量维度一致**，在文本模态方面，我们计算**所有单词向量的平均值$\text{mean}(T)$**，以此作为**整个句子的向量表示**。在视觉模态方面，我们取分类（CLS）标记的向量**$V_{CLS}$作为整个图像的表示**。然后，我们使用一个带有高斯误差线性单元**（GeLU）激活函数**（亨德里克斯和金佩尔，2016年）的线性层，**将其映射到与文本模态相同的特征空间中**。公式表示如下： 
$$
V^{\text{reshape}} = \text{GeLU}(W_{v}V_{CLS} + b_{v}) \quad (8)
$$
Considering that the text information from different modalities generated by the large model has already undergone(经历，承受) a degree of fusion within the text encoder, we therefore **concatenate these two vectors from both modalities to obtain the final fused vector representation.** The formula for this process is as follows:  

考虑到**由大模型生成的来自不同模态的文本信息已经在文本编码器中进行了一定程度的融合，因此我们将来自两种模态的这两个向量连接起来**，以获得最终的融合向量表示。这一过程的公式如下：
$$
E^{Mix }=\left[V^{reshape }, mean (T)\right]
$$
Finally, we use a linear layer and a softmax classifier for metaphor classification.  

最后，我们使用一个线性层和一个softmax分类器来进行隐喻分类。
$$
\hat{y}=softmax\left(W_{Mix } E^{Mix }+b_{Mix }\right) \quad(10)
$$
Considering the diverse sources of metaphorical features, we employ two separate classifiers to categorize metaphors predominantly driven by either the image modality or the text modality. The aim is to force the detection of metaphorical features in both image and text before their fusion, thereby reducing the classification complexity for the final classifier. This approach of fine-grained metaphor detection is based on the following formula:  

考虑到隐喻特征的来源多样，**我们采用两个独立的分类器，分别对主要由图像模态或文本模态驱动的隐喻进行分类。其目的是在图像和文本的特征融合之前，就促使对两者中隐喻特征的检测，从而降低最终分类器的分类复杂度。**这种细粒度的隐喻检测方法基于以下公式：
$$
E^{I}=\left[V^{reshape }, mean \left(T_{m^I} \right)\right] (11)\\
E^{T}=mean\left(\left[T_{x^{T}}, T_{m^{T}}\right]\right)
$$

---

 要理解上面两个公式, 

 1. **公式 $E^{I}=\left[V^{\text{reshape}}, \text{mean} \left(T_{m^I} \right)\right] $**    

 - **$V^{\text{reshape}}$**: 这是对视觉模态（图像）的处理结果。在模态融合之前，为了使图像和文本的向量维度一致，从图像编码器得到的向量 V（通常是图像中 CLS 标记的向量 $V_{CLS}$）通过一个带有 GeLU 激活函数的线性层进行变换，得到 $V^{\text{reshape}}$。它代表了经过处理后，与文本模态特征空间相匹配的图像特征表示。    
 -  **$\text{mean} \left(T_{m^I} \right)$**: $T_{m^I}$ 表示文本编码向量中描述图像的部分。对这部分文本编码向量取平均值，得到一个综合的向量表示，它总结了文本中与图像相关的信息。    
 - ($E^{I}$: 将上述两个部分连接起来$\left[V^{\text{reshape}}, \text{mean} \left(T_{m^I} \right)\right]$，形成了一个新的向量 E^{I}。这个向量融合了图像本身的特征和文本中对图像描述的特征，用于后续对以图像模态为主导的隐喻特征的检测和分类。 

 2. **公式 $E^{T}=\text{mean}\left(\left[T_{x^{T}}, T_{m^{T}}\right]\right)$**    
    - **$T_{x^{T}}$**: 它代表原始文本 $x^{T}$ 的编码向量，即文本自身的特征表示。    
    - **$T_{m^{T}}$**: 表示文本编码向量中由多模态大语言模型（MLLM）生成的与文本模态相关的额外信息部分。    
    -  **$E^{T}$**: 先将 $T_{x^{T}}$ 和 $T_{m^{T}}$ 组合成一个向量 $\left[T_{x^{T}}, T_{m^{T}}\right]$，然后对这个组合向量取平均值，得到 E^{T}。这个 E^{T} 向量综合了原始文本的特征和来自 MLLM 的额外文本信息特征，用于后续对以文本模态为主导的隐喻特征的检测和分类。

------

Here, $T_{m^{I}}$ , $T_{x^{T}}$ and $T_{m^{T}}$ respectively represent the parts of the text encoding vector that describe the image and the text. Finally, two classifiers are used to categorize the metaphorical features in the text and the image. The formula for this classification process is as follows:  

在这里，$T_{m^{I}}$、$T_{x^{T}}$和$T_{m^{T}}$分别表示文本编码向量中描述图像和文本的部分。最后，使用两个分类器对文本和图像中的隐喻特征进行分类。该分类过程的公式如下： 

**有m的代码M多模态，I是图像，T是文本，大的T是向量**
$$
\hat{y}^{I}=softmax\left(W_{I} E^{I}+b_{I}\right)\\
\hat{y}^{T}=softmax\left(W_{T} E^{T}+b_{T}\right) \quad(14)
$$
In the above-mentioned formulas, $ W_{v} $ , $W_{Mix }$ , $W_{I}$ and $W_{T}$ are trainable parameter matrices; $b_{v} $ ,$ b_{M i x} $, $b_{I} $and $b_{T}$ represent bias matrices.

在上述公式中，$W_{v}$、 $W_{Mix}$ 、$W_{I}$ 和 $W_{T}$ 是可训练的参数矩阵；$b_{v}$、$b_{Mix}$、$b_{I}$ 和 $b_{T}$ 表示偏置矩阵。 

### 3.5 Training

The training objective of our multi-modal metaphor detection model involves the integration of three distinct loss functions, denoted as  $L_{I}$  , $L_{T}$ and $L_{M}$ The loss function is as follows: 
$$
\mathcal{L}=\frac{1}{\left|\mathcal{D}_{ME}\right|} \sum_{i=1}^{\left|\mathcal{D}_{ME}\right|} L_{C E}(\hat{Y}, Y) \quad(15)
$$
where $D_{ME}$ is the number of samples in the dataset, The loss formula is parameterized as $L={L_{I}, L_{T}, L_{M}}$ , with $\hat{Y}={\hat{y}, \hat{y}^{I}, \hat{y}^{T}}$ and Y representing the model’s predicted outcomes and the true values, $L_{C E}$ is the cross-entropy loss function. 

我们的多模态隐喻检测模型的训练目标涉及整合三个不同的损失函数，分别记为

$L_{I}$、$L_{T}$和$L_{M}$。损失函数如下： $\mathcal{L}=\frac{1}{\left|\mathcal{D}_{ME}\right|} \sum_{i = 1}^{\left|\mathcal{D}_{ME}\right|} L_{CE}(\hat{Y}, Y) \quad(15)$ 其中$\mathcal{D}_{ME}$是数据集中的样本数量。损失公式以$L = \{L_{I}, L_{T}, L_{M}\}$ 进行参数化，$\hat{Y}=\{\hat{y}, \hat{y}^{I}, \hat{y}^{T}\}$ ，$Y$ 表示模型的预测结果和真实值，$L_{CE}$ 是交叉熵损失函数。

To optimize the overall performance, we define the aggregate loss $L_{sum }$ as a weighted combination of these individual losses. The final loss function is formulated as:  

 为了优化整体性能，我们将总损失$L_{sum}$定义为这些单个损失的加权组合。最终的损失函数公式为：
$$
\mathcal{L}_{sum }=0.5 \cdot \mathcal{L}_{I}+0.5 \cdot \mathcal{L}_{T}+\mathcal{L}_{M} \quad(16)
$$

## 4 Experiments

In this section, we begin by introducing the dataset used to validate our method, as well as the experimental setup. Following this, we report the experimental results and provide an analysis of these outcomes.

在本节中，我们首先介绍用于验证我们方法的数据集，以及实验设置。在此之后，我们报告实验结果，并对这些结果进行分析。 

### 4.1 Data and Setting

We selected the **multi-modal metaphor dataset proposed by Xu et al.** (2022), which consists of 10,000 meme(模因) images collected from social media. Text information was extracted from these images using OCR methods to construct the multi-modal metaphor dataset, which includes 6,000 entries in Chinese and 4,000 in English. In addition to the classification labels for metaphors, they also annotated the source of the metaphors and their associated emotions.

我们选用了徐等人（2022年）提出的多模态隐喻数据集，该数据集由从社交媒体收集的10000张梗图组成。通过光学字符识别（OCR）方法从这些图片中提取文本信息，从而构建了多模态隐喻数据集，其中包含6000条中文数据和4000条英文数据。除了对隐喻进行分类标注外，他们还标注了隐喻的来源及其相关情感。 

All trained models were set with a learning rate of 1e-5, a batch size of 8, and were trained for 100 epochs with an early stopping mechanism in place. The dataset was randomly shuffled and divided into training, validation, and test sets in a 6:2:2 ratio. All experiments were conducted on a single 3090 24G GPU. The final results of our method were obtained by taking the average of five different random seeds, with the average single run time within 20-30 minutes. Finally, the model’s performance was evaluated based on the F1 score.

所有训练的模型均设置学习率为$1\times10^{-5}$ ，批量大小为$8$ ，并在设置了提前停止机制的情况下训练$100$ 个轮次。数据集被随机打乱，并按照$6:2:2$ 的比例划分为训练集、验证集和测试集。所有实验均在一块显存为$24G$ 的英伟达$3090$  GPU上进行。我们的方法最终的结果是通过**取五个不同随机种子的平均值得到的，单次运行的平均时间在$20$ 到$30$ 分钟之间。最后，基于F1值来评估模型的性能。** 

> - **提前停止机制(an early stopping mechanism)**
> - **目的**：其主要目的是防止模型在训练过程中出现过拟合，同时也能提高训练效率，避免不必要的计算资源浪费和时间消耗。
> - **触发条件**：通常会根据验证集上的模型性能指标来判断是否触发。例如，在连续若干个训练轮次（epochs）中，模型在验证集上的损失函数值不再下降，或者准确率、F1 值等评估指标没有明显提升甚至出现下降趋势，就可能满足提前停止的条件。

The Low-Rank Adaptation (LoRA Hu et al. (2021)) fine-tuning approach was adopted for finetuning LLMs. All of the settings followed those used in Alpaca-LoRA*.

采用了低秩自适应（LoRA，胡等人（2021））微调方法对大语言模型（LLMs）进行微调。所有设置均遵循在“羊驼-低秩自适应（**Alpaca-LoRA***）”中**所使用的那些设置。**  

### 4.2 Baseline Methods

#### Language Models

We tested several common pre-trained models for this task, including the AutoEncoder MBERT (Pires et al., 2019), XLM-R (Conneau et al.,2019), as well as the AutoRegressive models MT5 (Xue et al., 2020) and M-BART (Liu et al., 2020). Additionally, we evaluated the capabilities of LLMs on this task by using LLaMA2 (Touvron et al., 2023) and ChatGLM3 (Zeng et al., 2022), due to their strong performance in both Chinese and English corpora. We fine-tuned both models separately using LoRA.

针对这项任务，我们测试了几种常见的预训练模型，其中包括自编码器类型的多语言BERT（MBERT，皮雷斯等人，2019年）、跨语言语言模型XLM - R（康诺等人，2019年），以及自回归模型MT5（薛等人，2020年）和M - BART（刘等人，2020年）。此外，**鉴于LLaMA2（图夫龙等人，2023年）和ChatGLM3（曾等人，2022年）在中文和英文语料上均表现出色，我们评估了它们在这项任务中的能力。**我们分别使用低秩自适应（LoRA）方法对这两个模型进行了微调。 

#### Vision Models

We also tested models from the vision domain, including Convolutional Neural Network (CNN) models such as VGG (Simonyan and Zisserman, 2014), ResNet (He et al., 2016), and ConvNeXt (Liu et al., 2022), as well as models based on the Transformer architecture, like ViT (Dosovitskiy et al., 2020) and Swin Transformer (Liu et al., 2021).

我们还测试了来自视觉领域的模型，其中包括卷积神经网络（CNN）模型，例如VGG（西蒙扬和齐斯曼，2014年）、ResNet（何等人，2016年）和ConvNeXt（刘等人，2022年），以及基于Transformer架构的模型，如ViT（多索维茨基等人，2020年）和Swin Transformer（刘等人，2021年）。 

#### Multi-modal Models

In the multi-modal model domain, we selected VILT (Kim et al., 2021), BLIP2 (Li et al., 2023a), and InternLM-XComposer (Zhang et al., 2023c) to test their capabilities in addressing the metaphor detection task. All three models employ the Transformer architecture, yet they differ significantly in model size. We tested the capabilities of these MLLMs both in a zero-shot setting and with LoRA fine-tuning.

在多模态模型领域，我们选择了VILT（金等人，2021年）、BLIP2（李等人，2023年a）以及书生·多模态（InternLM-XComposer，张等人，2023年c）来测试它们处理隐喻检测任务的能力。这三个模型都采用了Transformer架构，但它们在模型规模上有显著差异。**我们在零样本设置以及使用低秩自适应（LoRA）微调的情况下，测试了这些多模态大语言模型（MLLM）的能力。** 

#### Other Related Works

We also explored other works related to our task, thereby lending more credibility to our comparative analysis. Below, we introduce these works in detail. 

- **CLIP** (Zhao et al., 2023): Evaluation of various models for hate meme detection task. We adopted best performance CLIP to evaluate its effectiveness in multi-modal metaphor detection tasks. 
-  **Vilio** (Muennighoff, 2020): An excellent method which achieves 2nd place in the Hateful Memes Challenge. It Uses OCR and entity recognition technologies to extract text and visual features from memes for better meme harmfulness detection tasks. 
-  **CoolNet** (Xiao et al., 2023): Extracting text syntactic structure to boost model’s sentiment analysis ability on Twitter multi-modal data. 
- **MultiCMET** (Zhang et al., 2023b): A baseline model for chinese multi-modal metaphor detection task. It uses the CLIP model to generate additional information to assist in the fusion between modalities.

我们还探索了与我们的任务相关的其他研究成果，从而使我们的对比分析更具可信度。下面，我们将详细介绍这些研究。 

- **CLIP（赵等人，2023年）**：该研究对用于仇恨梗图检测任务的各种模型进行了评估。我们采用了性能最佳的CLIP模型，来评估它在多模态隐喻检测任务中的有效性。 
- **Vilio（米尼霍夫，2020年）**：这是一种出色的方法，在“仇恨梗图挑战赛”中获得了第二名。它使用光学字符识别（OCR）和实体识别技术，从梗图中提取文本和视觉特征，以更好地完成梗图有害性检测任务。
-  **CoolNet（肖等人，2023年）**：该研究通过提取文本句法结构，来提升模型在推特多模态数据上的情感分析能力。 
-  **MultiCMET（张等人，2023年b）**：这是用于中文多模态隐喻检测任务的一个基线模型。它使用CLIP模型生成额外的信息，以辅助模态之间的融合。 

### 4.3 Main Results

![image-20250223135213125](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250223135213125.png)

Table 1: Results of different methods on the task of multi-modal metaphor detection.

表1：不同方法在多模态隐喻检测任务上的结果。

---

Table 1 shows the capabilities of different models in the task of multi-modal metaphor detection. Here we only evaluated the main classification results $\hat{y}$We did not assess the outcomes of the two subtasks  $ \hat{y}^{I}$  and  $ \hat{y}^{T}$   as the two subtasks which were primarily designed to serve the main task.

表1展示了不同模型在多模态隐喻检测任务中的能力。在此，我们仅评估了主要的分类结果 $ \hat{y} $ 。 我们没有评估两个子任务 $ \hat{y}^{I} $ 和 $ \hat{y}^{T} $ 的结果，因为这两个子任务主要是为了服务于主任务而设计的。 

Our approach achieved the best results in both Chinese and English sample sets. Considering the outcomes produced directly by the MLLM (InternLM-XComposer-7b), we allowed it to indirectly generate additional features for images and texts, effectively leveraging the large model’s capabilities. Coupled with a downstream classifier, this approach resulted in an additive effect.

**我们的方法在中文和英文样本集中均取得了最佳结果。**考虑到多模态大语言模型（书生·多模态-70亿参数版，InternLM-XComposer-7b）直接生成的结果，我们让它为图像和文本间接生成额外的特征，有效地利用了大模型的能力。再结合下游的分类器，这种方法产生了叠加效应。 

The performance of multi-modal models varied widely, with most models not surpassing language models. This underscores the importance of textual modality in recognizing multi-modal metaphors. MLLMs did not perform well in zero-shot scenarios, partly due to our designed prompt templates. However, the primary reason is the models’ inability to understand the task. Encouragingly, after finetuning BLIP2, its capabilities surpassed all other comparative methods. This demonstrates the benefit of interaction between image and text modalities in the task and how large models can effectively understand and address this task after fine-tuning.

**多模态模型的表现差异很大，大多数模型的表现都没有超过语言模型。**这凸显了文本模态在识别多模态隐喻方面的重要性。多模态大语言模型（MLLM）在零样本场景下表现不佳，部分原因在于我们所设计的提示模板。然而，主要原因是这些模型无法理解这项任务。令人鼓舞的是，在对BLIP2进行微调后，它的能力超过了所有其他对比方法。这表明了在这项任务中图像和文本模态之间进行交互的益处，以及大模型在经过微调后如何能够有效地理解和处理这项任务。 

In related work, studies closely aligned with our own, such as those by Zhang et al. (2023b) and Muennighoff (2020), have achieved competitive performances. However, Twitter sentiment classification by Xiao et al. (2023), which differs somewhat from our task, consequently showed weaker performance.

在相关研究中，与我们的研究密切相关的一些成果，比如张等人（2023年b）和米尼霍夫（2020年）所做的研究，已经取得了具有竞争力的成果。然而，肖等人（2023年）所做的推特情感分类研究，与我们的任务在一定程度上有所不同，因此其表现相对较弱。 

### 4.4 Influence of Different Factors

> 1. **ACC（Accuracy）**：即准确率，指的是模型预测正确的样本数占总样本数的比例。它反映了模型在所有预测中正确预测的总体程度。例如，在一个文本分类任务中，共有100个样本，模型正确分类了80个，那么该模型的准确率ACC就是80%。
> 1.  **P.（Precision）**：即精确率，也叫查准率。它是指模型预测为正类的样本中，实际为正类的样本所占的比例。比如在垃圾邮件分类中，模型预测出100封邮件是垃圾邮件，其中实际上确实是垃圾邮件的有80封，那么精确率就是80÷100 = 80%。 
> 1.  **R.（Recall）**：即召回率，也叫查全率。它是指实际为正类的样本中，被模型正确预测为正类的样本所占的比例。继续以垃圾邮件分类为例，假如总共有120封垃圾邮件，模型正确识别出了其中的80封，那么召回率就是80÷120≈66.7% 。 
> 1.  **F1.（F1-score）**：是精确率和召回率的调和平均数，它综合考虑了精确率和召回率两个指标，能够更全面地反映模型的性能。当精确率和召回率都较高时，F1值也会较高。其计算公式为\(F1 = 2×\frac{Precision×Recall}{Precision + Recall}\)。 这个表格通过这些指标展示了不同模型在特定任务中的表现情况，方便对各模型的性能进行比较和评估。 


| Model | ACC  | P.   | R.   | F1.  |
| --- | ---- | ---- | ---- | ---- |
| Ours | 87.70 | 83.33 | 81.58 | 82.44 |
| 	-Fusion model | 85.66 | 77.87 | 83.12 | 80.41 |
| 	-CoT features | 85.06 | 78.42 | 79.75 | 79.08 |
| 	-Vision encoder | 86.25 | 78.36 | 84.53 | 81.33 |

Table 2: Ablation study for the components in the model on metaphor detection.

表2：针对模型中用于隐喻检测的各个组件的消融研究。 

> “Ablation study” 常见的意思是 “消融研究”，在机器学习和深度学习等领域中，它是一种用于评估模型中各个组件或部分重要性的研究方法。
>
> 具体来说，消融研究通过逐步移除、修改或禁用模型中的某些组件、层、特征或操作，然后观察这些改变对模型性能（如准确率、损失值、F1 分数等）的影响。如果移除某个组件后，模型性能大幅下降，那就表明该组件对模型的功能和效果起着关键作用；反之，如果性能变化不大，则说明该组件的重要性相对较低。
>
> 通过消融研究，研究人员可以更好地理解模型的工作机制，确定哪些部分是必要的，哪些部分可能是冗余的，从而优化模型结构，提高模型性能，同时也有助于解释模型的行为和决策过程。在上述文本中，“Ablation study for the components in the model on metaphor detection” 即针对用于隐喻检测的模型中各个组件所进行的消融研究。

Table 2 shows the effects demonstrated by our model after undergoing ablation experiments. 

Replacing the fusion structure in the model with a linear layer resulted in a significant decrease in performance. This suggests the necessity of additional fusion structures to help the model understand the extra features generated by the MLLM. Moreover, eliminating the CoT generation method of the MLLM, and relying solely on a one-step generation method, led to an even more noticeable performance drop. This also indicates that the CoT method can generate better additional features, thereby assisting downstream models in making more accurate judgments. 

Interestingly, the performance of the model declined only slightly when we removed **the image processing module**. This indicates that the MLLM can provide a certain level of visual information for smaller models, but more comprehensive information still requires the contribution of vision models.

表2展示了我们的模型在经过消融实验后所呈现出的效果。

 **用一个线性层替换模型中的融合结构，导致了性能的显著下降。这表明需要额外的融合结构来帮助模型理解由多模态大语言模型（MLLM）生成的额外特征。**此外，去除多模态大语言模型的思维链（CoT）生成方法，而仅依赖于单步生成方法，会导致性能出现更为明显的下降。这也表明思维链方法能够生成更好的额外特征，从而帮助下游模型做出更准确的判断。 

有趣的是，当我们移除图像处理模块时，模型的性能仅出现了轻微下降(**F1分数**)。这表明多模态大语言模型能够为较小的模型提供一定程度的视觉信息，但更全面的信息仍需要视觉模型的贡献。 

### 4.5 The Impact of Different Language Vision Model Combinations

We tested the capabilities of multiple visual and textual models during modal fusion. The language model was uniformly set to MBERT when testing vision models and the ViT was used consistently when testing language models.

我们在模态融合过程中**测试了多个视觉模型和文本模型的能力**。在测试视觉模型时，语言模型统一设置为多语言双向编码器表征（MBERT）；而在测试语言模型时，则始终使用视觉Transformer（ViT）模型。 

-----


| VM | LM | ACC | P. | R. | F1. | 
| --- | --- | --- | --- | --- | --- |
| ResNet | M-BERT | 82.38 | 78.29 | 69.48 | 73.62 |
| VGG |M-BERT | 85.86 | 84.60 | 73.42 | 78.61 |
| ViT | M-BERT |85.75 | 81.73 | 76.99 | 79.27 |
| ViT | M-T5 | 76.66 | 68.51 | 62.64 | 65.44 | 
| ViT | M-BART | 80.21 | 70.97 | 75.14 | 72.92 |
| ViT | XLMR | 86.39 | 83.68 | 76.54 | 79.92 |

Table 3: The impact of different language and vision model combinations on the metaphor detection task, VM for Vision Model and LM for Language Model. We then use a linear layer to fuse the features of two modalities.

表3：不同的语言模型和视觉模型组合对隐喻检测任务的影响，其中VM代表视觉模型，LM代表语言模型。然**后我们使用一个线性层来融合两种模态的特征。** 

----
From the data in Table 3 and Table 1, although in single modality settings, the vision model VGG and the textual model M-T5 achieved the best performance, the combination of ViT and XLM-R outperformed all others upon modal fusion. 

The combinations of ResNet + MBERT and VGG + MBERT are also baseline models proposed by Met-Meme (Xu et al., 2022). According to the results, we reported the same results as them.

从表3和表1中的数据来看，尽管在单模态设置下，视觉模型VGG和文本模型M-T5取得了最佳性能，**但在模态融合后，视觉Transformer（ViT）与跨语言语言模型（XLM-R）的组合表现优于其他所有模型组合**。

 残差网络（ResNet）与多语言双向编码器表征（MBERT）的组合以及VGG与MBERT的组合，也是由“Met-Meme”（徐等人，2022年）提出的基线模型。根据结果，我们得出了与他们相同的结论。 

### 4.6 The Impact of Language Model Size

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250223191902371.png" alt="image-20250223191902371" style="zoom: 67%;" />

Figure 3: The effect of different sizes of models with or without CoT generation and the rate of improvement. We controlled the intercept of the model size between 0-1 to show the effect of improvement on a single figure.

图3：不同规模的模型在有无思维链（CoT）生成情况下的效果以及提升比例。我们将模型规模的取值范围控制在0到1之间，以便在单张图表中展示性能提升的效果。 

Figure 3 illustrates the abilities of models of different sizes under our architecture. Considering that the improvement ratio is generally between 0 and 1, while the model size is typically in the hundreds of millions, we divided all model sizes by 400 million to scale them between 0 and 1, allowing us to display both model size and improvement on the same graph. It was evident that as the model size increased, especially when the model was initially small, there was a progressively noticeable performance improvement. When the model was too small, the additional textual information did not yield positive effects; rather, it could have the potential to negatively impact the model’s performance. It was only when the model size was increased that the model became capable of understanding longer contextual information.  

图3展示了在我们的架构下不同规模模型的能力。考虑到提升比例通常在0到1之间，而**模型规模一般以亿为单位**，我们将所有模型的规模除以4亿，以便将其**缩放至0到1之间**，这样我们就能在**同一张图上同时展示模型规模和性能提升情况**。很明显，**随着模型规模的增大，尤其是在模型最初较小时，性能提升逐渐变得显著。**当模型规模过小时，额外的文本信息不会产生积极效果；相反，它有可能对模型的性能产生负面影响。**只有当模型规模增大时，模型才能够理解更长的上下文信息。** 

### 4.7 Case Study

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250223193432707.png" alt="image-20250223193432707" style="zoom:67%;" />

<p align="center">Figure 4: Examples of case study.</p>

> 第一个例子 
>
> **结果** 标签：正确 我们的模型判断：正确 √ 基线模型判断：错误 × (是否含有隐喻意义)
>
> **输入** 文字：“I been watching too many cute seal videos”（我看了太多可爱的海豹视频） 
>
> **InternLM的三步输出** 
>
> 图像描述：图片展示的是一个特写镜头，画面中的土豆上画着一张海豹的脸。
>
> 图像中的文字含义：文字内容是 “我看了太多可爱的海豹视频”。
>
> 图像和文字结合传达的隐含意义：图像与文字的组合带有幽默和讽刺意味，暗示观看者看了太多可爱的海豹视频，证据就是这个画着海豹脸的土豆。 
>
> 第二个例子 
>
> **结果** 标签：错误 我们的模型判断：错误 √ 基线模型判断：正确 × 
>
> **输入** 文字：“Everyone: sit like a lady Me:”（大家：像淑女一样坐 我：） **InternLM的三步输出** 
>
> 图像描述：图片中有一位穿着红色连衣裙的女士坐在椅子上。她似乎在摆姿势拍照，双腿交叉，一只手放在臀部。 
>
> 图像中的文字含义：文字 “大家：像淑女一样坐 我：” 是一个谐音梗。意思是每个人都应该像淑女一样坐，这暗示着要有良好的举止和得体的行为。 
>
> 图像和文字结合传达的隐含意义：女士的姿势和文字说明相结合，营造出一种幽默和讽刺的语气，暗示这位女士并没有按照 “像淑女一样坐” 的建议去做。 

To further explore the effectiveness of our proposed model, we select two examples from the testing dataset illustrated in Figure 4. 

The first example demonstrates an image-led metaphor. By directly comparing a seal with a potato, it depicts the consequences of looking at too many cute seals. The MLLM, through its understanding of the image, accurately recognized the resemblance between the seal and the potato, thereby aiding the downstream model in making the correct judgment.

 In the second example, the MLLM identified features from both the image and text, and then combined these to correctly understand the humorous meaning expressed in the meme. The downstream model accurately recognized that it did not contain metaphorical features. In contrast, methods lacking the additional information from the large model judged it to be metaphorical based solely on the phrase "like a lady," leading to a misjudgment.

为了进一步探究我们所提出模型的有效性，我们从测试数据集中选取了两个示例，如图4所示。

 第一个示例展示了一个以图像为主导的隐喻。通过将一只海豹直接与一个土豆进行对比，它描绘了看了太多可爱海豹所产生的后果。多模态大语言模型（MLLM）通过对图像的理解，准确地识别出海豹和土豆之间的相似之处，从而帮助下游模型做出了正确的判断。 

在第二个示例中，多模态大语言模型（MLLM）从图像和文本中识别出特征，然后将这些特征结合起来，正确理解了该梗图所表达的幽默含义。下游模型准确地识别出它不包含隐喻特征。相比之下，那些缺乏来自大模型额外信息的方法，仅仅根据“像一位女士”这个短语就判定它是隐喻，从而导致了误判。 

## 5 Conclusion

Our study aimed to tackle the challenges of multimodal metaphor interpretation by leveraging advanced MLLMs. We designed a three-step method with CoT-prompting to extract richer information from both images and text. Augmented knowledge from MLLMs proved crucial in enhancing smaller models to grasp metaphorical features within each modality and in the fusion of modalities. This work not only advances multi-modal metaphor detection but also paves the way for future research exploring the potential of MLLMs in addressing complex language and vision challenges.

我们的研究旨在通过利用先进的多模态大语言模型（MLLM）来应对多模态隐喻解读方面的挑战。我们设计了一种带有思维链提示的三步法，以便从图像和文本中提取更丰富的信息。**事实证明，来自多模态大语言模型的增强知识对于提升较小规模的模型，使其能够掌握各模态内以及模态融合中的隐喻特征至关重要。**这项工作不仅推动了多模态隐喻检测的发展，还为未来探索多模态大语言模型在应对复杂的语言和视觉挑战方面的潜力的研究铺平了道路。 
