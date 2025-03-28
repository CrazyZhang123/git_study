## A Survey on Multimodal Large Language Models

多模态大语言模型综述

链接: [ A Survey on Multimodal Large Language Models](https://arxiv.org/abs/2306.13549)

## Abstract

Recently, Multimodal Large Language Model (MLLM) represented by GPT-4V has been a new rising research hotspot, which uses powerful Large Language Models (LLMs) as a brain to perform multimodal tasks. The surprising emergent(涌现) capabilities of MLLM, such as writing stories based on images and OCR-free(无OCR) math reasoning, are rare(稀有的,罕见的) in traditional multimodal methods, suggesting a potential path to artificial general intelligence. To this end(为此), both academia and industry have endeavored to(致力于) develop MLLMs that can compete with or even better than GPT-4V, pushing the limit of research at a surprising speed. In this paper, we aim to trace(追踪；追溯) and summarize the recent progress of MLLMs. First of all, we present the basic formulation((政策、计划等的)制定，构想；) of MLLM and delineate(描述) its related concepts, including architecture, training strategy and data, as well as evaluation. Then, we introduce research topics about how MLLMs can be extended to support more granularity(粒度), modalities(模态), languages(语言), and scenarios(场景). We continue with **multimodal hallucination(多模态幻觉)** and extended techniques, including Multimodal ICL (M-ICL), Multimodal CoT (M-CoT), and LLM-Aided Visual Reasoning (LAVR). To conclude the paper, we discuss existing challenges and point out promising(有希望的；有前途的；) research directions. **In light of(鉴于，根据)** the fact that the era(时代) of MLLM has only just begun, we will keep updating this survey and hope it can inspire more research. An associated GitHub link collecting the latest papers is available at https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models.

**Index Terms—Multimodal Large Language Model, Vision Language Model, Large Language Model.**

**摘要**：最近，以GPT-4V为代表的多模态大语言模型（MLLM）成为新的研究热点。它以强大的大语言模型（LLMs）为核心，执行多模态任务。MLLM展现出的惊人涌现能力，如基于图像编写故事和无OCR的数学推理，在传统多模态方法中很少见，这为通用人工智能提供了一条潜在路径。为此，学术界和工业界都致力于开发能与GPT-4V竞争甚至超越它的MLLM，以惊人的速度推动着研究的极限。在本文中，我们旨在追踪并总结MLLM的最新进展。首先，我们介绍MLLM的基本概念，阐述其相关概念，包括架构、训练策略、数据以及评估。然后，我们介绍有关如何扩展MLLM以支持更细粒度、更多模态、更多语言和更多场景的研究课题。接着，我们探讨多模态幻觉问题以及扩展技术，包括**多模态上下文学习（M-ICL）、多模态思维链（M-CoT）和大语言模型辅助视觉推理（LAVR）**。在文章结尾，我们讨论现有挑战并指出有前景的研究方向。鉴于MLLM时代才刚刚开始，我们将持续更新本综述，希望它能激发更多研究。收集最新论文的相关GitHub链接为：https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models 。 

**关键词——多模态大语言模型，视觉语言模型，大语言模型 **

## 1 Introduction

RECENT years have seen the remarkable progress of LLMs [1], [2], [3], [4], [5]. By scaling up data size and model size, these LLMs raise **extraordinary(卓越的，非凡的)** emergent abilities, **typically including instruction following [5], [6], InContext Learning (ICL) [7], and Chain of Thought (CoT)** [8]. Although LLMs have demonstrated surprising zero/fewshot reasoning performance on most Natural Language Processing (NLP) tasks, they are inherently(固有的) “blind” to vision since they can only understand discrete text. Concurrently(同时的), Large Vision Models (LVMs) can see clearly [9], [10], [11], [12], but commonly lag in reasoning.

近年来，大语言模型（LLMs）取得了显著进展[1], [2], [3], [4], [5]。通过扩大数据规模和模型规模，这些大语言模型展现出非凡的涌现能力，通常包括**指令遵循[5], [6]、上下文学习（ICL）[7]和思维链（CoT）[8**]。尽管大语言模型在大多数自然语言处理（NLP）任务上展现出惊人的零样本/少样本推理性能，但它们本质上对视觉 “视而不见”，因为它们只能理解离散的文本。与此同时，大型视觉模型（LVMs）能够清晰地 “看到” 图像[9], [10], [11], [12]，但通常在推理方面存在不足。 

In light of this **complementarity(互补性),** LLM and LVM run towards each other, leading to the new field of Multimodal Large Language Model (MLLM). Formally, it refers to the LLM-based model with the ability to receive, reason, and output with multimodal information. Prior to MLLM, there have been a lot of works devoted to multimodality, which can be divided into discriminative [13], [14], [15] and generative [16], [17], [18] paradigms. CLIP [13], as a representative of the former, projects visual and textual information into a unified representation space, building a bridge for downstream multimodal tasks. In contrast, OFA [16] is a representative of the latter, which unifies multimodal tasks in a sequence-to-sequence manner. MLLM can be classified as the latter according to the sequence operation, but it manifests two representative traits compared with the traditional counterparts: (1) MLLM is based on LLM with billionscale parameters, which is not available in previous models. (2) MLLM uses new training paradigms to unleash its full potential, such as using multimodal instruction tuning [19], [20] to encourage the model to follow new instructions. Armed with the two traits, MLLM exhibits new capabilities, such as writing website code based on images [21], understanding the deep meaning of a meme [22], and OCR-free math reasoning [23].

鉴于大语言模型（LLM）和大型视觉模型（LVM）之间的这种互补性，二者相互融合，催生出了多模态大语言模型（MLLM）这一崭新的研究领域。从形式上来说，MLLM是一种基于LLM构建的模型，具备接收多模态信息、进行推理以及输出多模态结果的能力。在MLLM出现之前，已有诸多针对多模态的研究成果，这些研究大致可划分为**判别式[13, 14, 15]和生成式[16, 17, 18]两种范式**。**以CLIP[13]为代表的判别式范式，将视觉信息和文本信息映射到一个统一的表征空间中，为后续的多模态任务奠定了基础。而以OFA[16]为代表的生成式范式，则采用序列到序列的方式来统一处理多模态任务**。从序列操作的角度来看，MLLM可归属于生成式范式，但与传统的生成式模型相比，它呈现出**两个显著特征**：其一，MLLM基于拥有数十亿参数规模的LLM构建，这是以往模型所不具备的；其二，MLLM采用全新的训练范式来充分挖掘自身潜力，例如运用多模态指令微调[19, 20]的方法，促使模型能够更好地理解并执行新的指令。凭借这两个特性，MLLM展现出一系列全新的能力，**比如能够基于图像编写网站代码[21]、理解表情包的深层含义[22]，以及实现无光学字符识别（OCR）的数学推理[23] 。** 

Ever since the release of GPT-4 [3], there has been a research frenzy over MLLMs because of the amazing multimodal examples it shows. Rapid development is fueled by efforts from both academia and industry. Preliminary research on MLLMs focuses on text content generation grounded in text prompts and image [20], [24]/video [25], [26]/audio [27]. Subsequent works have expanded the capabilities or the usage scenarios, including: (1) **Better granularity support**. Finer control on user prompts is developed to support specific regions through boxes [28] or a certain object through a click [29]. (2) **Enhanced support on input and output modalities** [30], [31], such as image, video, audio, and point cloud. Besides input, projects like NExT-GPT [32] further support output in different modalities. (3) Improved language support. Efforts have been made to extend the success of MLLMs to other languages (e.g. Chinese) with relatively limited training corpus [33], [34]. (4) Extension to more realms and usage scenarios. Some studies transfer the strong capabilities of MLLMs to other domains such as medical image understanding [35], [36], [37] and document parsing [38], [39], [40]. Moreover, multimodal agents are developed to assist in real-world interaction, e.g. embodied agents [41], [42] and GUI agents [43], [44], [45]. An MLLM timeline is illustrated in Fig. 1.

自GPT-4发布以来[3]，由于其展示出的惊人多模态示例，引发了对多模态大语言模型（MLLMs）的研究热潮。学术界和工业界的共同努力推动了该领域的快速发展。**对MLLMs的初步研究主要集中在基于文本提示和图像[20, 24]、视频[25, 26]或音频[27]生成文本内容。**随后的研究工作扩展了其能力和应用场景，包括： (1) **更好的粒度支持。**开发了对用户提示的更精细控制，通过边界框支持特定区域[28]，或通过点击支持特定对象[29]。 (2) **增强的输入和输出模态支持**[30, 31]，如图像、视频、音频和点云。除了输入，像NExT-GPT[32]这样的项目进一步支持不同模态的输出。 (3) **改进的语言支持。**人们致力于将MLLMs的成功扩展到训练语料相对有限的其他语言（如中文）[33, 34]。 (4) **扩展到更多领域和应用场景**。一些研究将MLLMs的强大能力应用于其他领域，如医学图像理解[35, 36, 37]和文档解析[38, 39, 40]。此外，还开发了多模态智能体来辅助现实世界中的交互，如具身智能体[41, 42]和图形用户界面（GUI）智能体[43, 44, 45]。图1展示了MLLMs的发展时间线。 

In view of such rapid progress and the promising results of this field, we write this survey to provide researchers with a grasp of the basic idea, main method, and current progress of MLLMs. Note that we mainly focus on visual and language modalities, but also include works involving other modalities like video and audio. Specifically, we cover the most important aspects of MLLMs with corresponding summaries and open a GitHub page that would be updated in real time. To the best of our knowledge, this is the first survey on MLLM.

鉴于该领域的快速发展以及取得的可观成果，我们撰写了这篇综述，旨在帮助研究人员了解多模态大语言模型（MLLM）的基本概念、主要方法和当前进展。需要注意的是，我们主要聚焦于视觉和语言模态，但也涵盖了涉及视频、音频等其他模态的研究成果。具体而言，我们对MLLM最重要的几个方面进行了总结，并开设了一个GitHub页面，会对相关内容进行实时更新。据我们所知，这是首篇关于MLLM的综述。 

![image-20250319172846317](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250319172846434.png)

Fig. 1: A timeline of representative MLLMs. We are witnessing rapid growth in this field. More works can be found in our released GitHub page, which is updated daily.

图1：代表性多模态大语言模型（MLLMs）的发展时间线。我们正见证着这个领域的快速发展。更多相关成果可在我们发布的GitHub页面上查看，该页面每日更新。 

The following parts of the survey are structured as such: the survey starts with a comprehensive review of the essential aspects of MLLMs, including (1) Mainstream architectures (§2); (2) A full recipe of training strategy and data (§3); (3) Common practices of performance evaluation (§4). Then, we delve into a deeper discussion on some important topics about MLLMs, each focusing on a main problem: (1) What aspects can be further improved or extended (§5)? (2) How to relieve the multimodal hallucination issue (§6)? The survey continues with the introduction of three key techniques (§7), each specialized in a specific scenario: MICL (§7.1) is an effective technique commonly used at the inference stage to boost few-shot performance. Another important technique is M-CoT (§7.2), which is typically used in complex reasoning tasks. Afterward, we delineate a general idea to develop LLM-based systems to solve composite reasoning tasks or to address common user queries (§7.3). Finally, we finish our survey with a summary and potential research directions.

本综述的后续部分结构如下：**首先全面回顾多模态大语言模型（MLLM）的关键方面**，包括：（1）**主流架构**（第2节）；（2）**完整的训练策略和数据方法**（第3节）；（3）**性能评估的常用做法**（第4节）。然后，我们深入探讨一些关于MLLM的重要议题，每个议题聚焦一个主要问题：（1）**哪些方面可以进一步改进或扩展**（第5节）？（2）如何**缓解多模态幻觉问题**（第6节）？

本综述接着介绍三种关键技术（第7节），每种技术都针对特定场景：**多模态上下文学习（M-ICL，第7.1节）是一种在推理阶段常用的有效技术，用于提升少样本学习性能。另一种重要技术是多模态思维链（M-CoT，第7.2节），通常用于复杂推理任务。**之后，我们阐述开发基于大语言模型的系统以解决复合推理任务或处理常见用户查询的总体思路（第7.3节）。最后，我们通过总结和提出潜在研究方向来结束本综述。 

## 2 ARCHITECTURE

A typical MLLM can be abstracted into three modules, i.e. a pre-trained modality encoder, a pre-trained LLM, and a modality interface to connect them. Drawing an analogy to humans, modality encoders such as image/audio encoders are human eyes/ears that receive and pre-process optical/acoustic signals, while LLMs are like human brains that understand and reason with the processed signals. In between, the modality interface serves to align different modalities. Some MLLMs also include a generator to output other modalities apart from text. A diagram of the architecture is plotted in Fig. 2. In this section, we introduce each module in sequence.

**一个典型的多模态大语言模型（MLLM）可以抽象为三个模块，即预训练的模态编码器、预训练的大语言模型（LLM）以及连接它们的模态接口。**打个比方，就人类而言，图像/音频编码器这类模态编码器就如同人类的眼睛/耳朵，负责接收和预处理光/声信号，而大语言模型则类似于人类的大脑，能够对经过处理的信号进行理解和推理。在这两者之间，模态接口起到对齐不同模态的作用。一些多模态大语言模型还包含一个生成器，用于输出除文本之外的其他模态内容。架构示意图如图2所示。在本节中，我们将依次介绍每个模块。 

### 2.1 Modality encoder

模态编码器

The encoders compress raw information, such as images or audio, into a more compact representation. Rather than training from scratch, a common approach is to use a pretrained encoder that has been aligned to other modalities. For example, CLIP [13] incorporates a visual encoder semantically aligned with the text through large-scale pretraining on image-text pairs. Therefore, it is easier to use such initially pre-aligned encoders to align with LLMs through alignment pre-training (see §3.1).

这些编码器**将图像或音频等原始信息压缩为更紧凑的表示形式。常见的做法并非从头开始训练，而是使用一个已与其他模态对齐的预训练编码器。**例如，**CLIP[13]包含一个视觉编码器，它通过对图像-文本对进行大规模预训练，在语义上与文本实现了对齐。因此，使用这种预先初步对齐的编码器，通过对齐预训练（见3.1节）来与大语言模型（LLMs）对齐会更加容易。** 

The series of commonly used image encoders are summarized in Table 1. Apart from vanilla CLIP image encoders [13], some works also explore using other variants. For example, MiniGPT-4 [21] adopts an EVA-CLIP [47], [48] (ViT-G/14) encoder, which is trained with improved training techniques. In contrast, Osprey [29] introduces a convolution-based ConvNext-L encoder [46] to utilize higher resolution and multi-level features. Some works also explore encoder-free architecture. For instance, the image patches of Fuyu-8b [49] are directly projected before sending to LLMs. Thus, the model naturally supports flexible image resolution input.

常用的一系列图像编码器总结在表1中。除了普通的CLIP图像编码器[13]之外，一些研究工作还探索使用其他变体。例如，MiniGPT-4[21]采用了EVA-CLIP[47, 48]（ViT-G/14）编码器，该编码器是使用改进的训练技术进行训练的。相比之下，Osprey[29]引入了基于卷积的ConvNext-L编码器[46]，以利用更高的分辨率和多层次特征。一些研究还探索了无编码器架构。例如，Fuyu-8b[49]的图像块在发送到大语言模型之前直接进行投影。因此，该模型自然支持灵活的图像分辨率输入。 

TABLE 1: A summary of commonly used image encoders.表1：常用图像编码器概述。 

| Variants | Pretraining Corpus | Resolution | Samples (B) | Parameter Size (M) |
| --- | --- | --- | --- | --- | 
| OpenCLIP-ConvNext-L [ 46 ] | LAION-2B | 320 | 29 | 197.4 |
| CLIP-ViT-L/14 [ 13 ] | OpenAI’s WIT | 224/336 | 13 | 304.0 | 
| EVA-CLIP-ViT-G/14 [ 47 ] | LAION-2B,COYO-700M | 224 | 11 | 1000.0 | | OpenCLIP-ViT-G/14 [ 46 ] | LAION-2B | 224 | 34 | 1012.7 |
| OpenCLIP-ViT-bigG/14 [ 46 ] | LAION-2B | 224 | 34 | 1844.9 |

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250320112052833.png" alt="image-20250320112052730" style="zoom:67%;" />

Fig. 2: An illustration of typical MLLM architecture. It includes an encoder, a connector, and a LLM. An optional generator can be attached to the LLM to generate more modalities besides text. The encoder takes in images, audios or videos and outputs features, which are processed by the connector so that the LLM can better understand. There are broadly three types of connectors: projection-based, querybased, and fusion-based connectors. The former two types adopt token-level fusion, processing features into tokens to be sent along with text tokens, while the last type enables a feature-level fusion inside the LLM.

图2：典型多模态大语言模型（MLLM）架构示意图。它包括一个编码器、一个连接器和一个大语言模型（LLM）。可以在大语言模型上附加一个可选的生成器，以便除了文本之外还能生成更多模态的内容。编码器接收图像、音频或视频，并输出特征，这些特征由连接器进行处理，从而使大语言模型能够更好地理解。连接器大致有三种类型：**基于映射的连接器、基于查询的连接器和基于融合的连接器(图中下面的三部分)。**前两种类型采用标记级融合，将特征处理成标记，以便与文本标记一起发送，而最后一种类型则在大语言模型内部实现特征级融合。 

When choosing encoders, one often considers factors like resolution, parameter size, and pretraining corpus. Notably, many works have empirically verified that using higher resolution can achieve remarkable performance gains [34], [50], [51], [52]. The approaches for scaling up input resolution can be categorized into direct scaling and patch-division methods. The direct scaling way inputs images of higher resolutions to the encoder, which often involves further tuning the encoder [34] or replacing a pre-trained encoder with higher resolution [50]. Similarly, CogAgent [44] uses a dual-encoder mechanism, where two encoders process high and low-resolution images, respectively. High-resolution features are injected into the lowresolution branch through cross-attention. Patch-division methods cut a high-resolution image into patches and reuse the low-resolution encoder. For example, Monkey [51] and SPHINX [53] divide a large image into smaller patches and send sub-images together with a downsampled highresolution image to the image encoder, where the subimages and the low-resolution image capture local and global features, respectively. In contrast, parameter size and training data composition are of less importance compared with input resolution, found by empirical studies [52]. 

Similar encoders are also available for other modalities. For example, Pengi [27] uses CLAP [54] model as the audio encoder. ImageBind-LLM [30] uses the ImageBind [55] encoder, which supports encoding image, text, audio, depth, thermal, and Inertial Measurement Unit (IMU) data. Equipped with the strong encoder, ImageBind-LLM can respond to the input of multiple modalities.

在选择编码器时，人们通常会考虑分辨率、参数规模以及预训练语料库等因素。值得注意的是，许多研究已经**通过实验验证，使用更高的分辨率能够显著提升模型性能**[34, 50, 51, 52]。**提升输入分辨率的方法可分为直接缩放法和分块法。**直接缩放法是将更高分辨率的图像输入到编码器中，这通常需要进一步微调编码器[34]，或者用更高分辨率的预训练编码器来替换原有的编码器[50]。类似地，CogAgent[44]采用了双编码器机制，由两个编码器分别处理高分辨率和低分辨率的图像。高分辨率特征通过交叉注意力机制注入到低分辨率分支中。**分块法则是将高分辨率图像切割成小块，然后复用低分辨率编码器。**例如，Monkey[51]和SPHINX[53]将大图像分割成较小的图块，并将子图像与下采样(**（subsampled）：又名降采样、缩小图像；比如池化**)的高分辨率图像一起发送到图像编码器中，其中子图像和低分辨率图像分别捕获局部和全局特征。相比之下，根据实证研究[52]发现，**参数规模和训练数据组成**与输入分辨率相比，重要性较低。

> 提高图片的分辨率:
> 例如，将一张原本是 1280×720 分辨率的图片提高到 1920×1080 分辨率，意味着在水平方向上像素点从 1280 个增加到 1920 个，垂直方向上从 720 个增加到 1080 个，整体像素数量大幅增加，从而能够呈现更多的细节。

 对于其他模态也有类似的编码器。例如，Pengi[27]使用CLAP[54]模型作为音频编码器。ImageBind-LLM[30]使用ImageBind[55]编码器，该编码器支持对图像、文本、音频、深度、热成像以及惯性测量单元（IMU）数据进行编码。借助强大的编码器，ImageBind-LLM能够对多种模态的输入做出响应。 

### 2.2 Pre-trained LLM

2.2 预训练大语言模型

Instead of training an LLM from scratch, it is more efficient and practical to start with a pre-trained one. Through tremendous pre-training on web corpus, LLMs have been embedded with rich world knowledge, and demonstrate strong generalization and reasoning capabilities.

**与其从头开始训练一个大语言模型（LLM），从一个预训练的大语言模型入手会更加高效且实用**。通过在网络语料库上进行大量的预训练，大语言模型已经嵌入了丰富的世界知识，并展现出强大的泛化和推理能力。 

We summarize the commonly used and publicly available LLMs in Table 2. Notably, most LLMs fall in the causal decoder category, following GPT-3 [7]. Among them, FlanT5 [56] series are relatively early LLMs used in works like BLIP-2 [59] and InstructBLIP [60]. LLaMA series [5], [57] and Vicuna family [4] are representative open-sourced LLMs that have attracted much academic attention. Since the two LLMs are predominantly pre-trained on English corpus, they are limited in multi-language support, such as Chinese. In contrast, Qwen [58] is a bilingual LLM that supports Chinese and English well.

我们在表2中总结了常用且可公开获取的大语言模型。值得注意的是，大多数大语言模型都属于因果解码器类别，沿袭了GPT-3[7]的架构。其中，FlanT5[56]系列是在BLIP-2[59]和InstructBLIP[60]等研究中相对较早使用的大语言模型。**LLaMA系列[5, 57]和Vicuna系列[4]是具有代表性的开源大语言模型，受到了学术界的广泛关注。**由于这两个大语言模型主要在英语语料库上进行预训练，它们在对多种语言（如中文）的支持方面存在局限。**相比之下，Qwen[58]是一个能很好地支持中文和英文的双语大语言模型。** 

> 关于 prefix-decoder、casual-decoder、encoder-decoder
>
> [大模型面经——从prefix-decoder、casual-decoder、encoder-decoder角度深入聊聊大模型 - 知乎](https://zhuanlan.zhihu.com/p/694953500)

TABLE 2: A summary of commonly used open-sourced LLMs. en, zh, fr, and de stand for English, Chinese, French, and German, respectively.

表2：常用开源大语言模型概述。“en”“zh”“fr”和“de”分别代表英语、中文、法语和德语。 

![img](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250320153044290.png)

It should be noted that scaling up the parameter size of LLMs also brings additional gains, similar to the case of increasing input resolution. Specifically, Liu et al. [50], [61] find that simply scaling up LLM from 7B to 13B brings comprehensive improvement on various benchmarks. Furthermore, when using a 34B LLM, the model shows emergent zero-shot Chinese capability, given that only English multimodal data are used during training. Lu et al. [62] see a similar phenomenon by scaling up LLMs from 13B to 35B and 65B/70B, where the larger model size brings consistent gains on benchmarks specifically designed for MLLMs. There are also works that use smaller LLMs to facilitate deployment on mobile devices. For example, MobileVLM series [63], [64] use downscaled LLaMA [5] (termed as MobileLLaMA 1.4B/2.7B), enabling efficient inference on mobile processors.

需要注意的是，**扩大大语言模型（LLMs）的参数规模也会带来额外的收益，这与提高输入分辨率的情况类似。**具体来说，刘等人[50, 61]发现，简单地将大语言模型的参数规模从70亿扩大到130亿，能在各种基准测试中带来全面的提升。此外，当使用一个拥有340亿参数的大语言模型时，尽管在训练过程中只使用了英语多模态数据，该模型仍展现出了新兴的零样本中文处理能力。陆等人[62]在将大语言模型的参数规模从130亿扩大到350亿以及650亿/700亿时，也观察到了类似的现象，即更大的模型规模在专门为多模态大语言模型（MLLMs）设计的基准测试中带来了持续的性能提升。也有一些研究使用参数规模较小的大语言模型，以便于在移动设备上部署。例如，MobileVLM系列[63, 64]使用了缩小规模的LLaMA模型[5]（称为MobileLLaMA 14亿/27亿参数版本），使得在移动处理器上能够进行高效的推理。 

Recently, explorations of Mixture of Experts (MoE) architecture for LLMs have garnered rising attention [65], [66], [67]. Compared with dense models, the sparse architecture enables scaling up total parameter size without increasing computational cost, by selective activation of the parameters. Empirically, MM1 [52] and MoE-LLaVA [68] find that MoE implementation achieves better performance than the dense counterpart on almost all the benchmarks.

最近，**针对大语言模型（LLMs）的专家混合（MoE）架构的探索已获得越来越多的关注[65, 66, 67]。**与稠密模型(dense models)相比，稀疏架构通过对参数的选择性激活，能够在不增加计算成本的情况下扩大总参数规模。根据实证研究，MM1[52]和MoE-LLaVA[68]发现，**在几乎所有的基准测试中，基于专家混合（MoE）的实现方式都比稠密模型取得了更好的性能。** 

### 2.3 Modality interface

模态接口

Since LLMs can only perceive text, bridging the gap between natural language and other modalities is necessary. However, it would be costly to train a large multimodal model in an end-to-end manner. A more practical way is to introduce a learnable connector between the pre-trained visual encoder and LLM. The other approach is to translate images into languages with the help of expert models, and then send the language to LLM.

由于大语言模型（LLMs）只能感知文本，因此弥合自然语言与其他模态之间的差距是有必要的。然而，以端到端的方式训练一个大型多模态模型成本会很高。**一种更实际的方法是在预训练的视觉编码器和大语言模型之间引入一个可学习的连接器。另一种方法是在专家模型的帮助下将图像转译为语言，然后将这些语言信息发送给大语言模型。** 

**Learnable Connector.** It is responsible for bridging the gap between different modalities. Specifically, the module projects information into the space that LLM can understand efficiently. Based on how multimodal information is fused, there are broadly two ways to implement such interfaces, i.e. token-level and feature-level fusion.

**可学习连接器。它负责弥合不同模态之间的差距。具体而言，该模块将信息投射到能让大语言模型（LLM）高效理解的空间中。基于多模态信息的融合方式，实现此类接口大致有两种途径，即标记级融合和特征级融合。 **

For token-level fusion, features output from encoders are transformed into tokens and concatenated with text tokens before being sent into LLMs. A common and feasible solution is to leverage a group of learnable query tokens to extract information in a query-based manner [69], which first has been implemented in BLIP-2 [59], and subsequently inherited by a variety of work [26], [60], [70]. Such Q-Formerstyle approaches compress visual tokens into a smaller number of representation vectors. In contrast, some methods simply use a MLP-based interface to bridge the modality gap [20], [37], [71], [72]. For example, LLaVA series adopts one/two linear MLP [20], [50] to project visual tokens and align the feature dimension with word embeddings.

对于**标记级融合(token-level fusion)，编码器输出的特征会被转换为标记，并在发送到大语言模型（LLMs）之前与文本标记连接起来。****一种常见且可行的解决方案是利用一组可学习的查询标记，以基于查询的方式提取信息[69]，**这种方法最早在BLIP-2[59]中得以实现，随后被众多研究工作[26, 60, 70]所采用。这种类似Q-Former的方法将视觉标记压缩为数量较少的表示向量。相比之下，一些方法只是简单地使用基于多层感知机（MLP）的接口来弥合模态差距[20, 37, 71, 72]。例如，LLaVA系列采用一个/两个线性多层感知机[20, 50]来投射视觉标记，并使特征维度与词嵌入相匹配。 

On a related note, MM1 [52] has ablated on design choices on the connector and found that for token-level fusion, the type of modality adapter is far less important than the number of visual tokens and input resolution. Nevertheless, Zeng et al. [73] compare the performance of token and feature-level fusion, and empirically reveal that the token-level fusion variant performs better in terms of VQA benchmarks. Regarding the performance gap, the authors suggest that cross-attention models might require a more complicated hyper-parameter searching process to achieve comparable performance.

与此相关的是，MM1[52]对连接器的设计选择进行了消融实验，结果发现，**对于标记级融合而言，模态适配器的类型远没有视觉标记的数量和输入分辨率重要。**尽管如此，曾等人[73]比较了标记级融合和特征级融合的性能，并通过实验揭示，在视觉问答（VQA）基准测试方面，标记级融合的变体表现更优。关于这种性能差距，作者认为，交叉注意力模型可能需要一个更为复杂的超参数搜索过程，才能达到与之相当的性能水平。 

As another line, feature-level fusion inserts extra modules that enable deep interaction and fusion between text features and visual features. For example, Flamingo [74] inserts extra cross-attention layers between frozen Transformer layers of LLMs, thereby augmenting language features with external visual cues. Similarly, CogVLM [75] plugs in a visual expert module in each Transformer layer to enable dual interaction and fusion between vision and language features. For better performance, the QKV weight matrix of the introduced module is initialized from the pre-trained LLM. Similarly, LLaMA-Adapter [76] introduces learnable prompts into Transformer layers. These prompts are first embedded with visual knowledge and then concatenated with text features as prefixes.

作为**另一种思路，特征级融合会插入额外的模块，以实现文本特征与视觉特征之间的深度交互与融合**。例如，Flamingo[74]在大语言模型（LLMs）已冻结的Transformer层之间**插入了额外的交叉注意力层**，从而利用外部视觉线索增强语言特征。类似地，CogVLM[75]在每个Transformer层中都插入了**一个视觉专家模块**，以实现视觉和语言特征之间的双向交互与融合。**为了获得更好的性能，所引入模块的查询-键-值（QKV）权重矩阵是由预训练的大语言模型初始化的。**同样，LLaMA-Adapter[76]在Transformer层中引入了可学习的提示。这些提示首先嵌入视觉知识，然后作为前缀与文本特征连接起来。 

In terms of parameter size, learnable interfaces generally comprise a small portion compared with encoders and LLMs. Take Qwen-VL [34] as an example, the parameter size of the Q-Former is about 0.08B, accounting for less than 1% of the whole parameters, while the encoder and the LLM account for about 19.8% (1.9B) and 80.2% (7.7B), respectively.

**在参数规模方面，与编码器和大语言模型相比，可学习接口通常只占一小部分。**以Qwen-VL[34]为例，Q-Former的参数规模约为0.8亿，占总参数的比例不到1%，而编码器和大语言模型的参数分别约占19.8%（19亿）和80.2%（77亿）。 

**Expert Model.** Apart from the learnable interface, using expert models, such as an image captioning model, is also a feasible way to bridge the modality gap [77], [78], [79], [80]. The basic idea is to convert multimodal inputs into languages without training. In this way, LLMs can understand multimodality by the converted languages. For example, VideoChat-Text [25] uses pre-trained vision models to extract visual information such as actions and enriches the descriptions using a speech recognition model. Though using expert models is straightforward, it may not be as flexible as adopting a learnable interface. The conversion of foreign modalities into text would cause information loss. For example, transforming videos into textual descriptions distorts spatial-temporal relationships [25].

**专家模型。**除了可学习接口之外，使用诸如图像描述生成模型之类的专家模型，也是弥合模态差距的一种可行方法[77, 78, 79, 80]。其**基本思路是在无需训练的情况下，将多模态输入转换为语言。通过这种方式，大语言模型能够借助转换后的语言来理解多模态信息。**例如，VideoChat-Text[25]使用预训练的视觉模型来提取诸如动作之类的视觉信息，并利用语音识别模型来丰富描述内容。尽管使用专家模型的方法简单直接，但它可能不如采用可学习接口那样灵活。将其他模态的信息转换为文本可能会导致信息丢失。例如，将视频转换为文本描述会扭曲时空关系[25]。 

## 3 TRAINING STRATEGY AND DATA

3 训练策略与数据

A full-fledged MLLM undergoes three stages of training, i.e. pre-training, instruction-tuning, and alignment tuning. Each phase of training requires different types of data and fulfills different objectives. In this section, we discuss training objectives, as well as data collection and characteristics for each training stage.

一个成熟的多模态大语言模型（MLLM）要经历三个训练阶段，即**预训练、指令微调以及对齐微调。**训练的每个阶段都需要不同类型的数据，并实现不同的目标。在本节中，**我们将讨论每个训练阶段的训练目标，以及数据收集方式和数据特点。** 

### 3.1 Pre-training

3.1 预训练

#### 3.1.1 Training Detail

3.1.1 训练细节

As the first training stage, pre-training mainly aims to align different modalities and learn multimodal world knowledge. Pre-training stage generally entails large-scale textpaired data, e.g. caption data. Typically, the caption pairs describe images/audio/videos in natural language sentences.

作为第一个训练阶段，**预训练主要旨在对齐不同的模态并学习多模态的世界知识。**预训练阶段通常需要大规模的文本配对数据，例如图像说明数据。**一般来说，这些图像说明配对是以自然语言句子来描述图像、音频或视频的。** 

Here, we consider a common scenario where MLLMs are trained to align vision with text. As illustrated in Table 3, given an image, the model is trained to predict autoregressively the caption of the image, following a standard cross-entropy loss. A common approach for pre-training is to keep pre-trained modules (e.g. visual encoders and LLMs) frozen and train a learnable interface [20], [35], [72]. The idea is to align different modalities without losing pre-trained knowledge. Some methods [34], [81], [82] also unfreeze more modules (e.g. visual encoder) to enable more trainable parameters for alignment. It should be noted that the training scheme is closely related to the data quality. For short and noisy caption data, a lower resolution (e.g. 224) can be adopted to speed up the training process, while for longer and cleaner data, it is better to utilize higher resolutions (e.g. 448 or higher) to mitigate hallucinations. Besides, ShareGPT4V [83] finds that with high-quality caption data in the pretraining stage, unlocking the vision encode promotes better alignment.

在这里，**我们考虑一种常见的场景，即训练多模态大语言模型（MLLMs）来实现视觉与文本的对齐。**如表3所示，**对于给定的一张图像，模型会按照标准的交叉熵损失，以自回归的方式来训练预测该图像的说明文字。**预训练的一种常见方法是==冻结预训练的模块（例如视觉编码器和大语言模型），并训练一个可学习的接口[20, 35, 72]。这样做的思路是在不丢失预训练知识的前提下对齐不同的模态。==一些方法[34, 81, 82]也会解冻更多的模块（例如视觉编码器），以便有更多可训练的参数来实现对齐。需要注意的是，==训练方案与数据质量密切相关==。对于简短且带有噪声的图像说明数据，可以采用较低的分辨率（例如224）来加快训练过程，而对于较长且更干净的数据，最好使用较高的分辨率（例如448或更高）来减少幻觉现象。此外，ShareGPT4V[83]发现，在预训练阶段使用高质量的图像说明数据时，解锁视觉编码器能促进更好的对齐效果。 

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250320170840451.png" alt="image-20250320170840396" style="zoom: 67%;" />

TABLE 3: A simplified template to structure the caption data. {<image>} is the placeholder for the visual tokens, and {caption} is the caption for the image. **Note that only the part marked in red is used for loss calculation.**

表3：构建图像说明数据的简化模板。{<图像>} 是视觉标记的占位符，{图像说明} 是该图像的说明文字。**请注意，只有用红色标注的部分用于损失计算。** 

#### 3.1.2 Data

Pretraining data mainly serve two purposes, i.e. (1) aligning different modalities and (2) providing world knowledge. The pretraining corpora can be divided into coarse-grained and fine-grained data according to granularities, which we will introduce sequentially. We summarize commonly used pretraining datasets in Table 4.

**预训练数据主要有两个作用，即：（1）对齐不同的模态；（2）提供世界知识。**根据粒度的不同，预训练语料库可以分为粗粒度数据和细粒度数据，我们将依次介绍。我们在表4中总结了常用的预训练数据集。 

Coarse-grained caption data share some typical traits in common: (1) The data volume is large since samples are generally sourced from the internet. (2) Because of the webscrawled nature, the captions are usually short and noisy since they originate from the alt-text of the web images. These data can be cleaned and filtered via automatic tools, for example, using CLIP [13] model to filter out imagetext pairs whose similarities are lower than a pre-defined threshold. In what follows, we introduce some representative coarse-grained datasets.

**粗粒度的图像**说明数据有一些共同的典型特征：（1）**数据量庞大**，因为样本通常来自互联网。（2）由于这些数据是从网页上抓取而来的，**图像说明文字通常简短且带有噪声**，因为它们源自网页图片的替代文本（alt文本）。这些数据可以通过自动工具进行清理和筛选，例如，==使用CLIP[13]模型来过滤掉图像与文本对之间相似度低于预定义阈值的样本。接下来，我们将介绍一些具有代表性的粗粒度数据集==。 

**CC.** CC-3M [84] is a web-scale caption dataset of 3.3M image-caption pairs, where the raw descriptions are derived from alt-text associated with images. The authors design a complicated pipeline to clean data: (1) For images, those with inappropriate content or aspect ratio are filtered. (2) For text, NLP tools are used to obtain text annotations, with samples filtered according to the designed heuristics. (3) For image-text pairs, images are assigned labels via classifiers. If text annotations do not overlap with image labels, the corresponding samples are dropped.

**CC。** ==CC-3M[84]是一个具有网页规模的图像说明数据集，包含330万组图像-说明文字对，其原始描述来源于与图像相关的替代文本==。作者设计了一套复杂的数据清理流程：（1）对于图像，会过滤掉内容不当或宽高比不合适的图像。（2）对于文本，使用自然语言处理工具获取文本注释，并根据设计的启发式方法对样本进行筛选。（3）对于图像-文本对，通过分类器为图像分配标签。如果文本注释与图像标签不匹配，相应的样本就会被剔除。 

CC-12M [85] is a following work of CC-3M and contains 12.4M image-caption pairs. Compared with the previous work, CC-12M relaxes and simplifies the data-collection pipeline, thus collecting more data.

CC-12M[85]是CC-3M的后续研究成果，包含1240万组图像-说明文字对。与之前的研究相比，CC-12M放宽并简化了数据收集流程，从而收集到了更多的数据。 

**SBU Captions [86]**. It is a captioned photo dataset containing 1M image-text pairs, with images and descriptions sourced from Flickr. Specifically, an initial set of images is acquired by querying the Flickr website with a large number of query terms. The descriptions attached to the images thus serve as captions. Then, to ensure that descriptions are relevant to the images, the retained images fulfill these requirements: (1) Descriptions of the images are of satisfactory length, decided by observation. (2) Descriptions of the images contain at least 2 words in the predefined term lists and a propositional word (e.g. “on”, “under”) that generally suggests spatial relationships.

**SBU 图像说明数据集**[86]。==这是一个带说明文字的照片数据集，包含100万组图像-文本对，图像和描述均来自Flickr网站==。具体来说，通过使用大量的查询词在Flickr网站上进行查询来获取初始的图像集合。因此，附加在这些图像上的描述就充当了图像说明文字。然后，为了确保描述与图像相关，保留下来的图像需满足以下要求：（1）通过观察确定，图像的描述具有合适的长度。（2）图像的描述中至少包含预定义术语列表中的两个单词，以及一个通常表示空间关系的介词（例如“在……上面（on）”、“在……下面（under）”） 。 

TABLE 4: Common datasets used for pre-training.

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250320172522707.png" alt="image-20250320172522665" style="zoom:67%;" />

>  Coarse-grained Image-Text（粗粒度图像 - 文本数据集）
>
> Fine-grained Image-Text（细粒度图像 - 文本数据集）
>
> Video-Text（视频 - 文本数据集）
>
> Audio-Text（音频 - 文本数据集）

**LAION.** This series are large web-scale datasets, with images scrawled from the internet and associated alt-text as captions. To filter the image-text pairs, the following steps are performed: (1) Text with short lengths or images with too small or too big sizes are dropped. (2) Image deduplication based on URL. (3) Extract CLIP [13] embeddings for images and text, and use the embeddings to drop possibly illegal content and image-text pairs with low cosine similarity between embeddings. Here we offer a brief summary of some typical variants:

**LAION。**这一系列是大规模的网页级数据集，其中的图像是从互联网上抓取的，相关的替代文本被用作图像说明文字。为了筛选图像-文本对，会执行以下步骤：（1）剔除文本长度过短的情况，以及尺寸过小或过大的图像。（2）基于统一资源定位符（URL）对图像进行去重。（3）提取图像和文本的CLIP[13]嵌入向量，并利用这些嵌入向量剔除可能存在的非法内容，以及嵌入向量之间余弦相似度较低的图像-文本对。下面我们简要总结一些典型的变体数据集： 

**LAION-5B [87]:** It is a research-purpose dataset of 5.85B image-text pairs. The dataset is multilingual with a 2B English subset.

LAION-5B[87]：这是一个用于研究目的的数据集，包含58.5亿组图像-文本对。该数据集是多语言的，其中有一个包含20亿组数据的英语子集。 

 **LAION-COCO [88]:** It contains 600M images extracted from the English subset of LAION-5B. The captions are synthetic, using BLIP [89] to generate various image cap- tions and using CLIP [13] to pick the best fit for the image. 

**LAION-COCO[88]：**它包含从LAION-5B的英语子集中提取的6亿张图像。这些图像说明文字是合成的，使用BLIP[89]生成各种图像说明，并使用CLIP[13]挑选出与图像最匹配的说明。

**COYO-700M [90].** It contains 747M image-text pairs, which are extracted from CommonCrawl. For data filtering, the authors design the following strategies: (1) For images, those with inappropriate size, content, format, or aspect ratio are filtered. Moreover, the images are filtered based on the pHash value to remove images overlapped with public datasets such as ImageNet and MS-COCO. (2) For text, only English text with satisfactory length, noun forms, and appropriate words are saved. Whitespace before and after the sentence will be removed, and consecutive whites- pace characters will be replaced with a single whitespace. Moreover, text appearing more than 10 times (e.g. “image for”) will be dropped. (3) For image-text pairs, duplicated samples are removed based on (image pHash, text) tuple

> **Common Crawl** 是一个[非营利](https://en.wikipedia.org/wiki/Nonprofit_organization)性 [501（c）（3）](https://en.wikipedia.org/wiki/501(c)_organization#501.28c.29.283.29) 组织，它[爬取](https://en.wikipedia.org/wiki/Web_crawler) Web 并免费向公众提供其档案和数据集。[[1$](https://en.wikipedia.org/wiki/Common_Crawl#cite_note-latimes-1)[[2$](https://en.wikipedia.org/wiki/Common_Crawl#cite_note-pressheretv-2) Common Crawl 的 [Web 档案](https://en.wikipedia.org/wiki/Web_archiving)包含自 2008 年以来收集的 [PB](https://en.wikipedia.org/wiki/Petabyte) 级数据。[[3$](https://en.wikipedia.org/wiki/Common_Crawl#cite_note-ready-3)它大约每月完成一次爬网。[[4$](https://en.wikipedia.org/wiki/Common_Crawl#cite_note-theverge-4)

 **COYO-700M[90]。**它包含7.47亿组图像-文本对，这些数据是从CommonCrawl中提取的。对于数据筛选，作者设计了以下策略：（1）对于图像，过滤掉尺寸、内容、格式或宽高比不合适的图像。此外，还会根据感知==哈希（pHash）值对图像进行筛选，以去除与ImageNet和MS-COCO等公开数据集重叠的图像==。（2）对于文本，只保留长度合适、为名词形式且用词恰当的英语文本。句子前后的空白字符将被删除，连续的空白字符将被替换为单个空白字符。此外，出现次数超过10次的文本（例如“用于……的图像”）将被剔除。（3）对于图像-文本对，根据（图像感知哈希值，文本）元组去除重复的样本。 

Recently, more works [83], [91], [92] have explored generating high-quality fine-grained data through prompting strong MLLMs (e.g. GPT-4V). Compared with coarsegrained data, these data generally contain longer and more accurate descriptions of the images, thus enabling finergrained alignment between image and text modalities. However, since this approach generally requires calling commercial-use MLLMs, the cost is higher, and the data volume is relatively smaller. Notably, ShareGPT4V [83] strikes a balance by first training a captioner with GPT-4V-generated 100K data, then scaling up the data volume to 1.2M using the pre-trained captioner.

最近，更多的研究工作[83, 91, 92]探索了通过**提示强大的多模态大语言模型（如GPT-4V）来生成高质量的细粒度数据。**与粗粒度数据相比，这些数据通常包含对图像更长且更准确的描述，从而能够在图像和文本模态之间实现更精细的对齐。然而，由于这种方法通常需要调用商用的多模态大语言模型，成本较高，并且数据量相对较小。值得注意的是，**ShareGPT4V[83]**通过以下方式取得了平衡：==首先使用GPT-4V生成的10万条数据训练一个图像描述生成器，然后利用预训练的图像描述生成器将数据量扩展到120万条==。 



![image-20250321174449266](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250321174449365.png)

Fig. 3: Comparison of three typical learning paradigms. The image is from [19].

图3：三种典型学习范式的比较。该图片来自文献[19]。 

### 3.2 Instruction-tuning

#### 3.2.1 Introduction

Instruction refers to the description of tasks. Intuitively, instruction tuning aims to teach models to better understand the instructions from users and fulfill the demanded tasks. Tuning in this way, LLMs can generalize to unseen tasks by following new instructions, thus boosting zero-shot performance. This simple yet effective idea has sparked the success of subsequent NLP works, such as ChatGPT [2], InstructGPT [95], FLAN [19], [56], and OPT-IML [96].

**指令是指对任务的描述。**直观来讲，==指令微调旨在教会模型更好地理解来自用户的指令，并完成所要求的任务。通过这种方式进行微调，大语言模型（LLMs）能够依据新的指令对未见过的任务进行泛化处理，从而提升零样本学习的性能==。这个简单却有效的理念引发了后续一系列自然语言处理工作的成功，比如ChatGPT[2]、InstructGPT[95]、FLAN[19, 56] 以及OPT-IML[96]。 

The comparisons between instruction tuning and related typical learning paradigms are illustrated in Fig. 3. The supervised fine-tuning approach usually requires a large amount of task-specific data to train a task-specific model. The prompting approach reduces the reliance on large-scale data and can fulfill a specialized task via prompt engineering. In such a case, though the few-shot performance has been improved, the zero-shot performance is still quite average [7]. Differently, instruction tuning learns how to generalize to unseen tasks rather than fitting specific tasks like the two counterparts. Moreover, instruction tuning is highly related to multi-task prompting [97].

指令微调与相关典型学习范式之间的比较如图3所示。**监督微调方法通常需要大量特定于任务的数据来训练一个特定任务的模型。**==提示方法减少了对大规模数据的依赖，并且可以通过提示工程来完成特定任务。在这种情况下，尽管少样本学习性能有所提高，但零样本学习性能仍然相当一般[7]==。**不同的是，指令微调学习的是如何对未见过的任务进行泛化，而不像上述两种方法那样去拟合特定任务。此外，指令微调与多任务提示密切相关[97]。** 

In this section, we delineate the format of instruction samples, the training objectives, typical ways to gather instruction data, and corresponding commonly used datasets.

在本节中，我们将阐述指令样本的格式、训练目标、收集指令数据的典型方法，以及相应的常用数据集。 

#### 3.2.2 Training Detail

3.2.2 训练细节

A multimodal instruction sample often includes an optional instruction and an input-output pair. The instruction is typically a natural language sentence describing the task, such as, “Describe the image in detail.” The input can be an image-text pair like the VQA task [99] or only an image like the image caption task [100]. The output is the answer to the instruction conditioned on the input. The instruction template is flexible and subject to manual designs [20], [25], [98], as exemplified in Table 5. Note that the instruction template can also be generalized to the case of multi-round conversations [20], [37], [71], [98].

**一个多模态指令样本通常包含一个可选的指令以及一组输入-输出对。**该指令一般是一个描述任务的自然语言句子，比如“详细描述一下这张图片”。输入可以是像视觉问答（VQA）任务[99]中的图像-文本对，也可以是像图像描述任务[100]中那样仅仅是一张图像。输出则是基于输入对指令做出的回答。指令模板是灵活可变的，并且可由人工设计[20, 25, 98]，具体示例如表5所示。需要注意的是，指令模板也能够推广应用到多轮对话的情形中[20, 37, 71, 98]。 

```sum
指令：一个描述任务的自然语言句子

输入：图像-文本对/图像

输出：基于输入对指令做出的回答
```

Formally, a multimodal instruction sample can be denoted in a triplet form, i.e. $(I, M, R)$ , where I 1 M , R represent the instruction, the multimodal input, and the ground truth response, respectively. The MLLM predicts an answer given the instruction and the multimodal input:  $\mathcal{A}=f(\mathcal{I}, \mathcal{M} ; \theta) (1)$

形式上，**一个多模态指令样本可以用三元组形式表示，即$(I, M, R)$ ，其中$I$、$M$、$R$分别表示指令、多模态输入以及真实答案。**多模态大语言模型（MLLM）根据给定的指令和多模态输入预测一个答案：

Here, A denotes the predicted answer, and θ are the parameters of the model. The training objective is typically the original auto-regressive objective used to train LLMs [20], [37], [71], [101], based on which the MLLM is encouraged to predict the next token of the response. The objective can be expressed as:  $\mathcal{L}(\theta)=-\sum_{i=1}^{N} log p\left(\mathcal{R}_{i} | \mathcal{I}, \mathcal{R}_{<i} ; \theta\right)$ where N is the length of the ground-truth response.

在这里，$$\mathcal{A}$$表示预测答案，而$$\theta$$是模型的参数。训练目标通常是用于训练大语言模型（LLMs）的原始自回归目标[20, 37, 71, 101]，基于此目标，多模态大语言模型（MLLM）被促使去预测答案的下一个标记。该目标可以表示为： $\mathcal{L}(\theta)=-\sum_{i = 1}^{N} \log p\left(\mathcal{R}_{i} | \mathcal{I}, \mathcal{R}_{<i} ; \theta\right)$ 其中$N$是真实答案的长度。 

> 在多模态大语言模型（MLLMs）的上下文中，形式化地表示一个多模态指令样本为三元组 $(I, M, R)$ 是非常直观的。这里，$I$ 代表指令或任务描述，$M$ 代表多模态输入（例如图像、音频、视频等），而 $R$ 则代表针对给定指令和输入的真实答案或预期响应。
>
> ### 预测过程
>
> 给定一个三元组 $(I, M, R)$，MLLM 的目标是基于指令 $I$ 和多模态输入 $M$ 来预测一个答案 $\mathcal{A}$。这个过程涉及到模型参数 $\theta$ 的使用，其中 $\theta$ 经过训练以优化模型对给定输入生成正确输出的能力。
>
> ### 训练目标
>
> 训练 MLLM 的核心在于最小化损失函数 $\mathcal{L}(\theta)$，该函数定义了模型预测的答案与真实答案之间的差距。对于自回归模型来说，这一目标通常被表述为：
>
> $$\mathcal{L}(\theta)=-\sum_{i=1}^{N} \log p\left(\mathcal{R}_{i} | \mathcal{I}, \mathcal{R}_{<i} ; \theta\right)$$
>
> 这里，
>
> - $N$ 表示真实答案的长度。
> - $\mathcal{R}_i$ 表示真实答案中的第 $i$ 个标记。
> - $\mathcal{R}_{<i}$ 表示截至到第 $i$ 个标记之前的所有标记。
> - $p\left(\mathcal{R}_{i} | \mathcal{I}, \mathcal{R}_{<i} ; \theta\right)$ 表示在给定指令 $\mathcal{I}$ 和截至当前时刻前的所有答案标记 $\mathcal{R}_{<i}$ 的条件下，模型预测下一个标记 $\mathcal{R}_i$ 的概率。
>
> 通过最小化上述损失函数，MLLM 学习如何根据给定的指令和多模态输入更准确地预测出完整且正确的答案。这种训练方式鼓励模型逐步改进其对复杂任务的理解和执行能力，特别是在处理需要理解和整合多种信息来源的任务时。
>

#### 3.2.3 Data Collection

Since instruction data are more flexible in formats and varied in task formulations, it is usually trickier and more costly to collect data samples. In this section, we summarize three typical ways to harvest instruction data at scale, i.e. data adaptation, self-instruction, and data mixture. 

**Data Adaptation.** Task-specific datasets are rich sources of high-quality data. Hence, abundant works [60], [70], [76], [82], [101], [102], [103], [104] have utilized existing highquality datasets to construct instruction-formatted datasets. Take the transformation of VQA datasets for an example, the original sample is an input-out pair where the input comprises an image and a natural language question, and the output is the textual answer to the question conditioned on the image. The input-output pairs of these datasets could naturally comprise the multimodal input and response of the instruction sample (see §3.2.2). The instructions, i.e. the descriptions of the tasks, can either derive from manual design or from semi-automatic generation aided by GPT. Specifically, some works [21], [35], [60], [70], [102], [105] hand-craft a pool of candidate instructions and sample one of them during training. We offer an example of instruction templates for the VQA datasets as shown in Table 6. The other works manually design some seed instructions and use these to prompt GPT to generate more [25], [82], [98].

由于指令数据在格式上更为灵活，且任务表述形式多样，所以收集数据样本通常更具挑战性，成本也更高。在本节中，我们总结了三种大规模获取指令数据的典型方法，即==数据适配、自指令生成和数据混合==。

 **数据适配**：特定任务的数据集是高质量数据的丰富来源。因此，大量的研究工作[60]、[70]、[76]、[82]、[101]、[102]、[103]、[104]都利用现有的高质量数据集来构建指令格式的数据集。**以视觉问答（VQA）数据集的转换为例，原始样本是一个输入-输出对，其中输入由一张图像和一个自然语言问题组成，输出则是基于该图像对问题的文本回答。**这些数据集中的输入-输出对自然可以构成指令样本中的多模态输入和回答（见3.2.2节）。**指令，也就是对任务的描述，可以来自人工设计，也可以借助GPT进行半自动生成。**具体来说，一些研究工作[21]、[35]、[60]、[70]、[102]、[105]手工设计了一组候选指令，并在训练过程中从中抽取一条。我们给出一个针对VQA数据集的指令模板示例，如表6所示。另外一些研究工作则人工设计一些种子指令，然后利用这些指令促使GPT生成更多指令[25]、[82]、[98]。 

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250324172318649.png" alt="image-20250324172318515" style="zoom:80%;" />

TABLE 5: A simplified template to structure the multimodal instruction data. <instruction> is a textual description of the task. {<image>, <text>} and <output> are input and output from the data sample. Note that <text> in the input may be missed for some datasets, such as image caption datasets merely have <image>. The example is adapted from [98].

表5：构建多模态指令数据的简化模板。<指令>是对任务的文本描述。{<图像>，<文本>}和<输出>分别是数据样本中的输入和输出。请注意，对于某些数据集，输入中的<文本>可能不存在，例如图像描述数据集可能仅包含<图像>。该示例改编自文献[98]。 

Note that since the answers of existing VQA and caption datasets are usually concise, directly using these datasets for instruction tuning may limit the output length of MLLMs. There are two common strategies to tackle this problem. The first one is to specify explicitly in instructions. For example, ChatBridge [104] explicitly declares short and brief for shortanswer data, as well as a sentence and single sentence for conventional coarse-grained caption data. The second one is to extend the length of existing answers [105]. For example, $$M^{3} IT$$ 105] proposes to rephrase the original answer by prompting ChatGPT with the original question, answer, and contextual information of the image (e.g. caption and OCR). 

**Self-Instruction.** Although existing multi-task datasets can contribute a rich source of data, they usually do not meet human needs well in real-world scenarios, such as multiple rounds of conversations. To tackle this issue, some works collect samples through self-instruction [106], which utilizes LLMs to generate textual instruction-following data using a few hand-annotated samples. Specifically, some instructionfollowing samples are hand-crafted as demonstrations, after which ChatGPT/GPT-4 is prompted to generate more instruction samples with the demonstrations as guidance. LLaVA [20] extends the approach to the multimodal field by translating images into text of captions and bounding boxes, and prompting text-only GPT-4 to generate new data with the guidance of requirements and demonstrations. In this way, a multimodal instruction dataset is constructed, called LLaVA-Instruct-150k. Following this idea, subsequent works such as MiniGPT-4 [21], ChatBridge [104], GPT4Tools [107], and DetGPT [72] develop different datasets catering for different needs. Recently, with the release of the more powerful multimodal model GPT4V, many works have adopted GPT-4V to generate data of higher quality, as exemplified by LVIS-Instruct4V [91] and ALLaVA [92]. We summarize the popular datasets generated through self-instruction in Table 7.

请注意，由于现有的视觉问答（VQA）和图像描述数据集的答案通常较为简洁，直接使用这些数据集进行指令微调**可能会限制多模态大语言模型（MLLMs）的输出长度**。有两种常见的策略来解决这个问题。==第一种是在指令中明确说明==。例如，ChatBridge[104]针对简短答案数据明确标注为“简短扼要（short and brief）”，对于传统的粗粒度图像描述数据则标注为“一个句子（a sentence）”和“单个句子（single sentence）” 。==第二种策略是扩展现有答案的长度[105]==。例如，$$M^{3} IT$$[105]提出通过使用原始问题、答案以及图像的上下文信息（如图像描述和光学字符识别（OCR）信息）来提示ChatGPT，从而对原始答案进行改写。 

**自指令生成**：尽管现有的多任务数据集可以提供丰富的数据来源，但它们在现实场景中==通常不能很好地满足人类的需求，比如多轮对话场景==。为了解决这个问题，一些研究工作通过自指令生成的方式来收集样本[106]，这种方法利用大语言模型（LLMs），基于少量人工标注的样本生成遵循文本指令的数据。具体来说，==先手工制作一些遵循指令的样本作为示例，然后提示ChatGPT/GPT-4以这些示例为指导生成更多的指令样本==。**LLaVA[20]将这种方法扩展到了多模态领域，它将图像转换为包含图像描述和边界框信息的文本，然后提示仅处理文本的GPT-4在需求和示例的指导下生成新的数据**。通过这种方式，构建了一个名为LLaVA-Instruct-150k的多模态指令数据集。遵循这一思路，后续的研究工作如MiniGPT-4[21]、ChatBridge[104]、GPT4Tools[107]和DetGPT[72]开发了满足不同需求的不同数据集。最近，随着更强大的多模态模型GPT4V的发布，许多研究工作采用GPT-4V来生成更高质量的数据，如LVIS-Instruct4V[91]和ALLaVA[92]就是例证。我们在表7中总结了通过自指令生成的常用数据集。 

![image-20250324191413621](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250324191413702.png)

TABLE 6: Instruction templates for VQA datasets, cited from [60]. <Image> and {Question} are the image and the question in the original VQA datasets, respectively.

表6：视觉问答（VQA）数据集的指令模板，引用自[60]。<图像>和{问题}分别是原始视觉问答数据集中的图像和问题。 

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250324191459127.png" alt="image-20250324191459065" style="zoom:80%;" />

TABLE 7: A summary of popular datasets generated by **self-instruction**. For input/output modalities, I: Image, T: Text, V: Video, A: Audio. For data composition, M-T and S-T denote multi-turn and single-turn, respectively.

表7：通过自指令生成的常用数据集概述。对于输入/输出模态，I代表图像，T代表文本，V代表视频，A代表音频。在数据构成方面，M-T和S-T分别表示多轮对话和单轮对话。 

---

**Data Mixture.** Apart from the multimodal instruction data, language-only user-assistant conversation data can also be used to improve conversational proficiencies and instruction-following abilities [81], [98], [101], [103]. LaVIN [101] directly constructs a minibatch by randomly sampling from both language-only and multimodal data. MultiInstruct [102] probes different strategies for training with a fusion of single modal and multimodal data, including mixed instruction tuning (combine both types of data and randomly shuffle) and sequential instruction tuning (text data followed by multimodal data).

**数据混合**：除了多模态指令数据之外，纯语言形式的用户-助手对话数据也可用于提升对话能力和遵循指令的能力[81]、[98]、[101]、[103]。**LaVIN[101]通过从纯语言数据和多模态数据中随机采样，直接构建一个小批次数据。**MultiInstruct[102]探索了融合单模态和多模态数据进行训练的不同策略，其中包括混合指令微调（将两种类型的数据合并并随机打乱）以及顺序指令微调（先使用文本数据，接着使用多模态数据） 。 

#### 3.2.4 Data Quality

Recent research has revealed that the data quality of instruction-tuning samples is no less important than quantity. Lynx [73] finds that models pre-trained on large-scale but noisy image-text pairs do not perform as well as models pre-trained with smaller but cleaner datasets. Similarly, Wei et al. [108] finds that less instruction-tuning data with higher quality can achieve better performance. For data filtering, the work proposes some metrics to evaluate data quality and, correspondingly, a method to automatically filter out inferior vision-language data. Here we discuss two important aspects regarding data quality. 

**Prompt Diversity.** The diversity of instructions has been found to be critical for model performance. Lynx [73] empirically verifies that diverse prompts help improve model performance and generalization ability.

**Task Coverage.** In terms of tasks involved in training data, Du et al. [109] perform an empirical study and find that the visual reasoning task is superior to captioning and QA tasks for boosting model performance. Moreover, the study suggests that enhancing the complexity of instructions might be more beneficial than increasing task diversity and incorporating fine-grained spatial annotations.

**最近的研究表明，指令微调样本的数据质量与数量同等重要**。Lynx[73]发现，在大规模但存在噪声的图像-文本对上进行预训练的模型，其表现不如在规模较小但数据更纯净的数据集上进行预训练的模型。同样，魏等人[108]发现，质量较高但指令微调数据量较少也能取得更好的性能。关于数据筛选，这项研究提出了一些评估数据质量的指标，以及相应的一种自动筛选出劣质视觉-语言数据的方法。在此，我们讨论与数据质量相关的两个重要方面。 

**提示多样性**：研究发现，指令的多样性对模型性能至关重要。Lynx[73]通过实证验证，==多样化的提示有助于提高模型的性能和泛化能力==。 

**任务覆盖范围**：就训练数据中涉及的任务而言，杜等人[109]进行了一项实证研究，发现视觉推理任务在提升模型性能方面优于图像描述任务和问答任务。此外，该研究表明，==增强指令的复杂性可能比增加任务多样性以及纳入细粒度的空间注释更为有益==。 

### 3.3 Alignment tuning

3.3 对齐微调

#### 3.3.1 Introduction

Alignment tuning is more often used in scenarios where models need to be aligned with specific human preferences, e.g. response with fewer hallucinations (see §6). Currently, Reinforcement Learning with Human Feedback (RLHF) and Direct Preference Optimization (DPO) are two main techniques for alignment tuning. In this section, we introduce the main ideas of the two techniques in sequence and offer some examples of how they are utilized in addressing practical problems, and finally, give a compilation of the related datasets.

对齐微调更多地应用于这样的场景：==模型需要与特定的人类偏好保持一致，例如产生更少幻觉的回复（详见第6节）==。目前，==基于人类反馈的强化学习（RLHF）和直接偏好优化（DPO）是用于对齐微调的两种主要技术==。在本节中，我们将依次介绍这两种技术的主要思想，并给出一些关于它们如何被用于解决实际问题的示例，最后，对相关的数据集进行整理汇总。 

#### 3.3.2 Training Detail

RLHF [110], [111]. This technique aims to utilize reinforcement learning algorithms to align LLMs with human preferences, with human annotations as supervision in the training loop. As exemplified in InstructGPT [95], RLHF incorporates three key steps: 

1 )  **Supervised fine-tuning.** This step aims to fine-tune a pre-trained model to present the preliminary desired output behavior. The fine-tuned model in the RLHF setting is called a policy model. Note that this step might be skipped since the supervised policy model $\pi^{SFT }$ can be initialized from an instruction-tuned model (see §3.2). 

2 ）**Reward modeling.** A reward model is trained using preference pairs in this step. Given a multimodal prompt ( $e.g$ . image and text) x and a response pair $(y_{w}, y_{l})$ , the reward model $r_{\theta}$ learns to give a higher reward to the preferred response $y_{w}$ , and vice versa for $y_{l}$ , according to the following objective:  $$\mathcal{L}(\theta)=-\mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}}\left[log \left(\sigma\left(r_{\theta}\left(x, y_{w}\right)-r_{\theta}\left(x, y_{l}\right)\right]\right.\right. (3)$$ 

where $D={(x, y_{w}, y_{l})}$ is the comparison dataset labeled by human annotators. In practice, the reward model $r_{\theta}$ shares a similar structure with the policy model. 

3 ) **Reinforcement learning**. In this step, the Proximal Policy Optimization (PPO) algorithm is adopted to optimize the RL policy model $\pi_{\phi}^{RL}$ . A per-token KL penalty is often added to the training objective to avoid deviating too far from the original policy [95], resulting in the objective:  $$\begin{aligned} \mathcal{L}(\phi) & =-\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\phi}^{R L}(y | x)}\left[r_{\theta}(x, y)\right. \\ & \left.-\beta \cdot \mathbb{D}_{K L}\left(\pi_{\phi}^{R L}(y | x) \| \pi^{R E F}(y | x)\right)\right] \end{aligned}$$

 where β is the coefficient for the KL penalty term. Typically, both the RL policy $\pi_{\phi}^{RL}$ and the reference model $\pi^{REF}$ are initialized from the supervised model $\pi^{SFT}$ The obtained RL policy model is expected to align with human preferences through this tuning process.

基于人类反馈的强化学习（RLHF）[110]、[111]。这项技术旨在利用强化学习算法，使大语言模型（LLMs）与人类偏好保持一致，在训练循环中以人类标注作为监督。以InstructGPT[95]为例，基于人类反馈的强化学习包含三个关键步骤： 

1 ) **监督微调**：这一步骤旨在对预训练模型进行微调，使其展现出初步期望的输出行为。==在基于人类反馈的强化学习设置中，经过微调的模型被称为策略模型。==请注意，这一步骤可能会被跳过，因为监督策略模型$\pi^{SFT }$可以从经过指令微调的模型进行初始化（见3.2节）。 

2 ) **奖励建模**：在这一步骤中，使用偏好对来训练一个奖励模型。给定一个多模态提示（例如，图像和文本）$x$以及一个回复对$(y_{w}, y_{l})$ ，奖励模型$r_{\theta}$根据以下目标学习为更受偏好的回复$y_{w}$赋予更高的奖励，而对于$y_{l}$则反之： $\mathcal{L}(\theta)=-\mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}}\left[\log \left(\sigma\left(r_{\theta}\left(x, y_{w}\right)-r_{\theta}\left(x, y_{l}\right)\right]\right.\right. (3)$ 其中$\mathcal{D}=\{(x, y_{w}, y_{l})\}$是由人类标注者标注的比较数据集。在实践中，奖励模型$r_{\theta}$与策略模型具有相似的结构。 

3 ) **强化学习**：在这一步骤中，采用近端策略优化（PPO）算法来优化强化学习策略模型$\pi_{\phi}^{RL}$ 。在训练目标中通常会添加每个标记的KL散度惩罚项，以避免与原始策略偏离太远[95]，从而得到如下目标函数：

 $\begin{aligned} \mathcal{L}(\phi) & =-\mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\phi}^{R L}(y | x)}\left[r_{\theta}(x, y)\right. \\ & \left.-\beta \cdot \mathbb{D}_{K L}\left(\pi_{\phi}^{R L}(y | x) \| \pi^{R E F}(y | x)\right)\right] \end{aligned}$ 

其中$\beta$是KL惩罚项的系数。通常，强化学习策略$\pi_{\phi}^{RL}$和参考模型$\pi^{REF}$都从监督模型$\pi^{SFT}$进行初始化。通过这个微调过程，得到的强化学习策略模型有望与人类偏好保持一致。 

Researchers have explored using the RLHF techniques for better multimodal alignment. For example, LLaVARLHF [112] collects human preference data and tunes a model with fewer hallucinations based on LLaVA [20]. DPO [113]. It learns from human preference labels utilizing a simple binary classification loss. Compared with the PPObased RLHF algorithm, DPO is exempt from learning an explicit reward model, thus simplifying the whole pipeline to two steps, i.e. human preference data collection and preference learning. The learning objective is as follows:  $\begin{aligned} \mathcal{L}(\phi) & =-\mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}}\left[ l o g \sigma \left(\beta log \frac{\pi_{\phi}^{RL}\left(y_{w} | x\right)}{\pi^{REF}\left(y_{w} | x\right)}\right.\right. \\ & \left.\left.-\beta log \frac{\pi_{\phi}^{RL}\left(y_{l} | x\right)}{\pi^{REF}\left(y_{l} | x\right)}\right)\right] \end{aligned}$ 

RLHF-V [114] collects fine-grained (segment-level) preference data pairs by correcting hallucinations in the model response and uses the obtained data to perform dense DPO. Silkie [115] instead collects preference data via prompting GPT-4V and distills the preference supervision into an instruction-tuned model through DPO.

研究人员已经探索了使用基于人类反馈的强化学习（RLHF）技术来实现更好的多模态对齐。例如，LLaVARLHF[112]收集人类偏好数据，并在LLaVA[20]的基础上对模型进行微调，使其产生更少的幻觉。

 **直接偏好优化（DPO）**[113]。它利用简单的二分类损失函数从人类偏好标签中学习。与基于近端策略优化（PPO）的基于人类反馈的强化学习（RLHF）算法相比，直接偏好优化无需学习一个明确的奖励模型，从而将整个流程简化为两个步骤，即人类偏好数据收集和偏好学习。其学习目标如下： $\begin{aligned} \mathcal{L}(\phi) & =-\mathbb{E}_{\left(x, y_{w}, y_{l}\right) \sim \mathcal{D}}\left[ \log \sigma \left(\beta \log \frac{\pi_{\phi}^{RL}\left(y_{w} | x\right)}{\pi^{REF}\left(y_{w} | x\right)}\right.\right. \\ & \left.\left.-\beta \log \frac{\pi_{\phi}^{RL}\left(y_{l} | x\right)}{\pi^{REF}\left(y_{l} | x\right)}\right)\right] \end{aligned}$

 RLHF-V[114]通过纠正模型回复中的幻觉来收集细粒度（片段级别）的偏好数据对，并使用得到的数据来执行密集的直接偏好优化。而Silkie[115]则是通过提示GPT-4V来收集偏好数据，并通过直接偏好优化将偏好监督信息提取到经过指令微调的模型中。 

#### 3.3.3 Data

The gist of data collection for alignment-tuning is to collect feedback for model responses, i.e. to decide which response is better. It is generally more expensive to collect such data, and the amount of data used for this phase is typically even less than that used in previous stages. In this part, we introduce some datasets and summarize them in Table 8. 

**LLaVA-RLHF [112].** It contains 10K preference pairs collected from human feedback in terms of honesty and helpfulness. The dataset mainly serves to reduce hallucinations in model responses. 

**RLHF-V [114]**. It has 5.7K fine-grained human feedback data collected by segment-level hallucination corrections.

 **VLFeedback [115].** It utilizes AI to provide feedback on model responses. The dataset contains more than 380K comparison pairs scored by GPT-4V in terms of helpfulness, faithfulness, and ethical concerns.

**对齐微调的数据收集要点在于收集对模型回复的反馈，也就是判断哪种回复更好。**一般来说，收集这类数据的成本更高，并且用于这个阶段的数据量通常甚至比之前阶段所使用的数据量还要少。在这一部分，我们介绍一些数据集并将它们总结在表8中。 

**LLaVA-RLHF**[112]：它包含从人类反馈中收集的10000个关于诚实性和有用性方面的偏好对。==该数据集主要用于减少模型回复中的幻觉现象==。

 **RLHF-V**[114]：它拥有5700条通过片段级幻觉纠正收集到的细粒度人类反馈数据。 **VLFeedback**[115]：它利用人工智能为模型回复提供反馈。该数据集包含超过38万个由GPT-4V根据有用性、忠实性和伦理问题进行评分的比较对。 

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250324194132375.png" alt="image-20250324194132294" style="zoom:67%;" />

TABLE 8: A summary of datasets for alignment-tuning. For input/output modalities, I: Image, T: Text.

表8：用于对齐微调的数据集概述。对于输入/输出模态，I代表图像，T代表文本。 

## 4 EVALUATION

Evaluation is an essential part of developing MLLMs since it provides feedback for model optimization and helps to compare the performance of different models. Compared with evaluation methods of traditional multimodal models, the evaluation of MLLMs exhibits several new traits: (1) Since MLLMs are generally versatile, it is important to evaluate MLLMs comprehensively. (2) MLLMs exhibit many emergent capabilities that require special attention (e.g. OCR-free math reasoning) and thus require new evaluation schemes. The evaluation of MLLMs can be broadly categorized into two types according to the question genres, including closed-set and open-set.

评估是开发多模态大语言模型（MLLMs）的重要组成部分，因为==它为模型优化提供反馈，并有助于比较不同模型的性能==。与传统多模态模型的评估方法相比，多模态大语言模型的评估呈现出几个新特点：（1）由于多模态大语言模型**通常具有多种功能，全面评估多模态大语言模型至关重要**。（2）多模态大语言模型展现出许多需要特别关注的新兴能力（例如无需光学字符识别（OCR）的数学推理能力），因此需要新的评估方案。根据问题类型，**多模态大语言模型的评估大致可分为两类，包括封闭集评估和开放集评估。** 

### 4.1 Closed-set

Closed-set questions refer to a type of question where the possible answer options are predefined and limited to a finite set. The evaluation is usually performed on taskspecific datasets. In this case, the responses can be naturally judged by benchmark metrics [20], [60], [70], [76], [101], [102], [103], [104]. For example, InstructBLIP [60] reports the accuracy on ScienceQA [116], as well as the CIDEr score [117] on NoCaps [118] and Flickr30K [119]. The evaluation settings are typically zero-shot [60], [102], [104], [105] or finetuning [20], [35], [60], [70], [76], [101], [103], [105]. The first setting often selects a wide range of datasets covering different general tasks and splits them into held-in and held-out datasets. After tuning on the former, zero-shot performance is evaluated on the latter with unseen datasets or even unseen tasks. In contrast, the second setting is often observed in the evaluation of domain-specific tasks. For example, LLaVA [20] and LLaMA-Adapter [76] report finetuned performance on ScienceQA [116]. LLaVA-Med [35] reports results on biomedical VQA [120], [121], [122].

**封闭集问题是指这样一类问题，其可能的答案选项是预先定义好的，并且限定在一个有限的集合内。评估通常在特定任务的数据集上进行。**在这种情况下，模型的回复可以自然地通过基准评估指标来判断[20]、[60]、[70]、[76]、[101]、[102]、[103]、[104]。例如，InstructBLIP[60]报告了在ScienceQA[116]数据集上的准确率，以及在NoCaps[118]和Flickr30K[119]数据集上的CIDEr分数[117]。评估设置通常为==零样本学习[60]、[102]、[104]、[105]或微调==[20]、[35]、[60]、[70]、[76]、[101]、[103]、[105]。第一种设置（零样本学习）通常会选择涵盖不同通用任务的广泛数据集，并将它们划分为训练集和测试集。在对训练集进行调整后，在测试集上对模型进行零样本性能评估，测试集可能是模型从未见过的数据集，甚至是从未见过的任务。相比之下，==第二种设置（微调）通常出现在特定领域任务的评估中。==例如，LLaVA[20]和LLaMA-Adapter[76]报告了在ScienceQA[116]数据集上微调后的性能。LLaVA-Med[35]报告了在生物医学视觉问答任务（涉及[120]、[121]、[122]相关内容）上的结果。 

> 怎么体现的零样本?
>
> 1. 测试集可能是模型从未见过的数据集，甚至是从未见过的任务”。也就是说，模型在训练阶段没有接触过测试集里的具体任务和数据，当用测试集去评估模型时，模型是首次面对这些新的内容，没有关于这些测试数据的先验训练经验，这符合零样本学习中模型对测试数据零接触的特点。
> 2. 在整个流程里，模型仅仅在训练集上进行调整，而这个调整不是针对测试集任务进行的特定训练。

 ```
  封闭集问题**
  
  其可能的答案选项是预先定义好的，并且限定在一个有限的集合内。评估通常在特定任务的数据集上进行
  
  - 零样本学习: 通常会选择涵盖不同通用任务的广泛数据集,划分训练集/测试集，在训练集上训练，测试集测试性能。
  - 微调：通常出现在特定领域任务的评估中。
 ```

  

The above evaluation methods are usually limited to a small range of selected tasks or datasets, lacking a comprehensive quantitative comparison. To this end, some efforts have endeavored to develop new benchmarks specially designed for MLLMs [123], [124], [125], [126], [127], [128], [129]. For example, Fu et al. [123] construct a comprehensive evaluation benchmark MME that includes a total of 14 perception and cognition tasks. All instruction-answer pairs in MME are manually designed to avoid data leakage. MMBench [124] is a benchmark specifically designed for evaluating multiple dimensions of model capabilities, using ChatGPT to match open responses with pre-defined choices. Video-ChatGPT [130] and Video-Bench [131] focus on video domains and propose specialized benchmarks as well as evaluation tools for assessment. There are also evaluation strategies designed to evaluate a specific aspect of the model [102], as exemplified by POPE [132] for assessment of hallucination degree.

**上述评估方法通常局限于一小部分选定的任务或数据集，缺乏全面的定量比较。**为此，一些研究人员努力开发了专门为多模态大语言模型（MLLMs）设计的新基准[123]、[124]、[125]、[126]、[127]、[128]、[129]。例如，傅等人[123]构建了一个全面的评估基准MME，该基准总共包含14项感知和认知任务。MME中的所有指令-答案对都是人工设计的，以避免数据泄露。MMBench[124]是一个专门为评估模型多维度能力而设计的基准，它使用ChatGPT将开放式回复与预定义的选项进行匹配。Video-ChatGPT[130]和Video-Bench[131]专注于视频领域，并提出了专门的基准以及评估工具用于评估。也有一些评估策略旨在评估模型的特定方面[102]，比如用于评估幻觉程度的POPE[132]就是一个例子。 

```bash
**专门为多模态大语言模型（MLLMs）设计的新基准**
  - MME 包含14项感知和认知任务
  - MMBench 评估模型多维度能力而设计的基准
  - POPE
  - Video-ChatGPT,Video-Bench 视频领域
  - POPE 评估幻觉程度的POPE
```

###   4.2 Open-set

In contrast to the closed-set questions, the responses to open-set questions can be more flexible, where MLLMs usually play a chatbot role. Because the content of the chat can be arbitrary, it would be trickier to judge than the closed-ended output. The criterion can be classified into manual scoring, GPT scoring, and case study. Manual scoring requires humans to assess the generated responses. This kind of approach often involves hand-crafted questions that are designed to assess specific dimensions. For example, mPLUG-Owl [81] collects a visually related evaluation set to judge capabilities like natural image understanding, diagram, and flowchart understanding. Similarly, GPT4Tools [107] builds two sets for the finetuning and zeroshot performance, respectively, and evaluates the responses in terms of thought, action, arguments, and the whole.

**与封闭集问题不同，对于开放集问题的回答可以更加灵活**，在这种情况下，多模态大语言模型（MLLMs）通常扮演聊天机器人的角色。==由于聊天的内容可以是任意的，因此相较于封闭式的输出，判断其回答会更具挑战性==。评估标准可以分为**人工评分、GPT评分和案例研究**。人工评分要求人类对生成的回答进行评估。这种方法通常涉及精心设计的问题，这些问题旨在评估特定的维度。例如，mPLUG-Owl[81]收集了一个与视觉相关的评估集，用于判断自然图像理解、图表以及流程图理解等方面的能力。同样地，GPT4Tools[107]分别构建了两个集合，用于评估微调性能和零样本性能，并从思路、行动、论据以及整体等方面对回答进行评估。 

```
开放集
==由于聊天的内容可以是任意的，因此相较于封闭式的输出，判断其回答会更具挑战性==
- 人工评分：要求人类对生成的回答进行评估
	- mPLUG-Owl 用于判断自然图像理解、图表以及流程图理解等方面的能力。
	- GPT4Tools 用于评估微调性能和零样本性能，并从思路、行动、论据以及整体等方面对回答进行评估。
- GPT评分：常被用于评估多模态对话方面的性能。
- 案例研究：GPT-4V进行了深入的定性分析，这些任务从诸如图像描述和物体计数等基础技能，到需要世界知识和推理能力的复杂任务，比如笑话理解以及作为具身智能体的室内导航
```

Since manual assessment is labor intensive, some researchers have explored rating with GPT, namely GPT scoring. This approach is often used to evaluate performance on multimodal dialogue. LLaVA [20] proposes to score the responses via text-only GPT-4 in terms of different aspects, such as helpfulness and accuracy. Specifically, 30 images are sampled from the COCO [133] validation set, each associated with a short question, a detailed question, and a complex reasoning question via self-instruction on GPT-4. The answers generated by both the model and GPT-4 are sent to GPT-4 for comparison. Subsequent works follow this idea and prompt ChatGPT [81] or GPT-4 [35], [70], [101], [104], [105] to rate results [35], [70], [81], [101], [104] or judge which one is better [103].

由于人工评估耗费人力，一些研究人员探索了使用GPT进行评分的方式，即==GPT评分。这种方法常被用于评估多模态对话方面的性能。==LLaVA[20]提出通过仅处理文本的GPT-4从不同方面，如有用性和准确性，对回复进行评分。具体来说，从COCO[133]验证集中采样30张图片，每张图片通过在GPT-4上进行自指令操作，关联一个简短问题、一个详细问题和一个复杂推理问题。由模型和GPT-4生成的答案都被发送给GPT-4进行比较。后续的研究工作遵循这一思路，并提示ChatGPT[81]或GPT-4[35]、[70]、[101]、[104]、[105]对结果进行评分[35]、[70]、[81]、[101]、[104]，或者判断哪一个更好[103]。 

A main issue of applying text-only GPT-4 as an evaluator is that the judge is only based on image-related text content, such as captions or bounding box coordinates, without accessing the image [35]. Thus, it may be questionable to set GPT-4 as the performance upper bound in this case. With the release of the vision interface of GPT, some works [77], [134] exploit a more advanced GPT-4V model to assess the performance of MLLMs. For example, Woodpecker [77] adopts GPT-4V to judge the response quality of model answers based on the image. The evaluation is expected to be more accurate than using text-only GPT-4 since GPT-4V has direct access to the image.

**将仅处理文本的GPT-4用作评估器的一个主要问题是，评判仅基于与图像相关的文本内容，例如图像描述或边界框坐标，而无法直接访问图像本身**[35]。因此，在这种情况下将GPT-4设定为性能上限可能是存在疑问的。随着GPT视觉接口的发布，一些研究[77]、[134]==利用更先进的GPT-4V模型来评估多模态大语言模型（MLLMs）的性能==。例如，Woodpecker[77]采用GPT-4V来根据图像判断模型答案的回复质量。由于GPT-4V能够直接访问图像，预计这种评估会比使用仅处理文本的GPT-4更为准确。 

A supplementary approach is to compare the different capabilities of MLLMs through case studies. For instance, some studies evaluate two typical advanced commercial-use models, GPT-4V and Gemini. Yang et al. [135] perform indepth qualitative analysis on GPT-4V by crafting a series of samples across various domains and tasks, spanning from preliminary skills, such as caption and object counting, to complex tasks that require world knowledge and reasoning, such as joke understanding and indoor navigation as an embodied agent. Wen et al. [136] make a more focused evaluation of GPT-4V by designing samples targeting automatic driving scenarios. Fu et al. [137] carry out a comprehensive evaluation on Gemini-Pro by comparing the model against GPT-4V. The results suggest that GPT-4V and Gemini exhibit comparable visual reasoning abilities in spite of different response styles.

一种补充的方法是**通过案例研究来比较多模态大语言模型（MLLMs）的不同能力。**例如，一些研究对两种典型的先进商用模型，即GPT-4V和Gemini进行了评估。杨等人[135]通过精心设计一系列涵盖不同领域和任务的样本，对GPT-4V进行了深入的定性分析，这些任务从诸如图像描述和物体计数等基础技能，到需要世界知识和推理能力的复杂任务，比如笑话理解以及作为具身智能体的室内导航。温等人[136]通过设计针对自动驾驶场景的样本，对GPT-4V进行了更有针对性的评估。傅等人[137]通过将Gemini-Pro模型与GPT-4V进行比较，对其展开了全面评估。结果表明，尽管GPT-4V和Gemini的回复风格不同，但它们表现出了相当的视觉推理能力。 

## 5 EXTENSIONS

\5. 扩展内容 

Recent studies have made significant strides in extending the capabilities of MLLMs, spanning from more potent foundational abilities to broader coverage of scenarios. We trace the principal development of MLLMs in this regard. 

**Granularity Support.** To facilitate better interaction between agents and users, researchers have developed MLLMs with finer support of granularities in terms of model inputs and outputs. On the input side, models that support finer control from user prompts are developed progressively, evolving from image to region [28], [138], [139] and even pixels [29], [140], [141]. Specifically, Shikra [28] supports region-level input and understanding. Users may interact with the assistant more flexibly by referring to specific regions, which are represented in bounding boxes of natural language forms. Ferret [141] takes a step further and supports more flexible referring by devising a hybrid representation scheme. The model supports different forms of prompts, including point, box, and sketch. Similarly, Osprey [29] supports point input by utilizing a segmentation model [9]. Aided by the exceptional capabilities of the pre-trained segmentation model, Osprey enables specifying a single entity or part of it with a single click. On the output side, grounding capabilities are improved in line with the development of input support. Shikra [28] supports response grounded in the image with box annotations, resulting in higher precision and finer referring experience. LISA [142] further supports masklevel understanding and reasoning, which makes pixel-level grounding possible.

最近的研究在扩展多模态大语言模型（MLLMs）的能力方面取得了重大进展，涵盖了从更强大的基础能力到更广泛的场景覆盖范围。我们在此追溯多模态大语言模型在这方面的主要发展历程。 

**粒度支持**：为了**促进智能体与用户之间更好的交互**，研究人员开发了在模型输入和输出方面对粒度提供更精细支持的多模态大语言模型。==在输入方面，支持对用户提示进行更精细控制的模型逐渐得以发展，从支持图像输入发展到支持区域输入[28]、[138]、[139]，甚至到支持像素输入[29]、[140]、[141]。具体而言，Shikra[28]支持区域级别的输入和理解==。用户可以通过引用以自然语言形式的边界框表示的特定区域，更灵活地与智能助手进行交互。Ferret[141]更进一步，通过设计一种混合表示方案来支持更灵活的指代。该模型支持不同形式的提示，包括点、框和草图。同样地，Osprey[29]通过利用一个分割模型[9]来支持点输入。在预训练分割模型的卓越能力辅助下，Osprey能够通过一次点击就指定单个实体或其一部分。在输出方面，随着输入支持的发展，定位能力也得到了提升。Shikra[28]支持**基于带有框注释的图像进行回复，从而带来更高的精度和更精细的指代体验**。LISA[142]进一步支持掩码级别的理解和推理，这使得像素级别的定位成为可能。 

**Modality Support.** Increased support for modalities is a tendency for MLLM studies. On the one hand, researchers have explored adapting MLLMs to support the input of more multimodal content, such as 3D point cloud [41], [143], [144], [145]. On the other hand, MLLMs are also extended to generate responses of more modalities, such as image [32], [146], [147], [148], audio [32], [147], [149], [150], and video [32], [151]. For example, NExT-GPT [32] proposes a framework that supports inputs and outputs of mixed modalities, specifically, combinations of text, image, audio, and video, with the help of diffusion models [152], [153] attached to the MLLM. The framework applies an encoder-decoder architecture and puts LLM as a pivot for understanding and reasoning.

**模态支持** ==增加对多种模态的支持是多模态大语言模型（MLLM）研究的一个趋势==。一方面，研究人员探索使多模态大语言模型能够适应并**支持更多多模态内容的输入**，例如三维点云数据[41]、[143]、[144]、[145]。另一方面，多模态大语言模型也被扩展到能够**生成更多模态的回复**，比如图像[32]、[146]、[147]、[148]，音频[32]、[147]、[149]、[150]以及视频[32]、[151]。 例如，NExT-GPT[32]提出了一个框架，该框架在连接到多模态大语言模型的扩散模型[152]、[153]的帮助下，支持混合模态的输入和输出，具体来说就是文本、图像、音频和视频的组合。这个框架采用了编码器-解码器架构，并将大语言模型作为理解和推理的核心。 

> 点云数据（point cloud data）是指**在一个三维坐标系统中的一组向量的集合**。 扫描资料以点的形式记录，每一个点包含有三维坐标，并且可以携带有关该点属性的其他信息，例如颜色、反射率、强度等。

**Language Support.** Current models are predominantly unilingual, probably due to the fact that high-quality nonEnglish training corpus is scarce. Some works have been devoted to developing multilingual models so that a broader range of users can be covered. VisCPM [33] transfers model capabilities to the multilingual setting by designing a multistage training scheme. Specifically, the scheme takes English as a pivotal language, with abundant training corpus. Utilizing a pre-trained bilingual LLM, the multimodal capabilities are transferred to Chinese by adding some translated samples during instruction tuning. Taking a similar approach, Qwen-VL [34] is developed from the bilingual LLM Qwen [58] and supports both Chinese and English. During pre-training, Chinese data is mixed into the training corpus to preserve the bilingual capabilities of the model, taking up 22.7% of the whole data volume.

**语言支持** 

目前的模型大多是单语的，这可能是因为高质量的非英语训练语料库较为稀缺。一些研究致力于==开发多语言模型，以便能够覆盖更广泛的用户群体==。 VisCPM[33]通过设计一个多阶段训练方案，将模型的能力迁移到多语言环境中。具体来说，**该方案以拥有丰富训练语料库的英语作为关键语言。利用一个预训练的双语大语言模型，在指令微调阶段添加一些翻译后的样本，从而将多模态能力迁移到中文上。** 采用类似的方法，Qwen-VL[34]是从双语大语言模型Qwen[58]开发而来的，支持中文和英文。在预训练阶段，中文数据被混入训练语料库中，以保留模型的双语能力，==中文数据占总数据量的22.7%==。 

**Scenario/Task Extension.** Apart from developing common general-purpose assistants, some studies have focused on more specific scenarios where practical conditions should be considered, while others extend MLLMs to downstream tasks with specific expertise.  

A typical tendency is to adapt MLLMs to more specific real-life scenarios. MobileVLM [63] explores developing small-size variants of MLLMs for resource-limited scenarios. Some designs and techniques are utilized for deployment on mobile devices, such as LLMs of smaller size and quantiza- tion techniques to speed up computation. Other works de- velop agents that interact with real-world [41], [154], [155], e.g. user-friendly assistants specially designed for Graphical User Interface (GUI), as exemplified by CogAgent [44], AppAgent [43], and Mobile-Agent [45]. These assistants excel in planning and guiding through each step to fulfill a task specified by users, acting as helpful agents for human- machine interaction. Another line is to augment MLLMs with specific skills for solving tasks in different domains, e.g. document understanding [38], [39], [156], [157] and medi- cal domains [35], [36], [37]. For document understanding, mPLUG-DocOwl [38] utilizes various forms of document- level data for tuning, resulting in an enhanced model in OCR-free document understanding. TextMonkey [39] incor- porates multiple tasks related to document understanding to improve model performance. Apart from conventional document image and scene text datasets, position-related tasks are added to reduce hallucinations and help mod- els learn to ground responses in the visual information. MLLMs can also be extended to medical domains by in- stilling knowledge of the medical domain. For example, LLaVA-Med [158] injects medical knowledge into vanilla LLaVA [20] and develops an assistant specialized in medical image understanding and question answering.

**场景/任务扩展** 除了开发通用的多功能智能助手外，一些研究聚焦于更特定的场景，在这些场景中需要考虑实际应用条件，而另一些研究则将多模态大语言模型（MLLMs）扩展到需要特定专业知识的下游任务中。 

```
场景/任务扩展
- 小尺寸变体
- 增强特定技能
	- 文档理解以及医疗领域
```

**一个典型的趋势是让多模态大语言模型适应更具体的现实生活场景**。MobileVLM[63]探索为资源受限的场景开发多模态大语言模型的==小尺寸变体==。一些设计和技术被用于在移动设备上进行部署，比如使用更小尺寸的大语言模型以及量化技术来加快计算速度。 其他研究开发了能够与现实世界交互的智能体[41]、[154]、[155]，例如专门为图形用户界面（GUI）设计的用户友好型智能助手，如CogAgent[44]、AppAgent[43]和Mobile-Agent[45]。这些智能助手擅长规划并引导每一个步骤，以完成用户指定的任务，在人机交互中扮演着得力助手的角色。 另一个方向是为多模态大语言模型==增强特定技能==，以解决不同领域的任务，例如**文档理解[38]、[39]、[156]、[157]以及医疗领域[35]、**[36]、[37]。 在文档理解方面，mPLUG-DocOwl[38]利用各种形式的文档级数据进行微调，从而在无需光学字符识别（OCR）的文档理解方面提升了模型性能。TextMonkey[39]纳入了与文档理解相关的多项任务，以提高模型性能。除了传统的文档图像和场景文本数据集外，还添加了与位置相关的任务，以减少幻觉现象，并帮助模型学会将回复基于视觉信息。 **多模态大语言模型也可以通过灌输医学领域的知识扩展到医疗领域**。例如，LLaVA-Med[158]将医学知识注入到基础的LLaVA[20]模型中，并开发出了一个专门用于医学图像理解和问答的智能助手。 

## 6 MULTIMODAL HALLUCINATION

多模态幻觉

Multimodal hallucination refers to the phenomenon of responses generated by MLLMs being inconsistent with the image content [77]. As a fundamental and important problem, the issue has received increased attention. In this section, we briefly introduce some related concepts and research development.

**多模态幻觉是指多模态大语言模型（MLLMs）生成的响应与图像内容不一致的现象[77]**。作为一个基础且重要的问题，这一问题已受到越来越多的关注。在本节中，我们将简要介绍一些相关概念和研究进展。

### 6.1 Preliminaries

6.1 预备知识 

Current research on multimodal hallucinations can be fur- ther categorized into three types [159]:

1 ) Existence Hallucination is the most basic form, meaning that models incorrectly claim the existence of certain objects in the image. 

2 ) Attribute Hallucination means describing the attributes of certain objects in a wrong way, e.g. failure to identify a dog’s color correctly. It is typically associated with ex- istence hallucination since descriptions of the attributes should be grounded in objects present in the image. 

3 ) Relationship Hallucination is a more complex type and is also based on the existence of objects. It refers to false descriptions of relationships between objects, such as relative positions and interactions. In what follows, we first introduce some specific eval- uation methods (§6.2), which are useful to gauge the per- formance of methods for mitigating hallucinations (§6.3). Then, we will discuss in detail the current methods for reducing hallucinations, according to the main categories each method falls into.

目前关于多模态幻觉的研究可以进一步分为三类[159]：

 1 ）**存在性幻觉是最基本的形式**，这意味着模型错误地声称图像中存在某些物体。

 2 ）**属性幻觉指的是以错误的方式描述某些物体的属性，例如未能正确识别一只狗的颜色**。它通常与存在性幻觉相关联，因为对属性的描述应该基于图像中实际存在的物体。

 3 ）**关系幻觉是一种更为复杂的类型，同样基于物体的存在**。==它是指对物体之间关系的错误描述，比如相对位置和相互作用==。 接下来，我们首先介绍一些具体的评估方法（第6.2节），这些方法对于衡量缓解幻觉的方法的性能很有用（第6.3节）。然后，我们将根据每种方法所属的主要类别，详细讨论目前减少幻觉的方法。 

### 6.2 Evaluation Methods

6.2 评估方法

CHAIR [160] is an early metric that evaluates hallucination levels in open-ended captions. The metric measures the proportion of sentences with hallucinated objects or hallucinated objects in all the objects mentioned. In contrast, POPE [132] is a method that evaluates closed-set choices. Specifically, multiple prompts with binary choices are formulated, each querying if a specific object exists in the image. The method also covers more challenging settings to evaluate the robustness of MLLMs, with data statistics taken into consideration. The final evaluation uses a simple watchword mechanism, i.e. by detecting keywords “yes/no”, to convert open-ended responses into closedset binary choices. With a similar evaluation approach, MME [123] provides a more comprehensive evaluation, covering aspects of existence, count, position and color, as exemplified in [77].

**CHAIR[160]是一种早期的衡量指标，用于评估开放式图像描述中的幻觉程度。**该指标衡量的是在所有提及的物体中，包含幻觉物体的句子，或者幻觉物体本身所占的比例。相比之下，**POPE[132]是一种评估封闭集选择的方法。具体来说，它会制定多个带有二选一选项的提示，每个提示都在询问图像中是否存在某个特定的物体。**这种方法还涵盖了更具挑战性的评估设置，以考量多模态大语言模型（MLLMs）的鲁棒性，同时也考虑了数据统计情况。最终的评估采用了一种简单的关键词机制，即通过检测“是/否”这样的关键词，将开放式的回复转换为封闭集的二选一选项。采用类似的评估方法，MME[123]提供了更全面的评估，涵盖了物体的存在性、数量、位置和颜色等方面，如文献[77]中所举例说明的那样。 

```
- CHAIR:该指标衡量的是在所有提及的物体中，包含幻觉物体的句子，或者幻觉物体本身所占的比例
- POPE[132]是一种评估封闭集选择的方法。具体来说，它会制定多个带有二选一选项的提示，每个提示都在询问图像中是否存在某个特定的物体。
- MME[123]提供了更全面的评估，涵盖了物体的存在性、数量、位置和颜色等方面。
```

Different from previous approaches that use matching mechanisms to detect and decide hallucinations, HaELM [161] proposes using text-only LLMs as a judge to automatically decide whether MLLMs’ captions are correct against reference captions. In light of the fact that text-only LLMs can only access limited image context and require reference annotations, Woodpecker [77] uses GPT-4V to directly assess model responses grounded in the image. FaithScore [162] is a more fine-grained metric based on a routine that breaks down descriptive sub-sentences and evaluates each sub-sentence separately. Based on previous studies, AMBER [163] is an LLM-free benchmark that encompasses both discriminative tasks and generative tasks and involves three types of possible hallucinations (see §6.1).

与以往使用匹配机制来检测和判定幻觉的方法不同，HaELM[161]提出**使用仅处理文本的大语言模型（LLMs）作为评判者，依据参考描述自动判定多模态大语言模型（MLLMs）生成的图像描述是否正确**。鉴于仅处理文本的大语言模型只能获取有限的图像上下文信息且需要参考注释，Woodpecker[77]使用GPT-4V来直接评估基于图像生成的模型回复。FaithScore[162]是一种更细粒度的衡量指标，它基于一种常规方法，即将描述性子句进行拆解，并分别对每个子句进行评估。在先前研究的基础上，AMBER[163]是一个无需大语言模型的基准测试，它涵盖了判别性任务和生成性任务，并且涉及三种可能的幻觉类型（详见6.1节）。 
