## Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

- **36th Conference on Neural Information Processing Systems (NeurIPS 2022).**

链接: https://arxiv.org/pdf/2201.11903

## Abstract

We explore how generating a chain of thought—a series of **intermediate(中间的)** reasoning steps—significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in **sufficiently(充分地，十分)** large language models via a simple method called chain-of-thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. 

Experiments on three large language models show that chain-of-thought prompting improves performance on a range of **arithmetic(算术), commonsense(常识), and symbolic reasoning(符号推理)** tasks. The empirical(经验主义的; ) gains can be striking(be striking 引人注目). For instance, prompting a PaLM 540B with just eight chain-of-thought exemplars achieves state-of-the-art accuracy on the GSM8K benchmark of math word problems, **surpassing(超过，优于)** even finetuned GPT-3 with a **verifier(验证器).** 

**我们探索了生成思维链（一系列中间推理步骤）如何显著提升大语言模型执行复杂推理的能力。**具体而言，我们展示了通过一种名为思维链提示的简单方法，这种推理能力如何在足够大的语言模型中自然涌现。在这种方法中，会在提示环节提供一些思维链示例作为范例。 

对三种大语言模型进行的实验表明，思维链提示法能提高模型在一系列算术、常识和符号推理任务上的表现。实际效果提升十分显著。例如，**仅用八个思维链示例对5400亿参数的PaLM模型进行提示**，就能在GSM8K数学应用题基准测试中达到最先进的准确率，甚至**超过了经过微调并配备验证器的GPT-3模型。** 

<img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250311171536661.png" alt="image-20250311171536544" style="zoom: 67%;" />

Figure 1: Chain-of-thought prompting enables large language models to tackle complex arithmetic, commonsense, and symbolic reasoning tasks. Chain-of-thought reasoning processes are highlighted.

图1：思维链提示使大语言模型能够处理复杂的算术、常识和符号推理任务。图中突出显示了思维链推理过程。 

## 1 Introduction

The NLP landscape has recently been revolutionized by language models (Peters et al., 2018; Devlin et al., 2019; Brown et al., 2020, inter alia). Scaling up the size of language models has been shown to confer a range of benefits, such as improved performance and sample efficiency (Kaplan et al., 2020; Brown et al., 2020, inter alia). However, scaling up model size alone has not proved sufficient for achieving high performance on challenging tasks such as arithmetic, commonsense, and symbolic reasoning (Rae et al., 2021).

自然语言处理（NLP）领域最近因语言模型而发生了革命性变化（彼得斯等人，2018；德夫林等人，2019；布朗等人，2020等）。研究表明，扩大语言模型的规模能带来诸多好处，比如提高性能和样本效率（卡普兰等人，2020；布朗等人，2020等）。然而，仅扩大模型规模并不足以在诸如算术、常识和符号推理等具有挑战性的任务上实现高性能（雷等人，2021）。 