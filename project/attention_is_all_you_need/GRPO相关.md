
## 1、GRPO损失
$$
\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min \left( \frac{\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} \hat{A}_{i,t}, \text{clip} \left( \frac{\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}, 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{\text{KL}}[\pi_{\theta} \| \pi_{\text{ref}}] \right]
$$

GRPO loss看起来复杂，可以看成三部分：

- 第一个连加的G为一个样本的采样数量，第二个$|o_i|$是第条输出的采样长度
- $min(\dots,\dots)$在 里，与标准PPO差异不大，这里的advantage我们已经提前计算好了，在一条采样回答数据中对于不同的 优势值都一样的。另外这里的ratio对比的是新旧策略。 这个式子是token-level的。
- KL项因子$\beta$ 控制约束力度，KL计算的是新策略和参考策略。

## 1.1 手撕GRPO

