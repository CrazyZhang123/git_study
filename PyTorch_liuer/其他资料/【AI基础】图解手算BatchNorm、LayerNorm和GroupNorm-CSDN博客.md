 

  这几天整理对比了一下网络中几个常用的Norm的方法，之前也看过，网上有很多讲的非常详细的资料，以前看一下理解了就过了，时间长了就模糊了，此次自己亲手算了一遍，加深了印象，特此整理一下，以便之后的回顾。

**设置一个Tensor，其Size为\[3,4,2,2\]，便于之后的理解**  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6f58f6b8e9f6355693c8281c893fd95d.png)

### 一、[BatchNorm](https://so.csdn.net/so/search?q=BatchNorm&spm=1001.2101.3001.7020)

  [BatchNorm详解](https://blog.csdn.net/qq_37541097/article/details/104434557?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164570421316780274134036%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=164570421316780274134036&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-104434557.nonecase&utm_term=batch&spm=1018.2226.3001.4450)  
  所有Norm方法无非都是减均值再除以标准差，无非是在哪个尺度上进行该操作的差异，而BatchNorm是在一个batch上，同一个通道上面进行Norm，那么有多少个通道就会计算多少个均值和标准差。  
  **Mean=所有batch同一个通道上的均值**  
  **Std=所有batch同一个通道上的标准差**  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/abb52caa54265a2ce7348b06954bb4bc.png)  
**图中红色虚线框为Norm的尺度，黄色矩形框中为更新后的值**

n n . B a t c h N o r m 2 d ( n u m \_ f e a t u r e s = 4 ) {nn.BatchNorm2d(num\\\_features=4)} nn.BatchNorm2d(num\_features\=4)其中 n u m \_ f e a t u r e s {num\\\_features} num\_features为Channel数  
代码验证：

```python
import torch
import torch.nn as nn

a = torch.tensor([[[[1.,1],[1,1]],[[0,1],[1,0]],[[0,0],[0,1]],[[1,1],[0,0]]],
                  [[[2.,2],[0,0]],[[2,0],[1,1]],[[1,0],[0,2]],[[2,1],[1,0]]],
                  [[[3.,1],[2,2]],[[3,0],[0,2]],[[2,3],[1,2]],[[3,3],[2,1]]]])
batch = nn.BatchNorm2d(num_features=4)
b = batch(a)
```

输出结果与手算一致：

```python
tensor([[[[-0.3922, -0.3922],
          [-0.3922, -0.3922]],
         [[-0.9611,  0.0874],
          [ 0.0874, -0.9611]],
         [[-1.0000, -1.0000],
          [-1.0000,  0.0000]],
         [[-0.2474, -0.2474],
          [-1.2372, -1.2372]]],

        [[[ 0.7845,  0.7845],
          [-1.5689, -1.5689]],
         [[ 1.1358, -0.9611],
          [ 0.0874,  0.0874]],
         [[ 0.0000, -1.0000],
          [-1.0000,  1.0000]],
         [[ 0.7423, -0.2474],
          [-0.2474, -1.2372]]],

        [[[ 1.9611, -0.3922],
          [ 0.7845,  0.7845]],
         [[ 2.1842, -0.9611],
          [-0.9611,  1.1358]],
         [[ 1.0000,  2.0000],
          [ 0.0000,  1.0000]],
         [[ 1.7320,  1.7320],
          [ 0.7423, -0.2474]]]], grad_fn=<NativeBatchNormBackward>)
```

### 二、[LayerNorm](https://so.csdn.net/so/search?q=LayerNorm&spm=1001.2101.3001.7020)

  [LayerNorm详解](https://blog.csdn.net/qq_37541097/article/details/117653177?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164570699716780366593622%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=164570699716780366593622&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-117653177.nonecase&utm_term=Layernorm&spm=1018.2226.3001.4450)  
  LayerNorm可以在3种不同的尺度进行  
  第一种： n n . L a y e r N o r m ( n o r m a l i z e d \_ s h a p e = \[ 4 , 2 , 2 \] ) {nn.LayerNorm(normalized\\\_shape=\[4,2,2\])} nn.LayerNorm(normalized\_shape\=\[4,2,2\])  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a10e0df44822ed02aa427ebb0bb787fa.png)

```python
layer1 = nn.LayerNorm(normalized_shape=[4,2,2])
c1 = layer1(a)
```

输出结果与手算一致：

```python
tensor([[[[ 0.8819,  0.8819],
          [ 0.8819,  0.8819]],
         [[-1.1339,  0.8819],
          [ 0.8819, -1.1339]],
         [[-1.1339, -1.1339],
          [-1.1339,  0.8819]],
         [[ 0.8819,  0.8819],
          [-1.1339, -1.1339]]],

        [[[ 1.2851,  1.2851],
          [-1.1339, -1.1339]],
         [[ 1.2851, -1.1339],
          [ 0.0756,  0.0756]],
         [[ 0.0756, -1.1339],
          [-1.1339,  1.2851]],
         [[ 1.2851,  0.0756],
          [ 0.0756, -1.1339]]],

        [[[ 1.1339, -0.8819],
          [ 0.1260,  0.1260]],
         [[ 1.1339, -1.8898],
          [-1.8898,  0.1260]],
         [[ 0.1260,  1.1339],
          [-0.8819,  0.1260]],
         [[ 1.1339,  1.1339],
          [ 0.1260, -0.8819]]]], grad_fn=<NativeLayerNormBackward>)
```

  第二种： n n . L a y e r N o r m ( n o r m a l i z e d \_ s h a p e = \[ 2 , 2 \] ) {nn.LayerNorm(normalized\\\_shape=\[2,2\])} nn.LayerNorm(normalized\_shape\=\[2,2\])  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/be151d90e353ae16f89953c10ded4045.png)

```python
layer2 = nn.LayerNorm(normalized_shape=[2,2])
c2 = layer2(a)
```

输出结果与手算一致：

```python
tensor([[[[ 0.0000,  0.0000],
          [ 0.0000,  0.0000]],
         [[-1.0000,  1.0000],
          [ 1.0000, -1.0000]],
         [[-0.5773, -0.5773],
          [-0.5773,  1.7320]],
         [[ 1.0000,  1.0000],
          [-1.0000, -1.0000]]],

        [[[ 1.0000,  1.0000],
          [-1.0000, -1.0000]],
         [[ 1.4142, -1.4142],
          [ 0.0000,  0.0000]],
         [[ 0.3015, -0.9045],
          [-0.9045,  1.5075]],
         [[ 1.4142,  0.0000],
          [ 0.0000, -1.4142]]],

        [[[ 1.4142, -1.4142],
          [ 0.0000,  0.0000]],
         [[ 1.3471, -0.9622],
          [-0.9622,  0.5773]],
         [[ 0.0000,  1.4142],
          [-1.4142,  0.0000]],
         [[ 0.9045,  0.9045],
          [-0.3015, -1.5075]]]], grad_fn=<NativeLayerNormBackward>)
```

  第三种： n n . L a y e r N o r m ( n o r m a l i z e d \_ s h a p e = 2 ) {nn.LayerNorm(normalized\\\_shape=2)} nn.LayerNorm(normalized\_shape\=2)  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7290a7fd574b3fe2ce2b7b051eee4d92.png)  
唔，太多了，不想算了…抽检吧

```python
layer3 = nn.LayerNorm(normalized_shape=2)
c3 = layer3(a)
```

输出：

```python
tensor([[[[ 0.0000,  0.0000],
          [ 0.0000,  0.0000]],
         [[-1.0000,  1.0000],
          [ 1.0000, -1.0000]],
         [[ 0.0000,  0.0000],
          [-1.0000,  1.0000]],
         [[ 0.0000,  0.0000],
          [ 0.0000,  0.0000]]],

        [[[ 0.0000,  0.0000],
          [ 0.0000,  0.0000]],
         [[ 1.0000, -1.0000],
          [ 0.0000,  0.0000]],
         [[ 1.0000, -1.0000],
          [-1.0000,  1.0000]],
         [[ 1.0000, -1.0000],
          [ 1.0000, -1.0000]]],

        [[[ 1.0000, -1.0000],
          [ 0.0000,  0.0000]],
         [[ 1.0000, -1.0000],
          [-1.0000,  1.0000]],
         [[-1.0000,  1.0000],
          [-1.0000,  1.0000]],
         [[ 0.0000,  0.0000],
          [ 1.0000, -1.0000]]]], grad_fn=<NativeLayerNormBackward>)
```

### 三、GroupNorm

  [GroupNorm详解](https://blog.csdn.net/qq_37541097/article/details/118016048?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164570962416780274176855%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fblog.%2522%257D&request_id=164570962416780274176855&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~blog~first_rank_ecpm_v1~rank_v31_ecpm-1-118016048.nonecase&utm_term=GroupNorm&spm=1018.2226.3001.4450)  
   n n . G r o u p N o r m ( n u m \_ g r o u p s = 2 , n u m \_ c h a n n e l s = 4 ) ) {nn.GroupNorm(num\\\_groups=2, num\\\_channels=4))} nn.GroupNorm(num\_groups\=2,num\_channels\=4))  
  其中 n u m \_ g r o u p s {num\\\_groups} num\_groups为分组数， n u m \_ c h a n n e l s {num\\\_channels} num\_channels为通道数  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ffec3edf8992ba95e51e799129da4880.png)

```python
group = nn.GroupNorm(num_groups=2, num_channels=4)
d = group(a)
```

输出结果与手算一致：

```python
tensor([[[[ 0.5773,  0.5773],
          [ 0.5773,  0.5773]],
         [[-1.7320,  0.5773],
          [ 0.5773, -1.7320]],
         [[-0.7746, -0.7746],
          [-0.7746,  1.2910]],
         [[ 1.2910,  1.2910],
          [-0.7746, -0.7746]]],

        [[[ 1.1547,  1.1547],
          [-1.1547, -1.1547]],
         [[ 1.1547, -1.1547],
          [ 0.0000,  0.0000]],
         [[ 0.1601, -1.1209],
          [-1.1209,  1.4411]],
         [[ 1.4411,  0.1601],
          [ 0.1601, -1.1209]]],

        [[[ 1.2376, -0.5625],
          [ 0.3375,  0.3375]],
         [[ 1.2376, -1.4626],
          [-1.4626,  0.3375]],
         [[-0.1601,  1.1209],
          [-1.4411, -0.1601]],
         [[ 1.1209,  1.1209],
          [-0.1601, -1.4411]]]], grad_fn=<NativeGroupNormBackward>)
```

**注意：只有BatchNorm与Batch有关，LayerNorm和GroupNorm都不会在Batch上进行计算**

本文转自 <https://blog.csdn.net/qq_43426908/article/details/123119919?spm=1001.2014.3001.5506>，如有侵权，请联系删除。