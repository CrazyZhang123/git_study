# 部署glm3-6b到能微调的环境部署

注：该文只是微调前的部署glm的配置环境，可以理解成lora微调的前置条件：[ChatGLM3/finetune_demo/lora_finetune.ipynb at main · THUDM/ChatGLM3 · GitHub](https://github.com/THUDM/ChatGLM3/blob/main/finetune_demo/lora_finetune.ipynb)

1、创建一个python3.11的虚拟环境

2、进入虚拟环境，查看服务器的显卡配置，能用的cuda版本

​	根据显示的cuda版本，去Pytorch官网找2.3.0并且比显示Cuda版本低的命令

> 比如这个服务器显示的cuda是12.4.0
>
> 那么找到pytorch官网2.3.0对应的那个命令
>
> 等着下载pytorch，这个命令时间很长，可以换成清华源，依旧时间很长

![image-20250217162449293](https://gitee.com/liubw21/picture/raw/master/img/202502171624902.png)

![image-20250217162410585](https://gitee.com/liubw21/picture/raw/master/img/202502171624669.png)

3、接着进入glm运行文件的配置，使用pip配置依赖

![image-20250217162626905](https://gitee.com/liubw21/picture/raw/master/img/202502171626016.png)

依赖安装完成之后如下，这时已经可以进行基本推理：
![image-20250217162717795](https://gitee.com/liubw21/picture/raw/master/img/202502171627207.png)

![image-20250217162803374](https://gitee.com/liubw21/picture/raw/master/img/202502171628621.png)

于是，此时想着能否进行网页版的推理，然而显示包不全。

![image-20250217162919158](https://gitee.com/liubw21/picture/raw/master/img/202502171629202.png)

这时发现缺少peft包，在微调目录下，还存在一个requirements.txt文档，刚好这个文档中存在peft包，便想着能否直接继续pip，然而结果不太行：

![image-20250217163048672](https://gitee.com/liubw21/picture/raw/master/img/202502171630655.png)

因为在安装到最后，会说mpi4py包(requirements.txt中的最后一个包)安装报错，然而此时，网上查阅资料显示前面的包应该已经安装完毕，比如：peft已经安装好了，然而再试试网页版的示例，仍然显示peft包未成功安装。接着再用pip命令查看发现，peft，jieba等包并未安装成功(猜测：如果使用pip install -r requirements.txt安装依赖，如果最后显示包未安装成功，可能前面的包即使安装好了也会回滚)。

![image-20250217163213478](https://gitee.com/liubw21/picture/raw/master/img/202502171632358.png)

所以，可以接着将requirements.txt文档中的最后一行mpi4py的那行删除，然后重新pip，成功截图如下：然后再查看peft包是否安装成功，显示确实存在。

![image-20250217163618307](https://gitee.com/liubw21/picture/raw/master/img/202502171636600.png)

![image-20250217163648991](https://gitee.com/liubw21/picture/raw/master/img/202502171636415.png)

那么再回去试试网页推理示例能否运行，然而还是不行。还是会报错，说peft并不能正常从transformers中导入**模块，但是普通命令行的推理示例正常。网上说两者版本不匹配，要解决要么要将transformers升级（我试过两次，第一次升级之后，可以正常运行，但是在后面lora微调时，会无法评估，后来发现升级之后命令行推理demo不能正常运行；第二次先装torch，之后再pip，再将transformers升级时显示出错；又由于requirements.txt文档中写的peft模块是大于等于0.10.0，但是pip安装的是0.14.0，那么将其直接降级到0.10.0），要么将peft降级。

![image-20250217163813058](https://gitee.com/liubw21/picture/raw/master/img/202502171638244.png)

当时，我想着可能基本的依赖并未配置成功，所以我又重新pip了一次，但是还是不能运行网页版示例，后来才想到降级peft。

![image-20250217164304486](https://gitee.com/liubw21/picture/raw/master/img/202502171643492.png)

![image-20250217164354477](https://gitee.com/liubw21/picture/raw/master/img/202502171643663.png)

![image-20250217164500381](https://gitee.com/liubw21/picture/raw/master/img/202502171645497.png)

将peft降级之后，继续尝试运行网页版示例。

![image-20250217164617961](https://gitee.com/liubw21/picture/raw/master/img/202502171646123.png)

于此，想试着去实现lora微调，但是直接运行发现缺少nltk包，直接pip安装之后，就可lora微调了。

![image-20250217164819385](https://gitee.com/liubw21/picture/raw/master/img/202502171648317.png)

lora微调最后完毕截图：

![image-20250217164854571](https://gitee.com/liubw21/picture/raw/master/img/202502171648795.png)