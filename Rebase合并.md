---
created: 2025-01-23T00:44
updated: 2025-01-23T21:51
---
# Rebase 处理合并冲突

##### 1.从远程仓orgin仓的 [remoteBranchName] 分支下载到本地，并在本地新建一个对应 [localBranchName] 分支

```
 // remoteBranchName和localBranchName 都是 redmiBook
 git fetch origin redmiBook
```

![image-20250120223442368](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120223442368.png)

##### 2.切换到远程分支下载的新建的本地分支 [localBranchName] 

```
// [localBranchName] ，即redmiBook
git checkout redmiBook
```

![image-20250120223412542](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120223412542.png)

##### 3、rebase 目标合并的本地分支 [targetBranchName] 
是想要把redmiBook分支分支合并到main分支
```
 // [targetBranchName] 也就是 main
 git rebase main
```

![image-20250120225550739](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120225550739.png)

>  rebase的前提是本地分支先add，commit到本地仓库才行。![image-20250120224620360](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120224620360.png)
>
>  main分支有未保存的修改
>
>  ![image-20250120224757961](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120224757961.png)
>
>  ```
>  git add .
>  git commit -m "main-修改-2025/1/19"
>  ```
>
>  ![image-20250120225039292](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120225039292.png)
>
>  ![image-20250120225101179](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120225101179.png)

##### 4、切换到main分支上（为了进行更新main的操作）

```
git checkout main
```

##### 5、#把main rebase 到 redmiBook 分支上（由于 redmiBook 继承自 main，所以 git 只是简单的把 main 分支的引用向前移动了一下而已。）

> 没有5的话，main还是原始的位置，没有redmiBook的提交，只是本地分支，将redmiBook分支 变基 到 main 上，也顺便同步了远程仓库，见下方。

```
 git rebase redmiBook
```

![image-20250120232249238](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120232249238.png)

6、(可选)删除 本地 remiBook

```
git branch -d remiBook
```

 

git branch 图

执行完 4后

![image-20250120230837557](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120230837557.png)

执行完5后

![image-20250120231944018](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120231944018.png)

最终

![image-20250120232156046](https://gitee.com/zhang-junjie123/picture/raw/master/image/image-20250120232156046.png)



## Merge和rebase区别

> 在 Git 的交汇处，每一次选择都是代码旅程的新起点。

在 Git 中，Merge 和 Rebase 是两种常用的分支整合方式，但是一些初学的小伙伴可能不知道它们之间有什么区别，以及二者该怎么选择。

本文将深入探讨 Merge 和 Rebase 的区别，帮助你更好地理解和运用它们，以提高代码管理的效率和质量。

### 场景 | 举个栗子

为了彻底地搞懂它们之间的区别，不妨先举个例子，假设现在有一个 main 分支和一个 feature 分支，如图（图中数字为提交的先后次序，即最早的提交是1，然后是2，其次是3）：  

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250124101121948.jpeg)

而你正在开发 feature 分支，目前最新的提交是4，但是**你的同事突然向 main 分支推送了一个提交5，**而且恰巧提交5中的某些代码你正好需要用到，此时你的想法肯定是**将 main 分支中的代码合并到你的 feature 分支中来。**

### Merge | 分支交汇的交叉路口

Merge的原理很简单，就是**将要合并分支的最新提交组合成一个新的提交，并且插入到目标分支中。**

现在你想要把 main 分支 merge 到你的 feature 分支上去，那么 git 会把两个分支的最新提交4和5合并成一个提交，并且合入目标分支 feature，也就是：

```bash
git checkout feature
git merge main
```

最终会得到：  

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250124101128272.jpeg)

现在，你同事的代码已经合并到你的 feature 分支中来了，你也可以愉快地开始编写代码了！

不过有一个比较难受的地方就是，假设在你开发 feature 分支的时候你的同事频繁地往 main 分支推送你需要的代码，merge 的方式会造成很多分叉，导致分支结构十分复杂和紊乱，污染你的提交历史。

**但是 merge 会保留所有历史提交，不会破坏历史提交的结构，这一点在多人协作的时候很重要！**

### Rebase | 提交历史的线性编织

[rebase](https://zhida.zhihu.com/search?content_id=240694154&content_type=Article&match_order=1&q=rebase&zhida_source=entity) 的使用方式与 merge 类似：

```bash
git checkout feature
git rebase main
```

与 merge 不同的是，rebase 并不会保留原有的提交，而是会**创建当前[分支比](https://zhida.zhihu.com/search?content_id=240694154&content_type=Article&match_order=1&q=%E5%88%86%E6%94%AF%E6%AF%94&zhida_source=entity)目标分支更新的所有提交的副本，**在上述例子中（将 feature [变基](https://zhida.zhihu.com/search?content_id=240694154&content_type=Article&match_order=1&q=%E5%8F%98%E5%9F%BA&zhida_source=entity)到 main）就是 2' 和 4'，然后将 2' 和 4' 按次序插入目标分支末尾：  

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250124101131635.jpeg)

这样就完成了一个 rebase 的过程（注意，这条分支是 feature 分支而不是 main 分支，原先的提交2和提交4已经被抛弃了），feature 分支实际的样子是：  

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250124101146651.png)

你可以理解为，把 feature 分支 rebase（变基）到 main 分支的意思是：让你现在的 feature 分支基于最新的 main 分支重新开发（创建副本并且移动到末尾）  

是不是很简单？  

不，一点都不简单。  

**rebase 有一个非常大的坑，那就是它会改变提交历史。**  

想象一下你和你同事都在开发 main 分支，然后在你开发的时候，你的同事突然把 main 变基了，假设你们之前是这样的：

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250124101200520.jpeg)

结果现在变成了这样子：

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250124101225915.jpeg)

原来的提交5直接消失了，你的分支已经跟 main 分支匹配不上了，那现在你还能正常把你的分支合入到 [main 分支](https://zhida.zhihu.com/search?content_id=240694154&content_type=Article&match_order=11&q=main+%E5%88%86%E6%94%AF&zhida_source=entity)中吗？显然不可以。  

**所以，main 分支是万万不能使用 rebase 的！！！**  

在你打算 rebase 的时候，一定要想想是否还有别人也在开发这个分支。

### 适用场景

从上面的例子中不难发现，merge 和 rebase 最大的区别在于是否会保留原有的提交（或者说破坏原有的提交结构）。  

**merge** 会对提交历史进行保留，很显然**更适合多人协作开发的场景**，因为如果出现问题也可以追溯到历史的每一次提交。  

而 **rebase** 则是会让提交历史更加简洁易读，保持提交历史的线性结构，所以**更适合个人开发和整理分支的情况。**  

如果我想要把某个特性分支 feature\_xxx 合并到 main 分支中的时候，最好的方式就是 merge，而当我一个人需要开发某个 feature\_xxx 分支的时候，最好的方式就是 rebase。  

**一句话概括就是，merge 适合团队协作，而 rebase 适合一个人开发的分支。**

本文转自 <https://zhuanlan.zhihu.com/p/686538265>，如有侵权，请联系删除。