---
created: 2025-01-23T00:44
updated: 2025-01-23T00:44
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