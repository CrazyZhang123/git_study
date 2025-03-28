### 初识Dockerfile

Dockerfile就是用来构建docker镜像的构建文件!先体验一下!

```

```

### 数据卷容器

```
docker run -it --name docker01 zjj/centos:v1.0
docker run -it --name docker02 --volumes-from docker01 zjj/centos:v1.0
```

![img](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250227171639333.png)

因为我们的同步目录是/data1/zjj/datatest1 ，所以只有容器里面的这个datatest1,在data1或zjj是无法实现数据同步的。

```
zhangjunjie@amax-iset:/data1/zjj/docker-test-volume$ cat Dockerfile1
FROM centos

VOLUME /data1/zjj/datatest1 /data2/zjj/datatest2

CMD echo "hello world"
CMD /bin/bash
```

创建第3个容器

```
docker run -it --name docker03 --volumes-from docker01  zjj/centos:v1
```

==新创建的容器依然可以看到这个同步的文件。==

![img](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250227191207931.png)

```
# 测试，可以删除docker01,在docker02,docker03中依然可以查看搭配test.txt文件!
```

![img](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250227191742776.png)

> 可能是容器内部在主机上开辟一块内存并指向这块内存，使用挂载就让主机上的挂载目录也指向这块内存，然后共同拥有这块内存的读写权限

结论:

容器间配置信息的传递，数据卷容器的生命周期一直持续到没有容器使用为止。

但是一旦持久化到了本地，这个时候，本地的数据是不会删除的!

## DockerFile

### Dockerfile介绍

dockerfile是用来构建docker镜像的文件! 命令参数脚本!

构建步骤:

1. 编写一个dockerfile 文件
2. docker build 构建成为一个镜像
3. docker run 运行镜像
4. docker push 发布镜像(DockeHub、阿里云镜像仓库)

dockerfile查看官方怎么写的

![img](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250227193723182.png)

![img](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250227200118437.png)

很多官方镜像都是基础包，很多功能没有，我们通常回自己搭建镜像!

官方可以，我们也可以!

### Dockfile构建过程



**基础知识:**

1. 每个保留关键字(指令)都必须是大写字母
2. 执行顺序从上到下
3. #表示注释
4. 每一个指令都会创建提交一个新的镜像层，并提交！
5. <img src="https://gitee.com/zhang-junjie123/picture/raw/master/image/20250227205539097.png" alt="img" style="zoom:50%;" />

dockerfile是面向开发的，我们以后发布项目，做镜像，就需要编写dockerfile,这个文件十分简单!

Docker镜像逐渐成为企业交付的标准，必须要掌握!

Dockerfile: 构建分解，定义了一切的步骤，源代码

Dockerimages:通过Dockerfile构建生成的镜像，最终发布和运行的产品!

Docker容器：容器就是镜像运行起来提供服务

### DockerFile的指令

```shell
FROM		# 基础镜像，一切从这里开始构建
MAINTAINER  # 镜像是谁写的，姓名+邮箱(已弃用)
LABEL		# 添加镜像的元数据，使用键值对的形式。
RUN         # 镜像构建的时候需要执行的命令
ADD			# 步骤，tomcat镜像，这个tomcat压缩包，添加的内容
WORKDIR		# 镜像的工作目录
VOLUME		# 挂载卷的位置
EXPOSE		# 指定暴露端口
RUN			# 在构建过程中在镜像中执行命令。
CMD			# 指定这个容器启动的时候需要运行的命令，只有最后一个会生效，可被替代
ENTRYPOINT	# 指定这个容器启动的时候需要运行的命令，追加
ONBUILD		# 当构建一个被继承 DockerFile 这个时候就会运行 ONBUILD 的指令，触发指令。
COPY		# 类似ADD 命令，将文件拷贝到镜像中
ENV			# 构建的时候设置环境变量
```

> 命令: docker run    -l
>
> CMD 写的是 ls -a 						--> 最终执行结果: 执行 -l 报错,CMD的直接被替换了
> ENTRYPOINT 写的是 ls -a		   --> 最终执行结果: 执行 ls -al,在后面添加

#### CMD 

`dockerfile`中只能包含一个`CMD`指令，如果存在多个，只有最后一个`CMD`生效。

- `CMD`的主要用途是为执行容器提供默认值。这些默认值可以包括可执行文件，或者也可以省略可执行文件，但在这种情况下，必须指定一个`ENTRYPOINT`。

### 实战测试

Docker Hub中 99%镜像都是从基础镜像过来的 FROM scratch, 然后配置需要的软件和配置来进行构建。

> 创建一个自己的ubuntu

1. **编写Dockerfile的文件**

```shell
FROM ubuntu

LABEL author=zhangjunjie

ENV MYPATH=/usr/local

WORKDIR $MYPATH

RUN apt update && apt install -y vim
RUN apt install -y net-tools

EXPOSE 80

# CMD ["echo", "$MYPATH"] 这种会把$MYPATH解析为字符串
CMD ["bash","-c","echo $MYPATH"]

```

2.**构建文件**

```shell
docker build -f mydockerfile_ubuntu -t zjj_ubuntu:1.0 .
```

输出

```shell
2025/02/27 22:45:04 in: []string{}
2025/02/27 22:45:04 Parsed entitlements: []
[+] Building 2.9s (8/8) FINISHED                                                                         docker:default                                                                                0.0s
 => [1/4] FROM docker.io/library/ubuntu:latest@sha256:72297848456d5d37d1262630108ab308d3e9ec7ed1c3286a32fe0985661  0.0s
...
 => => naming to docker.io/library/zjj_ubuntu:1.0
```

**测试** 

默认就是在run的时候执行CMD，所以就是输出

![](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250227224716780.png)



> CMD 和 ENTRYPOINT的区别

```
CMD			# 指定这个容器启动的时候需要运行的命令，只有最后一个会生效，可被替代
ENTRYPOINT	# 指定这个容器启动的时候需要运行的命令，追加
```

测试cmd

```sh
# 编写 dockerfile 文件
zhangjunjie@amax-iset:/data1/zjj/dockerfile$ cat cmdtest
FROM ubuntu

CMD ["ls","-a"]

# 构建镜像
zhangjunjie@amax-iset:/data1/zjj/dockerfile$ docker build -f cmdtest -t cmdtest .

# run运行，发现我们的ls -a生效
zhangjunjie@amax-iset:/data1/zjj/dockerfile$ docker run -it cmdtest:latest
.   .dockerenv  boot  etc   lib    media  opt   root  sbin  sys  usr
..  bin         dev   home  lib64  mnt    proc  run   srv   tmp  var

# 想追加一个命令 -l      ls -al
zhangjunjie@amax-iset:/data1/zjj/dockerfile$ docker run -it cmdtest:latest -l
docker: Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: exec: "-l": executable file not found in $PATH: unknown

# cmd的影响下 -l 替换了CMD ["ls","-a"]命令，-l 不是命令所以报错
```

ENTRYPOINT测试

```shell
zhangjunjie@amax-iset:/data1/zjj/dockerfile$ vim entrypoint_test
FROM    ubuntu

ENTRYPOINT ["ls","-a"]

zhangjunjie@amax-iset:/data1/zjj/dockerfile$ docker build -f entrypoint_test -t entrypoint_test .
zhangjunjie@amax-iset:/data1/zjj/dockerfile$ docker run entrypoint_test:latest
.
..
.dockerenv
home

# 我们追加命令，是直接拼接在 ENTRYPOINT 后面的
# 这里没讲清，在dockerfile中两个都是替换，追加是指在执行run时的追加命令会不会覆盖原有命令
zhangjunjie@amax-iset:/data1/zjj/dockerfile$ docker run entrypoint_test:latest -l
total 56
drwxr-xr-x   1 root root 4096 Feb 27 15:03 .
drwxr-xr-x   1 root root 4096 Feb 27 15:03 ..
-rwxr-xr-x   1 root root    0 Feb 27 15:03 .dockerenv

```

Dockerfile中很多命令都是非常相似，我们需要了解他们的区别，最好的学习就是对比他们，然后测试效果!



| Dockerfile 指令 | 说明                                                         |
| :-------------- | :----------------------------------------------------------- |
| FROM            | 指定基础镜像，用于后续的指令构建。                           |
| MAINTAINER      | 指定Dockerfile的作者/维护者。（已弃用，推荐使用LABEL指令）   |
| LABEL           | 添加镜像的元数据，使用键值对的形式。                         |
| RUN             | 在构建过程中在镜像中执行命令。                               |
| CMD             | 指定容器创建时的默认命令。（可以被覆盖）                     |
| ENTRYPOINT      | 设置容器创建时的主要命令。（不可被覆盖）                     |
| EXPOSE          | 声明容器运行时监听的特定网络端口。                           |
| ENV             | 在容器内部设置环境变量。                                     |
| ADD             | 将文件、目录或远程URL复制到镜像中。                          |
| COPY            | 将文件或目录复制到镜像中。                                   |
| VOLUME          | 为容器创建挂载点或声明卷。                                   |
| WORKDIR         | 设置后续指令的工作目录。                                     |
| USER            | 指定后续指令的用户上下文。                                   |
| ARG             | 定义在构建过程中传递给构建器的变量，可使用 "docker build" 命令设置。 |
| ONBUILD         | 当该镜像被用作另一个构建过程的基础时，添加触发器。           |
| STOPSIGNAL      | 设置发送给容器以退出的系统调用信号。                         |
| HEALTHCHECK     | 定义周期性检查容器健康状态的命令。                           |
| SHELL           | 覆盖Docker中默认的shell，用于RUN、CMD和ENTRYPOINT指令。      |

![从零开始学Docker（三）：DockerFile镜像定制_dockerfile怎么指定镜像-CSDN博客](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250227203710035.png)
