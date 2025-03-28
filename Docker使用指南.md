# Docker 使用指南

> **创建人:** 王聪 (DG21330033), 2024-08-28

## 为什么使用 Docker

为了实现在不同的机器上快速配资私有的开发环境, 包括:

- 系统版本,
- CUDA 和 cudnn 版本,
- conda 及 python 库,
- tmux, vim, ssh 等软件的安装及配置.

## 安装 Docker

> 现有的服务器都支持直接使用 Docker, 本章暂时空缺.

## 容器 (Image) 与镜像 (Container)

**容器 (Image)** 和 **镜像 (Container)** 是 Docker 中最重要的两个概念. 容器是静态的, 其中保存着环境所需的静态文件. 当我们有程序 (不单单指 python 程序, 而是所有的广义程序, 包括 SSH, bash 等) 需要调用这个环境时, 就需要基于这个镜像创建一个容器, 在这个创建的容器中运行我们的目标程序.

我们可以把**镜像**理解为编程语言中的**类**, 把**容器**理解为基于类创建的**对象**. 也就是说, 对于同一个镜像, 我们可以创建多个容器.

## 快速使用

### 下载基础镜像

我们在云盘中提供了一个[基础镜像文件](https://box.nju.edu.cn/smart-link/9eac60ae-c5df-41b3-ab84-c36ca84bcd40/), 下载之后上传到服务器.

> 简单介绍一下这个基础镜像的来源及额外的配置:
>
> 1. 来源于 PyTorch 的官方镜像 (版本: [pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel](https://hub.docker.com/layers/pytorch/pytorch/2.3.1-cuda12.1-cudnn8-devel/images/sha256-a22a1fca37f8361c8a1e859cd6eb6bd9d1fb384f9c0dcb2cfc691a178eb03d17?context=explore), 表示镜像中的 PyTorch 版本, CUDA 版本, cudnn 版本分别是 2.3.1, 12.1, 8. 如果该设置不符合您的需求, 请参考后续大章定制自己的镜像);
> 2. 安装了 vim, tmux, openssh-server 和 systemctl;
> 3. 设置 SSH 的端口为 22, 并允许以 root 用户访问;
> 4. 配置 tmux 允许鼠标操作;
> 5. 设置 root 用户密码为 `iset666`.

### 加载基础镜像

进入基础镜像所在的目录, 执行命令加载镜像:

```bash
docker load -i iset-pytorch-2.3.1-cuda12.1-cudnn8-devel.tar
```

使用以下命令查看镜像状态:

```bash
docker images | grep iset
# iset/pytorch  2.3.1-cuda12.1-cudnn8-devel  0a212aa0e703  4 hours ago  17.3GB
```

> 解释一下镜像状态的含义:
>
> - **iset/pytorch:** 镜像仓库名;
> - **2.3.1-cuda12.1-cudnn8-devel:** 镜像 tag;
> - **0a212aa0e703**: 镜像 ID, 在操作镜像是可以用 ID 指定镜像, 代替 iset/pytorch:2.3.1-cuda12.1-cudnn8-devel;
> - **4 hours ago**: 创建时间, 数值取决于现在距离该镜像被创建时的时间差;
> - **17.3 GB**: 镜像大小.

### 启动容器

使用以下命令启动一个私有容器:

```bash
docker run \
  --gpus all \ # 在容器中可使用的物理机 GPU
  --shm-size [shm-size] \ # 设置共享内存大小
  -d \ # 容器后台运行
  -p [port]:22 \ # 端口映射
  -v [path-before-mapping]:[path-after-mapping] \ # 存储空间映射
  --name [name] \ # 容器名称
  iset/pytorch:2.3.1-cuda12.1-cudnn8-devel \ # 镜像名称
  /usr/sbin/sshd -D # 使用容器环境的是 SSH 服务
```

解释一下该命令的含义:

- `--gpus`: 容器允许使用的物理机 GPU, 一般设置为 `all` 即可, 表示容器可以使用物理机的全部 GPU;
- `--shm-size`: 容器允许使用的共享内存大小, 用于在 PyTorch 的 DataLoader 中的多线程数据加载, 共享内存越大, 可加载数据的线程数 (num_workers 参数) 越多, 一般可以设置为 `32g`;
- `-d`: 容器被挂起 (detached) 在后台运行, 目的是启动容器之后可以直接关闭命令行, 容器并不会停止;
- `-p`: 设置端口映射, 比如: `-p 8878:22` 表示容器的 22 端口 (即 SSH 端口) 被映射到物理机的 8878 端口, 这样访问物理机的 8878 端口就相当于连接了容器的 SSH 服务; 此外, 也可以设置多个端口映射, 例如: `-p 8878:22 -p 8879:6006 -p 8880:8888` 表示容器的 22, 6006, 和 8888 端口分别被映射到了物理机的 8878, 8879, 和 8880 端口;
- `-v`: 设置存储空间映射, 比如 `-v /root/cw/workspace:/workspace` 表示容器下的路径 `/workspace` 被映射到物理机的 `/root/cw/workspace`, 即它们共享同一块存储空间, 读写物理机的 `/root/cw/workspace` 等同于读写容器的 `/workspace`, 同样, 读写容器的 `/workspace` 等同于读写物理机的 `/root/cw/workspace`; 这么做的好处是, 你在容器内会用到的大文件 (包括: 数据集, 模型等) 不会被一起打包到镜像里, 从而方便在不同机器之间迁移镜像;
- `--name`: 设置镜像的名称, 一般设置自己的名字即可, 例如: `cw`;
- `iset/pytorch:2.3.1-cuda12.1-cudnn8-devel`: 这个指的是启动容器所需镜像的名称;
- `/usr/sbin/sshd -D`: 这个指的是调用镜像环境的程序是 SSH 服务; 这样做的好处是整个容器都可以被看作一台虚拟化的机器, 可以直接通过 SSH 服务访问, 而不需要通过物理机做中转.

> 这里放一个我在某台机器上启动私有容器的命令供大家参考:
>
> ```bash
> docker run \
>   --gpus all \
>   --shm-size 32g \ # 共享内存的大小是 32G
>   -d \
>   -p 8878:22 -p 8879:8888 -p 8880:6006 \ # 三个端口映射, 分别对应 SSH, Jupyter, 和 TensorBoard
>   -v /nvme1n1/cw/cw_workspace:/workspace \ # 存储空间映射, 我会把任意一台物理机的个人目录映射到容器的 /workspace 下, 这样在不同机器上的容器内, 数据根路径都是一致的
>   --name cw # 容器名称
>   cw/pytorch:2.3.1-cuda12.1-cudnn8-devel \ # 镜像名称, 这里用了一个我自己定制的镜像
>   /usr/sbin/sshd -D
> ```

启动完成后, 通过命令就可以看到你的镜像状态为 `Up`, 这表示你的容器启动成功了:

```bash
docker ps | grep [name]
```

如果 `docker ps` 命令的返回结果中没有你的容器, 那么可能是启动容器时出错, 容器退出了, 你可以通过 `docker ps -a` 找到你的容器信息.

> 在上面的那个我启动私有容器的例子中, 我输入 `docker ps | grep cw` 命令, 会返回:
>
> ```bash
> e0fc3c3ac56d   cw/pytorch:2.3.1-cuda12.1-cudnn8-devel   "/opt/nvidia/nvidia_…"   5 days ago   Up 2 days   0.0.0.0:8878->22/tcp, :::8878->22/tcp, 0.0.0.0:8880->6006/tcp, :::8880->6006/tcp, 0.0.0.0:8879->8888/tcp, :::8879->8888/tcp   cw
> ```
>
> 其中 `Up 2 days` 表示我的镜像已经运行 2 天了.

### 连接容器

成功启动容器后, 就可以使用 SSH 远程访问容器. SSH 的用户是 `root`, 密码是 `iset666`. SSH 的 IP 和端口需要根据物理机的网络结构选择:

- 如果物理机的网络是可直连的, 那么访问容器的 SSH 只需要使用物理机的 IP 和启动容器时 22 端口映射的端口 `[port]` (比如在上面给出的例子中, 我需要连接物理机的 `8878` 端口来访问容器的 SSH) 即可;
- 如果容器的物理机是有跳板机结构的 (SSH 端口不是 22 的物理机一般都是有跳板机结构), 则需要在本地设置**端口转发**才能连接容器的 SSH.

> **如何设置端口转发?**
>
> 在本地的命令行中运行命令:
>
> ```bash
> ssh -N -L [local-port]:localhost:[container-port] -p [machine-port] root@[machine-ip]
> ```
>
> - [machine-port] 和 [machine-ip] 是物理机的端口和 IP;
> - [container-port] 是启动容器时 22 端口映射的端口 `[port]`;
> - [local-port] 是转发到本地的端口, 可以随意设置;
>
> 这样, 访问容器 SSH 的 IP 和 端口为 `localhost` 和 `[local-port]`.
>
> 举个例子, 这是我设置某台有跳板机结构的物理机的端口转发命令:
>
> ```bash
> ssh -N -L 8881:localhost:8878 -L 8882:localhost:8879 -L 8883:localhost:8880  -p 20067 root@10.60.200.2 # 启动容器时 22 端口映射的端口是 8878
> ```
>
> 这样我只需要访问 `localhost` 的 `8881` 即可连接容器 SSH. 同时可以看到端口转发命令是支持一次转发多个端口的. 如果在容器中开启的多个端口服务, 可以通过同时配置多个端口转发来访问这些服务.

### 配置私有环境

通过 SSH 进入容器内部后, 即可配置你自己的环境, 包括 `conda install`, `pip install`, `apt install` 等软件和库的安装, 系统和软件的配置. 不建议将**大文件**写入容器环境内, 这会是的后续打包的镜像大小过大, 建议将大文件放入挂载的存储空间内.

### 打包私有容器为镜像

在私有环境配置完成后, 需要在在其他机器上使用高环境时, 需要进入容器所在的物理机:

```bash
docker stop [name]
docker commit [name] [name]/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

这样, 你的私有容器环境就被保存成了一个镜像. 使用:

```bash
docker save -o [name]-pytorch-2.3.1-cuda12.1-cudnn8-devel.tar [name]/pytorch:2.3.1-cuda12.1-cudnn8-devel
```

可以将镜像打包成一个 tar 包, 将这个 tar 包. 在其他的机器上, 使用:

```bash
docker load -i [name]-pytorch-2.3.1-cuda12.1-cudnn8-devel.tar
```

即可加载私有镜像.



启动容器

```shell
docker run --gpus all --shm-size=32g -d -v /data1/zjj/workspace:/workspace -p 7877:22 -p 7878:80 -p 7879:8888 -p 7880:6006 -p 7881:11434 -p 7882:8000 -p 7883:5000 -p 7884:7860 -p 7885:8501  --name zjj_env1 pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime sleep infinity
```

安装常用包

```shell
# 1. 更新包
	# update 命令只会获得系统上所有包的最新信息，并不会下载或者安装任何一个包。而是 apt upgrade 命令来把这些包下载和升级到最新版本。
apt update		# 更新包缓存（可以知道包的哪些版本可以被安装或升级）
apt upgrade		# 升级包到最新版本

# 2. vim,curl
apt install -y vim
apt install -y curl
# 3. thefuck
#Ubuntu/Debian系统
sudo apt update
sudo apt install python3-dev python3-pip
sudo pip3 install thefuck # 最好进入conda环境安装

# 配置 PATH
export PATH="/root/.local/bin:$PATH"
echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
# 每次登录可用
echo 'eval "$(thefuck --alias)"' >> ~/.bashrc
# 生效
source ~/.bashrc
```

![img](https://gitee.com/zhang-junjie123/picture/raw/master/image/20250228160804176.png)





