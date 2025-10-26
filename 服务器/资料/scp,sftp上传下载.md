# 通过sftp/scp/rsync向Linux实例传输文件

更新时间：2025-09-08 10:16:20

[产品详情](https://www.aliyun.com/product/ecs)

[我的收藏](https://help.aliyun.com/my_favorites.html)

当需要从本地 Linux 或 macOS 系统向云服务器 ECS的 Linux 实例传输文件时，scp, sftp,rsync 是三种常用的命令行工具。选择和正确使用合适的工具，对于提升文件传输的效率、可靠性与安全性至关重要。本文旨在提供这三种工具的场景化使用指南和最佳实践，帮助用户应对从简单文件上传到大规模数据同步的需求。

## **工具介绍对比**

在开始操作前，可根据具体场景选择最合适的工具。

- **临时传输单个小文件或配置**：推荐使用 `scp`，语法最简单，上手最快。
    
- **需要交互式管理文件（如查看目录、删除、重命名）**：推荐使用 `sftp`，它提供一个类似 FTP 的交互式会话。
    
- **大规模数据同步、定期备份、传输大量小文件**：推荐使用 `rsync`，其增量同步算法效率最高。
    
- **网络环境不稳定，传输大文件**：推荐使用`rsync` 或 `sftp` 的断点续传功能。
    

> 下表提供了更详细的技术对比：

`scp/ sftp/ rsync`工具详细对比

## **操作步骤**

### 准备工作与前置检查

在执行文件传输前，请完成以下准备与检查工作。

1. **获取实例的公网 IP 地址** 本地向实例传输文件时，实例需[开通公网](https://help.aliyun.com/zh/ecs/user-guide/best-practices-for-configuring-public-bandwidth)。开通后可在 ECS 实例列表中记录目标实例的公网 IP 地址。后续所有命令都将使用此 IP。
    
2. **配置安全组规则** 文件传输依赖 SSH 协议，默认使用 22 端口。需确保实例所属的安全组已放行来自本地网络的访问请求。
    
    - **授权策略**：允许
        
    - **协议类型**：自定义 TCP
        
    - **端口范围**：`22/22` (或您的自定义 SSH 端口)
        
    - **授权对象**：为了安全，建议仅填写您的本地公网 IP 地址。
        
        > 可通过在本地终端执行 `curl ifconfig.me` 或 `curl ip.sb` 获取。
        
3. **检查实例内部防火墙** 除了安全组，实例操作系统内部的防火墙可能阻止连接。
    
    1. 登录实例，[查看防火墙状态](https://help.aliyun.com/zh/ecs/use-cases/enable-or-disable-the-system-firewall-in-a-linux-instance#6e0645c393ibn)。
        
    2. 若防火墙开启，参考[开放指定端口或服务](https://help.aliyun.com/zh/ecs/use-cases/enable-or-disable-the-system-firewall-in-a-linux-instance#ba5c6c3585tll)，需确保其已放行 SSH 服务或 22 端口。
        

## 方法一：使用 scp 传输文件

SCP（Secure Copy Protocol）用于在本地和远程主机之间进行简单的文件或目录复制。

### **上传文件或目录到实例**

需要上传文件或文件夹到实例时，可以在本地执行以下命令，执行命令后，会提示您输入密码。

```shell
# 上传单个文件到实例
sudo scp <本地文件路径> <云服务器登录名>@<云服务器公网IP地址>:<云服务器文件目录>

# 上传本地目录到实例
sudo scp -r <本地目录> <云服务器登录名>@<云服务器公网IP地址>:<云服务器文件目录>
```

#### **示例：**

将本地的`/opt/test.txt`文件上传到公网IP为1xx.xxx.xxx.121实例的`/home/ecs-user/`路径下，可通过以下命令实现：

```shell
sudo scp /opt/test.txt ecs-user@1xx.xxx.xxx.121:/home/ecs-user/
```

### **从实例下载文件或目录到本地**

需要从实例下载文件到本地时，可以在**本地**执行以下命令，执行命令后，会提示您输入密码。

> 当传输大量小文件时，`scp` 效率较低。建议先将文件打包成单个压缩文件（如 `.tar.gz`）再传输，或改用 `rsync`。

```shell
# 下载单个文件到本地
sudo scp <云服务器登录名>@<云服务器公网IP地址>:<云服务器文件路径> <本地目录>

# 下载实例目录到本地
sudo scp -r <云服务器登录名>@<云服务器公网IP地址>:<云服务器文件目录> <本地目录>
```

#### **示例：**

将公网IP为1xx.xxx.xxx.121实例的`/home/ecs-user/test.txt`文件下载到本地的`/opt/`路径下，可通过以下命令实现：

```shell
sudo scp ecs-user@1xx.xxx.xxx.121:/home/ecs-user/test.txt /opt/
```

## 方法二：使用 sftp 进行交互式文件传输

SFTP（SSH File Transfer Protocol）提供一个交互式会话，允许在传输文件的同时进行远程文件管理。

### **连接到实例**

在本地终端执行以下命令，建立 `sftp` 连接。连接成功后，终端提示符会变为 `sftp>`。

```shell
sudo sftp <云服务器实例登录名>@<云服务器实例公网IP地址>
```

SFTP常用交互命令

### **上传文件或整个目录到实例**

```shell
# 上传单个文件
sftp> put <本地文件路径> <云服务器文件目录>

# 上传整个目录
sftp> put -r <本地目录> <云服务器文件目录>
```

#### **示例：**

- 将本机`/opt/test.txt`文件上传至实例的`/home/ecs-user/`目录下：
    
    ```shell
    sftp> put /opt/test.txt /home/ecs-user
    ```
    
- 将本机`/opt/test/`目录上传至实例的`/home/ecs-user/`目录下：
    
    ```shell
    sftp> put -r /opt/test/ /home/ecs-user/
    ```
    

### **从实例下载文件或整个目录到本地**

```shell
# 下载单个文件
sftp> get <云服务器文件路径> <本地目录>

# 下载整个目录
sftp> get -r <云服务器文件目录> <本地目录>
```

#### **示例：**

- 将实例的`/home/ecs-user/test.txt`文件下载至本机的`/opt`目录下：
    
    ```shell
    sftp> get /home/ecs-user/test.txt /opt
    ```
    
- 将实例的`/home/ecs-user/test/`目录下载至本地的`/opt`目录下：
    
    ```shell
    sftp> get -r /home/ecs-user/test/ /opt
    ```
    

### **断点续传**

当大文件传输中断时，可使用 `reget`(上传) 和 `reput`(下载) 命令继续传输。

```shell
# 继续上传
sftp> reput <本地文件路径> <云服务器文件目录>

# 继续下载
sftp> reget <云服务器文件路径> <本地目录>
```

> SFTP断点续传仅对单个文件有效。若传输整个目录时中断，建议退出后使用rsync进行高效同步。

### **断开连接**

当完成传输任务后，可以通过`quit`或`bye`命令退出 `sftp` 会话。

## 方法三：使用 rsync 高效同步文件和目录

rsync适用于大规模、增量或重复性的传输任务。

### 安装工具

确保本地和 ECS 实例上都已安装 `rsync`。

```shell
# CentOS / Alibaba Cloud Linux
sudo yum install -y rsync

# Debian / Ubuntu
sudo apt-get update && sudo apt-get install -y rsync
```

### **常用参数**

`rsync` 的标准用法通常包含 `-avz` 参数：

- `-a` (archive): 归档模式，等同于 `-rlptgoD`，递归同步并保持文件所有属性（如权限、时间戳）。
    
- `-v` (verbose): 显示详细的传输过程。
    
- `-z` (compress): 在传输过程中压缩数据，可节省带宽。但在高带宽链路上，CPU 压缩可能成为瓶颈，不加 `-z` 反而更快。
    

#### **生产环境常用参数**

### **上传/同步文件或目录到ECS实例**

当需要上传文件到实例时，可以在**本地**执行以下命令，执行命令后，会提示您输入密码。

```shell
sudo rsync -avz -e ssh <本地文件或文件夹路径> <云服务器实例登录名>@<云服务器实例公网IP地址>:<实例目录>
```

#### **示例：**

- 将`/opt/test.txt`文件上传至公网IP为1xx.xxx.xxx.121实例的`/home/ecs-user`目录下，可使用以下命令：
    
    ```shell
    sudo rsync -avz -e ssh /opt/test.txt ecs-user@1xx.xxx.xxx.121:/home/ecs-user
    ```
    
- 将本地`/opt/test`目录，与公网IP为1xx.xxx.xxx.121实例的`/home/ecs-user/test`目录同步，可使用以下命令：
    
    ```shell
    sudo rsync -avz -e ssh /opt/test/ ecs-user@1xx.xxx.xxx.121:/home/ecs-user/test
    ```
    

### **从实例下载/同步文件或目录到本地**

当需要从实例下载文件到本地时，可以在本地执行以下命令，执行命令后，会提示您输入密码。

```shell
sudo rsync -avz -e ssh <云服务器实例登录名>@<云服务器实例公网IP地址>:<实例文件或文件夹路径> <本地目录>
```

## 应用于生产环境

- **传输海量小文件：先打包，再传输** 对于包含数万甚至数百万小文件的目录，直接使用 `scp` 或 `rsync` 会因大量的连接和元数据开销而变得缓慢。可以先在源端打包压缩，传输单个大文件，然后在目标端解压。
    
- **使用 SSH 配置文件简化命令** 在本地 `~/.ssh/config` 文件中为 ECS 实例设置别名（以`my-prod-server`为例），可以简化连接和传输命令。
    
    ```ini
    # 添加以下内容到 ~/.ssh/config
    Host my-prod-server
        HostName 118.178.x.x
        User ecs-user
        Port 22
        IdentityFile ~/.ssh/id_rsa_aliyun
        ServerAliveInterval 60
    ```
    
    配置后，命令将变得简洁，示例如下：
    
    ```shell
    # 原命令: 
    sudo scp -i ~/.ssh/id_rsa_aliyun local.txt ecs-user@118.178.x.x:/remote/
    # 现命令: 
    sudo scp local.txt my-prod-server:/remote/
    
    # 原命令: 
    sudo rsync -avz -e "ssh -i ~/.ssh/id_rsa_aliyun" local_dir/ ecs-user@...
    # 现命令: 
    sudo rsync -avz local_dir/ my-prod-server:/remote_dir/
    ```
    

## 常见问题

- **如何通过指定端口传输文件？**
    
    - **SCP：**通过`-P`参数指定端口，`scp -P <端口> <具体命令>`。
        
    - **SFTP：**通过`-P`参数指定端口，`sftp -P <端口> <具体命令>`。
        
    - **Rsync：**通过修改`-e`参数来指定端口，`rsync -avz -e "ssh -p <SSH服务的端口>" <具体命令>` 。
        
- #### **如何在通过密钥对连接实例传输文件时指定私钥？**
    
    - **SCP：**通过`-i`参数指定私钥文件，`scp -i <私钥文件路径> <本地文件路径> <具体命令>`。
        
    - **SFTP：**通过`-oIdentityFile`参数指定私钥文件，`sftp -oIdentityFile=<私钥文件路径> <具体命令>`。
        
    - **Rsync：**通过修改`-e`参数来指定端口，`rsync -e "ssh -i <私钥文件路径>" <具体命令>`。
        

## **相关文档**

- 若需上传至ECS的重要文件进行备份，请参见[创建快照](https://help.aliyun.com/zh/ecs/user-guide/create-a-snapshot)。
    
- 若需要上传文件到Windows实例，可使用其他文件传输方式，参见[选择传输文件的方式](https://help.aliyun.com/zh/ecs/user-guide/choose-how-to-transfer-files)。
    
- 若需从本地为Windows操作系统，需要向Linux实例传输文件，可以使用WinSCP工具完成文件传输操作，可参见[在本地Windows使用WinSCP向Linux实例传输文件](https://help.aliyun.com/zh/ecs/user-guide/use-winscp-to-upload-a-file-to-a-linux-instance)。