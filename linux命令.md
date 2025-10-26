
### (1)查看当前文件夹大小

```shell
du -sh
```

```sh
(iti) root@ubuntu:~/zjj/honest_llama-master# du -sh
22M     .
```

### (2) 查看硬盘占用情况

```sh
df -h
```

```sh
(iti) root@ubuntu:~/zjj/honest_llama-master# df -h
Filesystem                 Size  Used Avail Use% Mounted on
tmpfs                       23G  1.2M   23G   1% /run
/dev/vda2                  500G  450G   51G  90% /
tmpfs                      114G     0  114G   0% /dev/shm
tmpfs                      5.0M     0  5.0M   0% /run/lock
http://10.0.255.254:4918/  1.3T  763G  509G  61% /webdav
tmpfs                       23G  8.0K   23G   1% /run/user/0
```
