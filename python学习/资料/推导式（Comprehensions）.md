链接： https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Comprehensions.html#nested-comprehensions
## 历史渊源：推导式源自何处？
掌握推导式需要转变思维方式。

一旦理解推导式，是什么让它如此具有吸引力？

推导式是一种可从其他序列构建新序列的语法结构。Python 2.0 引入了列表推导式，而 Python 3.0 则新增了字典推导式和集合推导式。


## 列表推导式（List Comprehensions）
列表推导式由以下几个部分组成：
- 输入序列（Input Sequence）
- 代表输入序列元素的变量（Variable）
- 可选的判断表达式（Predicate expression）
- 输出表达式（Output Expression）：从满足判断条件的输入序列元素中生成输出列表的元素

假设我们需要从一个序列中提取所有整数并计算它们的平方：
```python
a_list = [1, '4', 9, 'a', 0, 4]

squared_ints = [e**2 for e in a_list if type(e) == types.IntType]

print(squared_ints)
# 输出：[1, 81, 0, 16]
```
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251026195748173.png)


- 迭代部分：遍历输入序列 `a_list` 中的每个元素 `e`。
- 判断条件：检查元素是否为整数。
- 若元素是整数，则将其传入输出表达式（计算平方），最终成为输出列表的一员。

通过内置函数 `map`、`filter` 以及匿名函数 `lambda` 也能实现类似功能：
- `filter` 函数对序列应用判断条件：
  ```python
  filter(lambda e: type(e) == types.IntType, a_list)
  ```
- `map` 函数修改序列中的每个元素：
  ```python
  map(lambda e: e**2, a_list)
  ```
- 两者结合使用：
  ```python
  map(lambda e: e**2, filter(lambda e: type(e) == types.IntType, a_list))
  ```

但上述方式存在不足：需要调用 `map`、`filter`、`type` 函数，还需两次调用 `lambda`；Python 中函数调用开销较大，且输入序列会被遍历两次，`filter` 还会生成一个中间列表。

而列表推导式被方括号包裹，能直观体现其最终生成列表的特性。它仅需调用一次 `type` 函数，无需使用晦涩的 `lambda`，而是通过常规迭代器、表达式以及可选的 `if` 判断表达式实现功能。


## 嵌套推导式（Nested Comprehensions）
n 阶单位矩阵（identity matrix）是一个 n×n 的方阵，主对角线上的元素为 1，其余元素均为 0。一个 3 阶单位矩阵如下：

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251026195801419.png)


在 Python 中，可通过“列表的列表”表示该矩阵，其中每个子列表代表矩阵的一行。3 阶单位矩阵的表示形式如下：
```python
[
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]
```

通过以下推导式可生成上述矩阵：
```python
[[1 if item_idx == row_idx else 0 for item_idx in range(0, 3)] for row_idx in range(0, 3)]
```


## 实用技巧（Techniques）
1. 使用 `zip()` 同时处理多个元素：
   ```python
   ['%s=%s' % (n, v) for n, v in zip(self.all_names, self)]
   ```
2. 处理多种类型（元组自动解包）：
   ```python
   [f(v) for (n, f), v in zip(cls.all_slots, values)]
   ```
3. 结合 `os.walk()` 的双层列表推导式：
   ```python
   # Comprehensions/os_walk_comprehension.py
   import os
   restFiles = [os.path.join(d[0], f) for d in os.walk(".")
                for f in d[2] if f.endswith(".rst")]
   for r in restFiles:
       print(r)
   ```


## 更复杂的示例（A More Complex Example）
> 注：以下示例将详细说明各部分功能。

```python
# CodeManager.py
"""
待办：是否需将检查功能拆分为两部分？
待办：update() 方法目前仍处于测试模式，尚未完全可用。

该脚本用于提取、显示、检查和更新 reStructured Text（.rst）文件中的代码示例。

使用方式：只需在 reStructured Text 文件中添加代码标记（codeMarker）和（缩进的）第一行（包含文件路径），
然后运行更新程序，即可自动插入文件的其余内容。
"""
import os, re, sys, shutil, inspect, difflib

# 生成所有 .rst 文件的路径列表（排除含 "_test" 的目录）
restFiles = [os.path.join(d[0], f) for d in os.walk(".") if "_test" not in d[0]
             for f in d[2] if f.endswith(".rst")]

class Languages:
    """策略设计模式（Strategy design pattern）"""
    
    class Python:
        codeMarker = "::\n\n"  # Python 代码标记
        commentTag = "#"       # Python 注释符号
        # 匹配 .rst 文件中 Python 代码块的正则表达式
        listings = re.compile("::\n\n( {4}#.*(?:\n+ {4}.*)*)")
    
    class Java:
        codeMarker = "..  code-block:: java\n\n"  # Java 代码标记
        commentTag = "//"                         # Java 注释符号
        # 匹配 .rst 文件中 Java 代码块的正则表达式
        listings = re.compile(".. *code-block:: *java\n\n( {4}//.*(?:\n+ {4}.*)*)")

def shift(listing):
    """将代码列表向左缩进 4 个空格（去除每行开头的 4 个空格）"""
    return [x[4:] if x.startswith("    ") else x for x in listing.splitlines()]

# 测试：在测试目录中复制 .rst 文件，用于测试 update() 方法
# 生成测试目录路径集合
dirs = set([os.path.join("_test", os.path.dirname(f)) for f in restFiles])
# 创建不存在的测试目录
if [os.makedirs(d) for d in dirs if not os.path.exists(d)]:
    # 将 .rst 文件复制到测试目录
    [shutil.copy(f, os.path.join("_test", f)) for f in restFiles]
# 生成测试目录中所有 .rst 文件的路径列表
testFiles = [os.path.join(d[0], f) for d in os.walk("_test")
             for f in d[2] if f.endswith(".rst")]

class Commands:
    """
    每个静态方法均可从命令行调用。如需添加新命令，只需在此处新增静态方法。
    """
    
    @staticmethod
    def display(language):
        """打印所有 .rst 文件中的代码示例"""
        for f in restFiles:
            # 读取文件内容并匹配所有代码块
            listings = language.listings.findall(open(f).read())
            if not listings:  # 无代码块则跳过
                continue
            # 打印文件分隔符和文件名
            print('=' * 60 + "\n" + f + "\n" + '=' * 60)
            # 遍历并打印每个代码块
            for n, l in enumerate(listings):
                print("\n".join(shift(l)))
                # 非最后一个代码块则打印分隔线
                if n < len(listings) - 1:
                    print('-' * 60)
    
    @staticmethod
    def extract(language):
        """
        从 .rst 文件中提取代码示例，并将每个示例写入单独的文件。
        若代码文件与 .rst 文件内容不一致，默认不覆盖；如需强制覆盖，需使用 "extract -force" 命令。
        """
        # 判断是否使用强制覆盖模式
        force = len(sys.argv) == 3 and sys.argv[2] == '-force'
        paths = set()  # 用于存储已处理的文件路径，避免重复
        
        # 提取所有 .rst 文件中的代码块并去除缩进
        for listing in [shift(listing) for f in restFiles
                        for listing in language.listings.findall(open(f).read())]:
            # 从代码块第一行提取文件路径（去除注释符号）
            path = listing[0][len(language.commentTag):].strip()
            # 检查文件路径是否重复
            if path in paths:
                print("错误：文件名重复：%s" % path)
                sys.exit(1)
            else:
                paths.add(path)
            
            # 构建目标文件的完整路径
            path = os.path.join("..", "code", path)
            dirname = os.path.dirname(path)
            # 若目标目录不存在则创建
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            
            # 若文件已存在且非强制模式，检查内容是否一致
            if os.path.exists(path) and not force:
                # 比较现有文件与提取的代码块
                for i in difflib.ndiff(open(path).read().splitlines(), listing):
                    if i.startswith("+ ") or i.startswith("- "):
                        print("错误：现有文件与 .rst 文件内容不一致")
                        print("如需强制覆盖，请使用 'extract -force' 命令")
                        Commands.check(language)
                        return
            
            # 将提取的代码块写入目标文件
            with open(path, 'w') as f:
                f.write("\n".join(listing))
    
    @staticmethod
    def check(language):
        """
        确保外部代码文件存在，并检查哪些外部文件与 .rst 文件中的代码示例不一致。
        在 _deltas 子目录中生成 HTML 文件，展示具体的差异内容。
        """
        # 内部辅助类：用于存储检查结果
        class Result:
            def __init__(self, **kwargs):
                self.__dict__ = kwargs
        
        result = Result(missing=[], deltas=[])  # 初始化结果：缺失文件列表和差异文件列表
        
        # 提取所有 .rst 文件中的代码块及对应文件路径
        listings = [Result(code=shift(code), file=f)
                    for f in restFiles for code in
                    language.listings.findall(open(f).read())]
        # 生成外部代码文件的完整路径
        paths = [os.path.normpath(os.path.join("..", "code", path)) for path in
                 [listing.code[0].strip()[len(language.commentTag):].strip()
                  for listing in listings]]
        
        # 若 _deltas 目录已存在，先删除该目录
        if os.path.exists("_deltas"):
            shutil.rmtree("_deltas")
        
        # 遍历每个代码块及其对应的外部文件路径
        for path, listing in zip(paths, listings):
            if not os.path.exists(path):
                # 外部文件缺失，添加到缺失列表
                result.missing.append(path)
            else:
                # 读取外部文件内容
                code = open(path).read().splitlines()
                # 比较代码块与外部文件内容
                for i in difflib.ndiff(listing.code, code):
                    if i.startswith("+ ") or i.startswith("- "):
                        # 存在差异，生成 HTML 差异报告
                        d = difflib.HtmlDiff()
                        if not os.path.exists("_deltas"):
                            os.makedirs("_deltas")
                        # 构建 HTML 文件名
                        html_filename = os.path.basename(path).split('.')[0] + ".html"
                        html_path = os.path.join("_deltas", html_filename)
                        # 写入 HTML 差异报告
                        with open(html_path, 'w') as f:
                            f.write(
                                "<html><h1>左侧：%s<br>右侧：%s</h1>" % (listing.file, path) +
                                d.make_file(listing.code, code)
                            )
                        # 将差异信息添加到结果列表
                        result.deltas.append(Result(
                            file=listing.file,
                            path=path,
                            html=html_path,
                            code=code
                        ))
                        break  # 找到一个差异即可，无需继续比较
        
        # 打印缺失文件信息
        if result.missing:
            print("缺失 %s 文件：\n%s" % (language.__name__, "\n".join(result.missing)))
        # 打印差异文件信息
        for delta in result.deltas:
            print("%s 中的代码与 %s 不一致，详情请查看 %s" % (delta.file, delta.path, delta.html))
        
        return result
    
    @staticmethod
    def update(language):  # 处于测试阶段，待验证可靠性
        """将外部代码文件的最新内容更新到 .rst 文件中"""
        # 先执行检查，确保无缺失文件
        check_result = Commands.check(language)
        if check_result.missing:
            print(language.__name__ + " 更新中止")
            return
        
        changed = False  # 标记是否有文件被更新
        
        # 内部辅助函数：用于替换 .rst 文件中的代码块
        def _update(matchobj):
            # 提取并处理代码块（去除缩进）
            listing = shift(matchobj.group(1))
            # 从代码块第一行提取文件路径
            path = listing[0].strip()[len(language.commentTag):].strip()
            # 构建外部代码文件的完整路径
            path = os.path.join("..", "code", path)
            # 读取外部代码文件的最新内容
            code = open(path).read().splitlines()
            # 重新构建带缩进的代码块，并添加代码标记
            indented_code = ["    " + line for line in listing]
            return language.codeMarker + "\n".join(indented_code).rstrip()
        
        # 遍历所有测试目录中的 .rst 文件，更新代码块
        for f in testFiles:
            # 读取文件内容并替换代码块
            updated_content = language.listings.sub(_update, open(f).read())
            # 将更新后的内容写回文件
            with open(f, 'w') as f_out:
                f_out.write(updated_content)

if __name__ == "__main__":
    # 获取 Commands 类中的所有静态方法（命令）
    commands = dict(inspect.getmembers(Commands, inspect.isfunction))
    # 检查命令行参数是否合法
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print("命令行选项：\n")
        # 打印所有可用命令及其说明
        for name in commands:
            print(name + ": " + commands[name].__doc__)
    else:
        # 执行指定命令（对每种语言均执行）
        for language in inspect.getmembers(Languages, inspect.isclass):
            commands[sys.argv[1]](language[1])
```


## 集合推导式（Set Comprehensions）
集合推导式的原理与列表推导式一致，唯一区别是最终生成的序列为集合（set）。

假设我们有一个姓名列表，其中包含大小写不同的重复姓名，以及仅含一个字符的姓名。我们仅关注长度大于 1 的姓名，并希望统一姓名格式：首字母大写，其余字母小写。

给定姓名列表：
```python
names = ['Bob', 'JOHN', 'alice', 'bob', 'ALICE', 'J', 'Bob']
```

目标集合：
```python
{'Bob', 'John', 'Alice'}
```

> 注：集合的语法为用大括号包裹元素，且集合中无重复元素。

通过以下集合推导式可实现上述需求：
```python
{name[0].upper() + name[1:].lower() for name in names if len(name) > 1}
```


## 字典推导式（Dictionary Comprehensions）
假设我们有一个字典，键为字符，值为该字符在某段文本中出现的次数。当前字典会区分大小写字符，我们需要将大小写字符的出现次数合并，生成一个新字典。

原始字典：
```python
mcase = {'a': 10, 'b': 34, 'A': 7, 'Z': 3}
```

通过以下字典推导式合并大小写字符的出现次数：
```python
mcase_frequency = {k.lower(): mcase.get(k.lower(), 0) + mcase.get(k.upper(), 0) for k in mcase.keys()}

# mcase_frequency 的结果为：{'a': 17, 'z': 3, 'b': 34}
```