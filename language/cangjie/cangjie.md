
### 一、hello 仓颉
```c
main() {
println("你好，仓颉")
}

> cjc hello.cj -o hello
> ./hello
你好，仓颉
```

**项目常用命令：**

| 命令                           | 说明                            |
| ---------------------------- | ----------------------------- |
| `cjpm init`                  | 初始化一个新的仓颉项目，生成`cjpm.toml`配置文件 |
| `cjpm add <package-name>`    | 安装指定包并添加到依赖（如`cjpm add http`） |
| `cjpm install`               | 根据`cjpm.toml`安装所有依赖           |
| `cjpm update`                | 更新已安装的包到最新兼容版本                |
| `cjpm remove <package-name>` | 移除指定依赖包                       |
| `cjpm list`                  | 列出当前项目已安装的包                   |
| `cjpm build`                 | 构建项目（部分版本可能仍使用                |

### 二、变量


**可变变量**： var name: type = expr
**不可变变量**：let name: type = expr
**常量**：const name: type = exprconst
当初始值具有明确类型时，可以省略变量类型标注，编译器会自动推断出变量类型。

|关键字|含义|是否可重新赋值|
|---|---|---|
|`let`|声明不可变变量|❌ 不可|
|`var`|声明可变变量|✅ 可以|

示例
```c
let a = 5
var b = 10

b = 20   // ✅ OK
// a = 6 // ❌ 编译错误
```

### 三、类型

![image.png|500](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111111350776.png)


相同的数据，赋予不同的类型/协议，解析和操作结果并不相同。

#### 1、基础数据类型
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111111453077.png)

| 语法      | 含义     | 是否包含右端点  |
| ------- | ------ | -------- |
| `a..b`  | 左闭右开区间 | ❌ 不包含`b` |
| `a..=b` | 左闭右闭区间 | ✅ 包含`b`  |

#### 2、表达式

##### (1) 管道操作符
let result = data |> fn1 |> fn2 |> fn3
> 使用了 **管道操作符（Pipe Operator）`|>`**，这是一种常见的函数式编程语法，在仓颉语言（Cangjie）、Elixir、F#、OCaml、以及 JavaScript 的提案中都有类似设计。

等价于  let result = fn3(fn2(fn1(data)))

##### (2) case
let color = Color.Red(100)

```c
let result = match (color) {
    case Red(value)   => value + 10          // value = 100，结果 110
    case Green(text)  => text.length()
    case _            => 0
}
// result == 110
```
- `match (color)`：对变量 `color` 进行模式匹配。
- 每个 `case` 尝试将 `color` 与某种**构造器**（如 `Red`、`Green`）匹配。
- 如果匹配成功，会**解构**（extract）出其中的值（如 `value`），并在 `=>` 右侧的代码块中使用。
- `case _` 是**通配符**，匹配所有未被前面 case 覆盖的情况（类似 `else` 或 `default`）。

##### (3)if 表达式

如果exprBool 取值为true，将执行if 分支，反之执行else 分支。如果执行了某个分支或没有可选分
支，都会跳出if 表达式并执行后续代码。
如果if 表达式**具有else 代码块**，==则if 表达式的值就等于所执行代码块最后一个表达式的值。其他情况的if 表达式类型为Unit。==
![image.png|400](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111112600470.png)
##### (4)while 表达式
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111112803177.png)
##### (5) for-in 表达式
遍历对象的类型**需要实现迭代器接口Iterable\<T\>**，运行时，将逐次调用迭代器取值并执行循环体,在循环体中可以通过循环变量引用对应值。

规定for-in 表达式的类型是Unit。
![image.png|600](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111112933936.png) 

示例
```c
// 可使用where 引导一个Bool 表达式，取值为true 才会执行循环体。
for (i in 0..10 where i % 2 == 1) {
println(i)
}
```

### 四、函数

#### 1、定义函数

参数主要分为两种基本类型：**普通参数**​ 和 **命名参数**。第三种是前两者的组合。

函数类型的表达方式() -> type

函数不仅可以被调用，还可以作为值去使用，如赋值给变量、作为函数的参数和返回值等。
##### 1. 普通参数
- 
    **代码表示**： `params_normal := name: type, name: type`
- 
    **解释**：
    - 这是最常见的参数形式，也称为**位置参数**。
    - 在调用函数或方法时，传入的实参**必须严格按照形参定义的顺序**进行匹配。
    - 例如，如果定义为 `greet(name: String, age: Int)`，调用时必须写 `greet("Alice", 25)`，而不能打乱顺序。
        
##### 2. 命名参数

- **代码表示**： `params_named := name!: type, name!: type`
    
- **解释**：
    - 这种参数通过名字来传递，而不是位置。在定义时，参数名后面带有一个感叹号 `!`作为标识。
    - 在调用时，可以显式地指定参数名，从而**允许打乱传递顺序**，提高了代码的可读性和灵活性。
    - 例如，如果定义为 `connect(timeout!: Int, url!: String)`，调用时可以写 `connect(url: "example.com", timeout: 30)`。

##### 3. 综合参数

- **代码表示**： `params := params_normal ?, params_named ?`
    
- **解释**：
	- 在实际的函数定义中，可以同时包含普通参数和命名参数。
    - 这里的问号 `?`表示“零个或一个”，即普通参数部分和命名参数部分都是可选的。一个合法的参数列表可以是：只有普通参数、只有命名参数，或者两者都有。

##### 4、参数顺序

> **命名参数只能写在非命名参数之后**

这意味着在一个参数列表中：**必须先列出所有的普通参数（位置参数）**，**然后再列出命名参数**。

#### 2、调用函数
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111115949466.png)

#### 3、lambda 表达式

lambda 表达式可以让函数的创建和使用更加灵活， lambda 表达式的值就是一个匿名函数。
lambda 表达式中**无须标注返回值类型，仓颉编译器会从上下文中自动推导。**
{ params => blockfunc }

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111120226865.png)

## 五、枚举

#### 1、定义与实例化

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111120425961.png)

#### 2、成员访问规则
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111120555155.png)

### 3、match 表达式
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111120646624.png)
应用实例表达式计算
![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111122548297.png)

这里采用默认优先级，运算符重载是对let x=表达式的运算符进行重载；它让我们可以用写数学公式的自然方式（如 `1.2 + 3.4 * 2.0`）来“搭建”表达式树，而不是繁琐地直接构造节点（如 `Add(Num(1.2), Mul(Num(3.4), Num(2.0)))`）。

所以，当程序执行 `let x = Num(1.2) + Num(3.4) * Num(2.0) ...`这行代码时，**并没有进行数学计算**，它只是在内存中构建出了上面那棵复杂的“表达式树”，并把树根赋值给了变量 `x`。

a.calc() + b.calc()才是真正的运算。

### 4、Option
在部分应用场景中，**一个变量无法在整个生命周期内都被赋予有效值**，例如存在异常情况或可选的初始化设计等，为了高效且安全地表达这种“或有或无”的值，仓颉语言提供了Option 类型。

![image.png](https://gitee.com/zhang-junjie123/picture/raw/master/image/20251111123013349.png)

