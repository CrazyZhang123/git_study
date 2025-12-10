
# Lab1 实验报告

## 1. 代码结构与设计

### 1.1 整体架构

本实验实现了一个基于访问者模式（Visitor Pattern）的解释器求值器。核心设计遵循以下原则：

- **访问者模式**：`Evaluator` 类继承自 `Visitor<Value>`，通过重写各个 `visit` 方法实现对不同 AST 节点的求值
- **链式作用域管理**：使用 `Environment` 类实现词法作用域（lexical scoping），采用链式结构（cactus stack）支持嵌套作用域
- **运行时值表示**：使用 `Value` 枚举类型统一表示所有运行时值（整数、字符串、布尔值、单元类型）

### 1.2 表达式求值设计

表达式求值采用递归下降的方式，通过访问者模式优雅地处理各种表达式类型：

#### 二元表达式求值

二元表达式的求值逻辑集中在 `visit(BinaryExpr)` 方法中，采用模式匹配（pattern matching）处理不同运算符：

- **短路求值**：对于逻辑运算符 `&&` 和 `||`，实现了短路求值机制。例如，`&&` 运算符在左操作数为 `false` 时直接返回，不计算右操作数
- **类型多态**：加法运算符支持整数和字符串两种类型，乘法运算符还支持字符串与整数的组合（字符串重复）
- **溢出处理**：算术运算的溢出通过捕获 `OverflowException` 异常来处理，并转换为相应的运行时错误

```cangjie
case TokenKind.ADD => match ((left, right)) {
    case (VInteger(a), VInteger(b)) => Value.from(a + b)
    case (VString(a), VString(b)) => Value.from(a + b)
    case _ => throw CjcjRuntimeErrorWithLocation(...)
}
```

#### 一元表达式求值

一元表达式（取负、逻辑非、位非）同样使用模式匹配，确保类型安全。

#### 特殊表达式处理

- **字符串重复**：实现了 `repeatString` 辅助函数，支持字符串与整数的乘法运算，并检查负数情况
- **幂运算**：实现了 `powerInt` 函数，通过循环乘法计算幂，并在每次乘法时捕获溢出异常
- **除零检查**：实现了 `safeDivide` 和 `safeModulo` 函数，在执行除法/取余前检查除数是否为零

### 1.3 变量作用域管理

作用域管理是本实验的核心设计之一，采用链式环境（cactus stack）实现：

#### Environment 类设计

`Environment<K, V>` 是一个泛型类，采用链式结构：
- 每个环境维护一个 `enclosing` 引用，指向外层环境
- `getGlobal` 方法递归向上查找变量，实现词法作用域
- `getLocal` 方法仅查找当前作用域
- `declare` 方法在当前作用域声明变量，防止重复定义

#### VarInfo 结构

变量信息通过 `VarInfo` 类封装：
- `value: ?Value`：变量的值（可选）
- `mutable: Bool`：是否为可变变量（`var` vs `let`）
- `initialized: Bool`：是否已初始化
- `declaredType: ?String`：声明的类型（可选，用于类型检查）

#### 作用域创建与恢复

- **块作用域**：每个 `Block` 节点求值时创建新环境，求值结束后恢复原环境
- **条件分支作用域**：`if` 表达式的 `then` 和 `else` 分支各自创建独立作用域
- **循环作用域**：`while` 循环的每次迭代都创建新的块作用域，确保迭代间变量隔离

```cangjie
public open override func visit(block: Block): Value {
    let oldEnv = environment
    environment = Environment(oldEnv)  // 创建新作用域
    
    var ret = Value.VUnit
    for (stmt in block.nodes.iterator()) {
        ret = stmt.traverse(this)
    }
    
    environment = oldEnv  // 恢复原作用域
    return ret
}
```

### 1.4 控制流实现

#### if 表达式

`if` 表达式的实现考虑了以下情况：
- 条件表达式必须是布尔类型
- `then` 分支在独立作用域中求值
- 如果没有 `else` 分支，且条件为真，返回 `VUnit`；否则返回 `else` 分支的值
- 通过 `hasElse` 标志判断是否存在 `else` 分支（`Block` 或 `IfExpr` 类型）

#### while 循环

`while` 循环的实现较为复杂：
- 使用 `inLoop_` 标志跟踪是否在循环中，用于检查 `break`/`continue` 的合法性
- 每次迭代创建新的块作用域，确保迭代间变量隔离
- 使用异常机制实现 `break` 和 `continue`：
  - `BreakException`：跳出循环
  - `ContinueException`：继续下一次迭代
- 使用 `try-finally` 确保环境状态正确恢复

```cangjie
public open override func visit(expr: WhileExpr): Value {
    let oldInLoop = inLoop_
    inLoop_ = true
    let outerEnv = environment
    try {
        while (true) {
            // 条件检查
            // 每次迭代创建新作用域
            let savedEnv = environment
            environment = Environment(savedEnv)
            try {
                expr.block.traverse(this)
            } catch (e: BreakException) {
                environment = savedEnv
                break
            } catch (e: ContinueException) {
                environment = savedEnv
                continue
            }
            environment = savedEnv
        }
    } finally {
        environment = outerEnv
        inLoop_ = oldInLoop
    }
    Value.VUnit
}
```

### 1.5 类型检查与错误处理

#### 运行时类型检查

所有类型检查在运行时进行：
- `sameType` 辅助函数检查两个值是否类型相同
- `typeName` 辅助函数获取值的类型名称
- 每个运算符都有对应的类型检查，不符合时抛出 `CjcjRuntimeErrorWithLocation`

#### 错误码系统

使用 `ErrorCode` 枚举定义所有可能的运行时错误，每个错误包含：
- 错误码（`ErrorCode`）
- 额外消息（`extraMessage`）
- 错误位置（`fromWhere: Node`）

### 1.6 设计亮点

1. **优雅的表达式处理**：通过访问者模式和模式匹配，表达式求值逻辑清晰、易于扩展
2. **灵活的作用域管理**：链式环境结构支持任意深度的嵌套作用域，且实现简洁
3. **类型安全的运行时检查**：所有类型检查集中在求值过程中，错误信息包含位置信息
4. **异常驱动的控制流**：使用异常实现 `break`/`continue`，代码结构清晰
5. **辅助函数封装**：将复杂逻辑（如字符串重复、幂运算）封装为辅助函数，提高代码可读性

## 2. 遇到的问题与解决方案

### 2.1 作用域管理问题

**问题描述**：在实现 `while` 循环时，最初没有为每次迭代创建新的作用域，导致迭代间变量状态相互影响。

**解决方案**：在循环体的每次迭代开始时创建新的环境，迭代结束时恢复，确保每次迭代都有独立的变量作用域。

### 2.2 未初始化 let 变量的赋值

**问题描述**：根据语言规范，未初始化的 `let` 变量可以通过赋值语句初始化，但已初始化的 `let` 变量不可重新赋值。

**解决方案**：在 `visit(AssignExpr)` 中，检查变量是否为 `let` 类型且未初始化，如果是，则允许赋值并标记为已初始化。

```cangjie
if (!varInfo.mutable) {
    if (!varInfo.initialized) {
        // 允许赋值给未初始化的 let 变量
        varInfo.value = rightValue
        varInfo.initialized = true
        return Value.VUnit
    } else {
        throw CjcjRuntimeErrorWithLocation(...)
    }
}
```

### 2.3 if 表达式返回值处理

**问题描述**：`if` 表达式在没有 `else` 分支时，如果条件为真，应该返回什么值？

**解决方案**：根据语言规范，没有 `else` 分支的 `if` 表达式在条件为真时返回 `VUnit`，否则返回 `VUnit`（因为没有 `else` 分支）。

### 2.4 溢出检测

**问题描述**：如何检测整数运算的溢出？

**解决方案**：依赖 Cangjie 语言的 `OverflowException` 异常机制。在执行可能溢出的运算时，捕获该异常并转换为相应的运行时错误。对于幂运算，在循环乘法的每一步都捕获溢出异常。

### 2.5 短路求值实现

**问题描述**：逻辑运算符 `&&` 和 `||` 需要实现短路求值，但访问者模式要求先访问所有子节点。

**解决方案**：在 `visit(BinaryExpr)` 中，对于 `&&` 和 `||` 运算符，先求值左操作数，根据结果决定是否求值右操作数。这需要在访问右操作数之前进行判断。

### 2.6 实验难度评估

本实验的难度主要体现在：
1. **作用域管理的复杂性**：需要正确理解词法作用域，并在各种控制结构中正确创建和恢复环境
2. **类型系统的实现**：需要在运行时进行类型检查，确保类型安全
3. **控制流的异常处理**：使用异常实现 `break`/`continue` 需要仔细处理环境状态
