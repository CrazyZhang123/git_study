# 1、继承

### （1）关于super()调用父类方法
详见文档：
[[python中super().]]



# 2、函数

## (1) map函数
它能够将一个函数作用于一个或多个可迭代对象的元素上，从而实现快速高效的数据处理。
详见[[Python的map()函数]]

```python
numbers = [1, 2, 3, 4, 5]
squared_numbers = map(lambda x: x * x, numbers)

print(list(squared_numbers))
# 输出: [1, 4, 9, 16, 25]
```


# 3、python内置特性
## (1) \_\_call__方法

Python 拥有一系列内置方法，`__call__` 便是其中之一。该方法使 Python 开发者能够编写 “实例可像函数一样调用” 的类。当类中定义了 `__call__` 方法后，调用对象（如 `obj(arg1, arg2)`）会自动触发 `obj.__call__(arg1, arg2)`。这种特性让对象具备了函数的行为，从而实现更灵活、可复用的代码。
详见：

[[Python 中的 __call__ 方法]]