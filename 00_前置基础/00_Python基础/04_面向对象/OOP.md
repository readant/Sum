# 面向对象编程

## 类与对象

```python
class Person:
    # 类属性
    species = "人类"

    # 初始化方法
    def __init__(self, name, age):
        self.name = name    # 实例属性
        self.age = age

    # 实例方法
    def say_hello(self):
        return f"你好，我是{self.name}"

    # 魔术方法
    def __str__(self):
        return f"Person({self.name}, {self.age})"

# 创建对象
p1 = Person("张三", 20)
p2 = Person("李四", 22)

print(p1.say_hello())
print(p1)
```

## 继承

```python
class Student(Person):
    def __init__(self, name, age, school):
        super().__init__(name, age)  # 调用父类
        self.school = school

    # 重写方法
    def say_hello(self):
        return f"我是{self.name}，就读于{self.school}"

s = Student("王五", 18, "清华大学")
print(s.say_hello())
```

## 私有属性

```python
class Account:
    def __init__(self, balance):
        self.__balance = balance  # 私有属性（名称重整）

    def get_balance(self):
        return self.__balance

    def deposit(self, amount):
        self.__balance += amount

acc = Account(1000)
print(acc.get_balance())
# acc.__balance  # 错误，无法直接访问
```

## 类方法与静态方法

```python
class MathUtils:
    @staticmethod
    def add(x, y):
        return x + y

    @classmethod
    def create_double(cls, x):
        return cls(x * 2)

# 调用
MathUtils.add(1, 2)
MathUtils.create_double(5)
```
