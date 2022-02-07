class Stack(object):
    def __init__(self) -> None:
        self.items = []

    def is_empty(self):
        """判断是否为空."""
        return self.items == []

    def push(self, item):
        """入栈
        """
        self.items.append(item)

    def pop(self):
        """出栈
        """
        return self.items.pop()

    def peek(self):
        return self.items[-1]

    def length(self):
        return len(self.items)


if __name__ == '__main__':
    stack = Stack()
    stack.push('hello')
    stack.push('world')
    stack.push('itcast')
    print(stack.length())
    print(stack.peek())
    print(stack.pop())
    print(stack.pop())
    print(stack.pop())
