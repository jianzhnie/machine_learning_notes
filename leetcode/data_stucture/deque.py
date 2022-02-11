class Deque(object):
    """双端队列."""
    def __init__(self) -> None:
        self.items = []

    def is_empty(self):
        return self.items == []

    def add_front(self, item):
        """在队头添加元素."""
        self.items.insert(0, item)

    def add_rear(self, item):
        """在队尾添加元素."""
        self.items.append(item)

    def remove_front(self):
        """从队头删除元素."""
        return self.items.pop(0)

    def remove_rear(self):
        """从队尾删除元素."""
        return self.items.pop(0)

    def length(self):
        return len(self.items)


if __name__ == '__main__':
    q = Deque()
    q.add_front('hello')
    q.add_front('world')
    q.add_front('itcast')

    q.add_rear('deep')
    q.add_rear('learning')
    q.add_rear('!!!')

    print(q.length())
    print(q.remove_front())
    print(q.remove_front())
    print(q.remove_front())
    print(q.remove_front())
    print(q.remove_front())
    print(q.remove_front())
