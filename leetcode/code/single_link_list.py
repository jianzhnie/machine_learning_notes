class Node(object):
    """链表的结点."""
    def __init__(self, item):
        # item存放数据元素
        self.item = item
        # next是下一个节点的标识
        self.next = None


class SingleCycleLinkList(object):
    def __init__(self):
        self.head = None

    def is_empty(self):
        """判断链表是否为空."""
        return self.head is None

    def length(self):
        """链表长度."""
        # 链表为空
        if self.is_empty():
            return 0
        # 链表不为空
        count = 1
        cur = self.head
        while cur.next != self.head:
            count += 1
            # 指针下移
            cur = cur.next
        return count

    def items(self):
        """遍历链表."""
        # 链表为空
        if self.is_empty():
            return
        # 链表不为空
        cur = self.head
        while cur.next != self.head:
            yield cur.item
            cur = cur.next
        yield cur.item

    def add(self, item):
        """头部添加结点."""
        node = Node(item)
        if self.is_empty():  # 为空
            self.head = node
            node.next = self.head
        else:
            # 添加结点指向head
            node.next = self.head
            cur = self.head
            # 移动结点，将末尾的结点指向node
            while cur.next != self.head:
                cur = cur.next
            cur.next = node
        # 修改 head 指向新结点
        self.head = node

    def append(self, item):
        """尾部添加结点."""
        node = Node(item)
        if self.is_empty():  # 为空
            self.head = node
            node.next = self.head
        else:
            # 寻找尾部
            cur = self.head
            while cur.next != self.head:
                cur = cur.next
            # 尾部指针指向新结点
            cur.next = node
            # 新结点指针指向head
            node.next = self.head

    def insert(self, index, item):
        """指定位置添加结点."""
        if index <= 0:  # 指定位置小于等于0，头部添加
            self.add(item)
        # 指定位置大于链表长度，尾部添加
        elif index > self.length() - 1:
            self.append(item)
        else:
            node = Node(item)
            cur = self.head
            # 移动到添加结点位置
            for i in range(index - 1):
                cur = cur.next
            # 新结点指针指向旧结点
            node.next = cur.next
            # 旧结点指针 指向 新结点
            cur.next = node

    def remove(self, item):
        """删除一个结点."""
        if self.is_empty():
            return
        cur = self.head
        pre = Node
        # 第一个元素为需要删除的元素
        if cur.item == item:
            # 链表不止一个元素
            if cur.next != self.head:
                while cur.next != self.head:
                    cur = cur.next
                # 尾结点指向 头部结点的下一结点
                cur.next = self.head.next
                # 调整头部结点
                self.head = self.head.next
            else:
                # 只有一个元素
                self.head = None
        else:
            # 不是第一个元素
            pre = self.head
            while cur.next != self.head:
                if cur.item == item:
                    # 删除
                    pre.next = cur.next
                    return True
                else:

                    pre = cur  # 记录前一个指针
                    cur = cur.next  # 调整指针位置
        # 当删除元素在末尾
        if cur.item == item:
            pre.next = self.head
            return True

    def find(self, item):
        """查找元素是否存在."""
        return item in self.items()
