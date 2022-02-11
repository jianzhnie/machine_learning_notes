class Node(object):
    """节点"""
    def __init__(self, item):
        self.item = item
        self.next = None


class SinCycLinkedlist(object):
    """单向循环链表."""
    def __init__(self):
        self.head = None

    def is_empty(self):
        """判断链表是否为空."""
        return self.head is None

    def length(self):
        """返回链表的长度."""
        # 如果链表为空，返回长度0
        if self.is_empty():
            return 0
        count = 1
        cur = self.head
        while cur.next != self.head:
            count += 1
            cur = cur.next
        return count

    def add(self, item):
        """头部添加节点."""
        node = Node(item)
        if self.is_empty():
            self.head = node
            node.next = self.head
        else:
            # 添加的节点指向head
            node.next = self.head
            # 移到链表尾部，将尾部节点的next指向node
            cur = self.head
            while cur.next != self.head:
                cur = cur.next
            cur.next = node
            # head指向添加node的
            self.head = node

    def append(self, item):
        """尾部添加节点."""
        node = Node(item)
        if self.is_empty():
            self.head = node
            node.next = self.head
        else:
            # 移到链表尾部
            cur = self.head
            while cur.next != self.head:
                cur = cur.next
            # 将尾节点指向node
            cur.next = node
            # 将node指向头节点head
            node.next = self.head

    def insert(self, pos, item):
        """在指定位置添加节点."""
        if pos <= 0:
            self.add(item)
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            node = Node(item)
            cur = self.head
            count = 0
            # 移动到指定位置的前一个位置
            while count < (pos - 1):
                count += 1
                cur = cur.next
            node.next = cur.next
            cur.next = node

    def remove(self, item):
        """删除一个节点."""
        # 若链表为空，则直接返回
        if self.is_empty():
            return
        # 将cur指向头节点
        cur = self.head
        pre = None
        # 若头节点的元素就是要查找的元素item
        if cur.item == item:
            # 如果链表不止一个节点
            if cur.next != self.head:
                # 先找到尾节点，将尾节点的next指向第二个节点
                while cur.next != self.head:
                    cur = cur.next
                # cur指向了尾节点
                cur.next = self.head.next
                self.head = self.head.next
            else:
                # 链表只有一个节点
                self.head = None
        else:
            pre = self.head
            # 第一个节点不是要删除的
            while cur.next != self.head:
                # 找到了要删除的元素
                if cur.item == item:
                    # 删除
                    pre.next = cur.next
                    return
                else:
                    pre = cur
                    cur = cur.next
            # cur 指向尾节点
            if cur.item == item:
                # 尾部删除
                pre.next = cur.next

    def search(self, item):
        """查找节点是否存在."""
        if self.is_empty():
            return False
        cur = self.head
        if cur.item == item:
            return True
        while cur.next != self.head:
            cur = cur.next
            if cur.item == item:
                return True
        return False
