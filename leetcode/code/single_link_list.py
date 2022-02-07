class Node(object):
    """链表的结点."""
    def __init__(self, item):
        # item存放数据元素
        self.item = item
        # next是下一个节点的标识
        self.next = None


class SingleLinkList(object):
    def __init__(self):
        self.head = None

    def is_empty(self):
        """判断链表是否为空."""
        return self.head is None

    def length(self):
        """链表长度."""
        count = 0
        cur = self.head
        while cur is not None:
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
        while cur.next is not None:
            yield cur.item
            cur = cur.next
        yield cur.item

    def add(self, item):
        """头部添加结点."""
        node = Node(item)
        # 将新节点的链接域next指向头节点，即head指向的位置
        node.next = self.head
        # 修改 head 指向新结点
        self.head = node

    def append(self, item):
        """尾部添加结点."""
        node = Node(item)

        # 先判断链表是否为空，若是空链表，则将head指向新节点
        if self.is_empty():  # 为空
            self.head = node
        # 若不为空，则找到尾部，将尾节点的next指向新节点
        else:
            # 寻找尾部
            cur = self.head
            while cur.next is not None:
                cur = cur.next
            # 尾部指针指向新结点
            cur.next = node

    def insert(self, index, item):
        """指定位置添加结点."""
        # 若指定位置pos为第一个元素之前，则执行头部插入
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
        while cur is not None:
            # 找到了指定元素
            if cur.item == item:
                # 如果第一个就是删除的节点
                if not pre:
                    # 将头指针指向头节点的后一个节点
                    self.head = cur.next
                else:
                    # 将删除位置前一个节点的next指向删除位置的后一个节点
                    pre.next = cur.next
            else:
                # 继续按链表后移节点
                pre = cur
                cur = cur.next

    def find(self, item):
        """查找元素是否存在."""
        return item in self.items()
