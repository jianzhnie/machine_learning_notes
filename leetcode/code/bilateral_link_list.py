class Node(object):
    """双向链表的结点."""
    def __init__(self, item):
        # item存放数据元素
        self.item = item
        # next 指向下一个节点的标识
        self.next = None
        # prev 指向上一结点
        self.prev = None


class BilateralLinkList(object):
    """双向链表."""
    def __init__(self):
        self.head = None

    def is_empty(self):
        """判断链表是否为空."""
        return self.head is None

    def length(self):
        """链表长度."""
        # 初始指针指向head
        cur = self.head
        count = 0
        # 指针指向None 表示到达尾部
        while cur is not None:
            count += 1
            # 指针下移
            cur = cur.next
        return count

    def items(self):
        """遍历链表."""
        # 获取head指针
        cur = self.head
        # 循环遍历
        while cur is not None:
            # 返回生成器
            yield cur.item
            # 指针下移
            cur = cur.next

    def add(self, item):
        """向链表头部添加元素."""
        node = Node(item)
        if self.is_empty():
            # 头部结点指针修改为新结点
            self.head = node
        else:
            # 新结点指针指向原头部结点
            node.next = self.head
            # 原头部 prev 指向 新结点
            self.head.prev = node
            # head 指向新结点
            self.head = node

    def append(self, item):
        """尾部添加元素."""
        node = Node(item)
        if self.is_empty():  # 链表无元素
            # 头部结点指针修改为新结点
            self.head = node
        else:  # 链表有元素
            # 移动到尾部
            cur = self.head
            while cur.next is not None:
                cur = cur.next
            # 新结点上一级指针指向旧尾部
            node.prev = cur
            # 旧尾部指向新结点
            cur.next = node

    def insert(self, index, item):
        """指定位置插入元素."""
        if index <= 0:
            self.add(item)
        elif index > self.length() - 1:
            self.append(item)
        else:
            node = Node(item)
            cur = self.head
            for i in range(index):
                cur = cur.next
            # 新结点的向下指针指向当前结点
            node.next = cur
            # 新结点的向上指针指向当前结点的上一结点
            node.prev = cur.prev
            # 当前上一结点的向下指针指向node
            cur.prev.next = node
            # 当前结点的向上指针指向新结点
            cur.prev = node

    def remove(self, item):
        """删除结点."""
        if self.is_empty():
            return
        cur = self.head
        # 删除元素在第一个结点
        if cur.item == item:
            # 只有一个元素
            if cur.next is None:
                self.head = None
                return True
            else:
                # head 指向下一结点
                self.head = cur.next
                # 下一结点的向上指针指向None
                cur.next.prev = None
                return True
        # 移动指针查找元素
        while cur.next is not None:
            if cur.item == item:
                # 上一结点向下指针指向下一结点
                cur.prev.next = cur.next
                # 下一结点向上指针指向上一结点
                cur.next.prev = cur.prev
                return True
            cur = cur.next
        # 删除元素在最后一个
        if cur.item == item:
            # 上一结点向下指针指向None
            cur.prev.next = None
            return True

    def find(self, item):
        """查找元素是否存在."""
        return item in self.items()
