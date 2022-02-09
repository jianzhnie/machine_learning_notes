class ListNode(object):
    """链表的结点."""
    def __init__(self, item):
        # item存放数据元素
        self.item = item
        # next是下一个节点的标识
        self.next = None


class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        node = ListNode(-1)
        while l1 and l2:
            if l1.item <= l2.item:
                node.next = l1
                l1 = l1.next
            else:
                node.next = l2
                l2 = l2.next
            prev = node.next

        # 合并后 l1 和 l2 最多只有一个还未被合并完，
        # 我们直接将链表末尾指向未合并完的链表即可
        prev.next = l1 if l1 is not None else l2

        return node.next

    def mergeTwoLists_(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.item < l2.item:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
