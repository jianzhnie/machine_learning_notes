

## 数组中只出现一次的数

描述: 一个整型数组里除了两个数字只出现一次，其他的数字都出现了k次。请写程序找出这两个只出现一次的数字。

```python
class Solution:
    def FindNumsAppearOnce(self, array):
​        res = []
​        for n in array:
​            if n not in res:
​                res.append(n)
​            else:
​                res.remove(n)
​        return res

class Solution:
    def foundOnceNumber(self, arr, k):
        dic = {}
        for i in arr:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1

        for i in dic:
            if dic.get(i) == 1:
                return i
```

## Leetcode2

描述: 输入一个长度为n的整型数组array，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

数据范围:

1 <= n <= 10^51<=*n*<=105

-100 <= a[i] <= 100−100<=*a*[*i*]<=100

要求:时间复杂度为 O(n)，空间复杂度为 O(n)

进阶:时间复杂度为 O(n)，空间复杂度为 O(1)

```python
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        sum = 0
        max = -100
        for i in array:
            sum += i
            if sum > max:
                max = sum
            if sum < 0:
                sum = 0
        return max
```

## Leetcode3

给定一个单链表的头结点pHead，长度为n，反转该链表后，返回新链表的表头。

数据范围： n\leq1000*n*≤1000

要求：空间复杂度 O(1)*O*(1) ，时间复杂度 O(n)*O*(*n*) 。

如当输入链表{1,2,3}时，

经反转后，原链表变为{3,2,1}，所以对应的输出为{3,2,1}。

以上转换过程如下图所示：

![img](https://uploadfiles.nowcoder.com/images/20211014/423483716_1634206291971/4A47A0DB6E60853DEDFCFDF08A5CA249)

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

# @param head ListNode类
# @return ListNode类
class Solution:
    def ReverseList(self , head: ListNode) -> ListNode:
        # write code here
        if not head:
        	return None
        pre, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre, cur = cur, nxt
        return pre
```

## 最长公共前缀

描述: 给你一个大小为 n 的字符串数组 strs ，其中包含n个字符串 , 编写一个函数来查找字符串数组中的最长公共前缀，返回这个公共前缀。

- 将字符串数组看作一个二维空间，每一次从第一列开始。
- 确定所有字符子串中第一列字符。
- 逐层扫描后面每一列，遇到不同字符停止扫描。
- 图解：
- ![图片说明](https://uploadfiles.nowcoder.com/images/20210714/583846419_1626256249067/3E866D7DC0F594B37D675D8164116AB5)

```python
# @param strs string字符串一维数组
# @return string字符串
#
class Solution:
    def longestCommonPrefix(self , strs ):
        # write code here
        if not strs:
            return ""
        result = ""
        min_len = min([len(s)for s in strs])
        for i in range(min_len):
            now = strs[0][i]
            if all(s[i]==now for s in strs):
                result += now
            else:
                return result
        return result
```

## leetcode5

**描述:**

请实现有重复数字的升序数组的二分查找

给定一个 元素有序的（升序）长度为n的整型数组 nums 和一个目标值 target ，写一个函数搜索 nums 中的第一个出现的target，如果目标值存在返回下标，否则返回 -1

```python
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
# 如果目标值存在返回下标，否则返回 -1
# @param nums int整型一维数组
# @param target int整型
# @return int整型
#
import sys
class Solution:
    def search(self, nums, target):
        # find in range [left, right)
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        if nums[left] != target:
            return -1
        else:
            return left
```



## leetcode6

描述: 判断给定的链表中是否有环。如果有环则返回true，否则返回false。

输入分为2部分，第一部分为链表，第二部分代表是否有环，然后将组成的head头结点传入到函数里面。-1代表无环，其它的数字代表有环，这些参数解释仅仅是为了方便读者自测调试。实际在编程时读入的是链表的头节点。

例如输入{3,2,0,-4},1时，对应的链表结构如下图所示：

![img](https://uploadfiles.nowcoder.com/images/20211105/423483716_1636083991397/9A058E6590B998B9F7B637155842F993)

可以看出环的入口结点为从头结点开始的第1个结点，所以输出true。

```python
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

#
#
# @param head ListNode类
# @return bool布尔型
#
class Solution:
    def hasCycle(self , head: ListNode) -> bool:
        if not head:
          return False
        fast = slow = head
        while fast and fast.next:
            fast = fast.next
            slow = slow.next
            if fast == slow:
                return True
        return False
```



## 跳台阶

描述： 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个 n 级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

此题和斐波拉契数列做法一样。也将用三个方法来解决，从入门到会做。 考察知识：递归，记忆化搜索，动态规划和动态规划的空间优化。 难度：一星

\#题解 ###方法一：递归 题目分析，假设f[i]表示在第i个台阶上可能的方法数。逆向思维。如果我从第n个台阶进行下台阶，下一步有2中可能，一种走到第n-1个台阶，一种是走到第n-2个台阶。所以f[n] = f[n-1] + f[n-2]. 那么初始条件了，f[0] = f[1] = 1。 所以就变成了：f[n] = f[n-1] + f[n-2], 初始值f[0]=1, f[1]=1，目标求f[n] 看到公式很亲切，代码秒秒钟写完。

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        p = 0
        q = 1
        r = 1
        for i in range(number-1):
            p = q
            q = r
            r = p+q
        return r
```

```python
# -*- coding:utf-8 -*-
class Solution:
    def jumpFloor(self, number):
        if number < 1:
            return 0
        a, b = 0, 1
        for _ in range(number):
            a, b = b, a + b
        return b
```

## 合并两个有序的数组

给出一个有序的整数数组 A 和有序的整数数组 B ，请将数组 B 合并到数组 A 中，变成一个有序的升序数组
注意：
1.保证 A 数组有足够的空间存放 B 数组的元素， A 和 B 中初始的元素数目分别为 m 和 n，A的数组空间大小为 m+n

2.不要返回合并的数组，将数组 B 的数据合并到 A 里面就好了

3.A 数组在[0,m-1]的范围也是有序的

```python
# @param A int整型一维数组
# @param B int整型一维数组
# @return void
#
class Solution:
    def merge(self , A, m, B, n):
        # write code here
        for i in range(len(A)-m):
            A.pop()
        for j in range(n):
            A.append(B[j])
        A.sort()
        return A

class Solution:
    def merge(self , A, m, B, n):
        A[m:] = B
        A.sort()
        return A

class Solution:
    def merge(self , A, m, B, n):
        # write code here
        if B == []:
            return A
        if A == []:
            return B

        # 从后往前
        # m+n长度，但是序号要减一
        while m> 0 and n>0:
            if A[m-1]> B[n-1]:
                A[m+n-1] = A[m-1]
                m -= 1
            else:
                A[m+n-1] = B[n-1]
                n -= 1
        if m >= 0:
					A[:m+1] = A[:m+1]   #A还有剩余
        if n > 0:
          A[:n] = B[:n]  # B还有剩余
        return A
```

## 连续子数组的最大和

输入一个长度为n的整型数组array，数组中的一个或连续多个整数组成一个子数组。求所有子数组的和的最大值。

```python
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
#
# @param array int整型一维数组
# @return int整型
#
class Solution:
    def FindGreatestSumOfSubArray(self , array: List[int]) -> int:
        max = array[0]
        sum = array[0]
        for i in range(1, len(array)):
            if sum > 0:
                sum += array[i]
            else:
                sum = array[i]
            if max < sum:
                max = sum
        return max


class Solution:
    def FindGreatestSumOfSubArray(self, array):
        # write code here
        sum=0
        max=-100
        for i in array:
            sum+=i
            if sum>max:
                max=sum
            if sum<0:
                sum=0
        return max
```

## 买卖股票的最好时机

假设你有一个数组prices，长度为n，其中prices[i]是股票在第i天的价格，请根据这个价格数组，返回买卖股票能获得的最大收益

1.你可以买入一次股票和卖出一次股票，并非每天都可以买入或卖出一次，总共只能买入和卖出一次，且买入必须在卖出的前面的某一天

2.如果不能获取到任何利润，请返回0

### 算法思想一：暴力法

需要找出给定数组中两个数字之间的最大差值（即，最大利润）。此外，第二个数字（卖出价格）必须大于第一个数字（买入价格）。
形式上，对于每组 i 和 j（其中 j>i）我们需要找出 max(prices[j]−prices[i])

```python
class Solution:
    def maxProfit(self , prices ):
        # write code here
        res = 0
        for i in range(len(prices)):
            for j in range(i + 1, len(prices)):
                res = max(res, prices[j] - prices[i])
        return res

class Solution:
    def maxProfit(self , prices ):
        # write code here
        if not prices:
          return 0
        maxp = 0
        minp = prices[0]
        for p in prices:
            maxp = max(maxp, p-minp)
            minp = min(minp, p)
        return maxp
```

```python
def f(x):
  return x**2

def h(x):
  return 2*x

def gradient_desent(x0, value=a, step=0.01):
  x = x0
  count = 0
  diff = f(x) - value
  while abs(diff) > 0.00001:
    x = x - step * h(x)
    diff = f(x) - value
    count +=1
  return x, count
```

## 求平方根

**题解一：二分**
题解思路： 二分查找比a<=sqrt(x)<=b
如果 mid * mid <=x 且(mid+1)(mid+1) <x 返回mid
如果mid * mid > x right = mid-1;
否则 left = mid+1;
图示：

```python
# @param x int整型
# @return int整型
#
class Solution:
    def sqrt(self , x ):
        # write code here
        if x<=1:
          return x
        left=1
        right=x
        while left<=right:
            mid= (left+right)//2
            if mid*mid>x:
                right=mid-1
            elif mid*mid<x:
                left=mid+1
            else:
              return mid
        return right
```



## 括号序列

描述    给出一个仅包含字符'(',')','{','}','['和']',的字符串，判断给出的字符串是否是合法的括号序列括号必须以正确的顺序关闭，"()"和"()[]{}&quo

算法思想一：栈+哈希表

算法流程

1、构建哈希表 k，其中key为 右括号，value为左括号

2、遍历字符串

  1、判断字符是否在 k.values() 中；

​    若在其中则字符入栈；

​    否则判断栈是否为空 或者 该字符的 values 是否与 栈顶元素相同；若不同则直接返回 false，若相同则栈顶元素出栈

3、判断栈内元素是否为空，为空则返回 true，反之返回 false

**图解：**

![img](https://uploadfiles.nowcoder.com/images/20210714/889362376_1626232199184/040215D195C8C0AEA2545C4789825E5E)

```python
# @param s string字符串
# @return bool布尔型
#
class Solution:
    def isValid(self , s ):
        # write code here
        mapping = {")":"(", "]":"[", "}":"{"}
        stack = []
        for i, char in enumerate(s):
            if char not in mapping:#left
                stack.append(char)
            else:
                if not stack or stack[-1] != mapping[char]:
                    return False
                stack.pop()
        return len(stack) == 0

```

## 数字在升序数组中出现的次数

描述： 给定一个长度为 n 的非降序数组和一个非负数整数 k ，要求统计 k 在数组中出现的次数

```python
# -*- coding:utf-8 -*-
class Solution:
    def GetNumberOfK(self, data, k):
        # write code here
        if not data:
            return 0
        return data.count(k)


class Solution:
    def GetNumberOfK(self, data, k):
        num = 0
        for i in range (0,len(data)):
            if data[i] > k :
                break
            if data[i] == k:
                num = num + 1
        return num
```

## 矩阵的最小路径和

描述

给定一个 n * m 的矩阵 a，从左上角开始每次只能向右或者向下走，最后到达右下角的位置，路径上所有的数字累加起来就是路径和，输出所有的路径中最小的路径和。

```python
# @param matrix int整型二维数组 the matrix
# @return int整型
#
class Solution:
    def minPathSum(self, matrix):
        # write code here
        n, m = len(matrix), len(matrix[0])
        dp = [[0]*m for i in range(n)]
        for i in range(n):
            for j in range(m):
                if i==0 and j==0:
                    dp[i][j] = matrix[0][0]
                elif i==0 and j!=0:
                    dp[i][j] = dp[i][j-1] + matrix[i][j]
                elif j==0 and i!=0:
                    dp[i][j] = dp[i-1][j] + matrix[i][j]
                else:
                    dp[i][j] = min(dp[i][j-1], dp[i-1][j]) + matrix[i][j]
        return dp[-1][-1]
```
