# 21. 最长上升子序列，给定一个可重复乱序序列，找到最长上升子序列
# Hash 表，先set去重，再为每个数找到上升序列，O(n)
class Solution:
    def longestConsecutive(self, nums):
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)

        return longest_streak
# 22. 只出现一次的数字，给数列，除一个元素外其余均出现两次
# 位运算，亦或本身相当于抵消
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return reduce(lambda x, y: x ^ y, nums)
# 23. 检测链表是否存在环路
# 快慢指针（要求常数内存的情况下）
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if not head or not head.next:
            return False
        slow = head
        fast = head.next
        while slow != fast:
            if not fast or not fast.next:
                return False
            slow = slow.next
            fast = fast.next.next
        return True
# 24. 最小残石，给一堆石头，每两块互碰，等重则一起消失，不等则剩下大的减小的，问最小能剩多少
# DP, 分为两堆，寻找差值最小的分法, dp[i+1][j] 表示 stones[:i+1] 是否能凑出重量 j
class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        total = sum(stones)
        n, m = len(stones), total // 2
        dp = [[False] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = True

        for i in range(n):
            for j in range(m + 1):
                if j < stones[i]:
                    dp[i + 1][j] = dp[i][j]
                else:
                    dp[i + 1][j] = dp[i][j] or dp[i][j - stones[i]]
        ans = None
        for j in range(m, -1, -1):
            if dp[n][j]:
                ans = total - 2 * j
                break
        return ans
