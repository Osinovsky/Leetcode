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
