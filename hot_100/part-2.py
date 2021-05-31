# 11. 旋转数列查找，升序数列 x[1..n] 被转成 x[k+1:] + x[:k]
# 折半查找，利用nums[0] 判断位置，O(logn)
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        l, r = 0, len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[len(nums) - 1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
# 12. 最大子序列之和
# 动态规划，以第i个数结尾的最大子序列和f(i) = max(f(i-1)+nums[i], nums[i]), O(n)
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        numsLen = len(nums)
        last = -1e6
        maxSum = -1e6
        for i, n in enumerate(nums):
            last = max(last+n, n)
            if last > maxSum:
                maxSum = last
        return maxSum
# 分治法，分段，然后综合，每一段记录区间之和、区间内子序列和最大、区间内左/右邻子序列最大
# 虽然理论上复杂度低，但实际上效果不好
class Status:
    def __init__(self):
        self.l = 0
        self.r = 0
        self.i = 0
        self.m = 0
def pushUp(l, r):
    s = Status()
    s.l = max(l.l, l.i + r.l)
    s.r = max(r.r, l.r + r.i)
    s.i = l.i + r.i
    s.m = max(l.r + r.l, l.m, r.m)
    return s
def get(nums, l, r):
    if l == r:
        s = Status()
        n = nums[l]
        s.l, s.r, s.i, s.m = n, n, n, n
        return s
    mid = (l + r) // 2
    sl = get(nums, l, mid)
    sr = get(nums, mid+1, r)
    s = pushUp(sl, sr)
    return s
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        s = get(nums, 0, len(nums)-1)
        return s.m
# 13. 最远距离跳动，给定一个序列，每个数字代表此位置最远步数，问是否能从0开始到达最后一个元素
# 我的解法，DFS，超时
# 贪心算法，遍历每一个位置，如果该位置可达且能跳得更远则更新最远距离，O(n)
class Solution:
    def canJump(self, nums) :
        max_i = 0
        for i, jump in enumerate(nums):
            if max_i >= i and i + jump > max_i: 
                max_i = i + jump
        return max_i >= i