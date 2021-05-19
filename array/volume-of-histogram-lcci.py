# https://leetcode-cn.com/problems/volume-of-histogram-lcci/
# 给定一个直方图(也称柱状图)，假设有人从上面源源不断地倒水，最后直方图能存多少水量?直方图的宽度为 1。

from typing import List

# 题解：
# 水漫法，双指针两侧逼近，以最低的一侧计算这一层的体积
# 算完总体积后，减去柱子的体积

class Solution:
    def trap(self, height) -> int:
        columns = sum(height)
        size = len(height)
        left, right = 0, size - 1
        volume, high = 0, 0
        while left <= right:
            # calculate region area
            nextHigh = min(height[left], height[right])
            delta = nextHigh - high
            volume += (right - left + 1) * delta
            # update next horizontal region
            while left <= right and height[left] <= high:
                left += 1
            while left <= right and height[right] <= high:
                right -= 1
            high = nextHigh
        return volume - columns

print(Solution().trap([0,1,0,2,1,0,1,3,2,1,2,1]))
print(Solution().trap([]))
