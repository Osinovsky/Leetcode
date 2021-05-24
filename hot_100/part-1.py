# https://leetcode-cn.com/problemset/leetcode-hot-100/

# 1. 两数之和，给数列，找到之和为目标的两数下标
# 输入：nums = [2,7,11,15], target = 9
# 输出：[0,1]
# 解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
# 我的方法：双指针，和小于目标则左指针向右，大于则右指针向左
# 由于引入了排序，复杂度与Python的sorted复杂度一致(O(n log n))
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        indexNums = sorted(enumerate(nums), key=lambda x: x[1])
        left, right = 0, len(indexNums)-1
        leftIndex = indexNums[left][0]
        leftValue = indexNums[left][1]
        rightIndex = indexNums[right][0]
        rightValue = indexNums[right][1]
        sumSum = leftValue + rightValue
        while left < right:
            if sumSum == target:
                return [leftIndex, rightIndex]
            elif sumSum < target:
                while sumSum < target and left < right:
                    left += 1
                    leftValue = indexNums[left][1]
                    sumSum = leftValue + rightValue
                leftIndex = indexNums[left][0]
            else:
                while sumSum > target and left < right:
                    right -= 1
                    rightValue = indexNums[right][1]
                    sumSum = leftValue + rightValue
                rightIndex = indexNums[right][0]
        return []
# 官方方法：Hash表，便历，差值在表里就返回，否则加入表。和我的结果相差不大，但复杂度是 O(N)
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashtable = dict()
        for i, num in enumerate(nums):
            if target - num in hashtable:
                return [hashtable[target - num], i]
            hashtable[num] = i
        return []

# 2. 最长连续子串
# 我的方法，使用队列记录子串中的各个字符和位置，遇到重复的则popleft
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        maxLen = 0
        sq = collections.deque()
        iq = collections.deque()
        left = 0
        for ind, val in enumerate(s):
            if val in sq:
                popSize = sq.index(val)
                for _ in range(popSize):
                    sq.popleft()
                    iq.popleft()
                sq.popleft()
                l = ind - left
                if l > maxLen:
                    maxLen = l
                left = iq.popleft() + 1
            sq.append(val)
            iq.append(ind)
        l = len(sq)
        if l > maxLen:
            maxLen = l
        return maxLen
# 最快方法，不需要清空记录的队列或者字典，可以用与left比较的方式实现
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        visited = {}
        maxLen, left = 0, -1
        for i, c in enumerate(s):
            if visited.get(c, -1) >= left:
                left = visited.get(c, -1)
            visited[c] = i
            if i - left > maxLen:
                maxLen = i - left
        return maxLen
# 3. 寻找两个正序数组的中位数
# 官方解法，划分数组, O(log min(m,n)))
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            return self.findMedianSortedArrays(nums2, nums1)
        infinty = 2**40
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        # median1：前一部分的最大值
        # median2：后一部分的最小值
        median1, median2 = 0, 0
        while left <= right:
            # 前一部分包含 nums1[0 .. i-1] 和 nums2[0 .. j-1]
            # // 后一部分包含 nums1[i .. m-1] 和 nums2[j .. n-1]
            i = (left + right) // 2
            j = (m + n + 1) // 2 - i
            # nums_im1, nums_i, nums_jm1, nums_j 分别表示 nums1[i-1], nums1[i], nums2[j-1], nums2[j]
            nums_im1 = (-infinty if i == 0 else nums1[i - 1])
            nums_i = (infinty if i == m else nums1[i])
            nums_jm1 = (-infinty if j == 0 else nums2[j - 1])
            nums_j = (infinty if j == n else nums2[j])
            if nums_im1 <= nums_j:
                median1, median2 = max(nums_im1, nums_jm1), min(nums_i, nums_j)
                left = i + 1
            else:
                right = i - 1
        return (median1 + median2) / 2 if (m + n) % 2 == 0 else median1
# 4. 最长回文子串
# Manacher 方法，O(n)
class Solution:
    def expand(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return (right - left - 2) // 2

    def longestPalindrome(self, s: str) -> str:
        end, start = -1, 0
        s = '#' + '#'.join(list(s)) + '#'
        arm_len = []
        right = -1
        j = -1
        for i in range(len(s)):
            if right >= i:
                i_sym = 2 * j - i
                min_arm_len = min(arm_len[i_sym], right - i)
                cur_arm_len = self.expand(s, i - min_arm_len, i + min_arm_len)
            else:
                cur_arm_len = self.expand(s, i, i)
            arm_len.append(cur_arm_len)
            if i + cur_arm_len > right:
                j = i
                right = i + cur_arm_len
            if 2 * cur_arm_len + 1 > end - start:
                start = i - cur_arm_len
                end = i + cur_arm_len
        return s[start+1:end+1:2]
# 5. 简易正则表达式 . 与 *\
# 官方解法，O(mn)
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)

        def matches(i: int, j: int) -> bool:
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]

        f = [[False] * (n + 1) for _ in range(m + 1)]
        f[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    f[i][j] |= f[i][j - 2]
                    if matches(i, j - 1):
                        f[i][j] |= f[i - 1][j]
                else:
                    if matches(i, j):
                        f[i][j] |= f[i - 1][j - 1]
        return f[m][n]
# 6. 盛最多水的容器，柱状图里找两个柱子中间装水，找到最大面积
# 双指针法，每次移动小的一侧指针，相当于贪婪搜索（可用反证法证明正确）
class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height)-1
        hl, hr = height[l], height[r]
        step = r - l
        maxA = 0
        while l < r:
            if hl < hr:
                area = step * hl
                l += 1
                hl = height[l]
            else:
                area = step * hr
                r -= 1
                hr = height[r]
            if area > maxA:
                maxA = area
            step -= 1
        return maxA
# 7. 三数之和，全部非重复解 （sum=0）
# 双指针，跳过重复数值以避免重复解，时间复杂度 O(n^2)
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        if n < 3:
            return []
        nums.sort()
        res = []
        nim1 = 1
        ni = 1
        for i in range(n - 2):
            nim1 = ni
            ni = nums[i]
            if ni > 0:
                return res
            if ni == nim1:
                continue
            l = i + 1
            r = n - 1
            while l < r:
                nl = nums[l]
                nr = nums[r]
                sumThree = ni + nl + nr
                if sumThree == 0:
                    res.append([ni, nl, nr])
                    while l < r and nl == nums[l + 1]:
                        l += 1
                    while l < r and nr == nums[r - 1]:
                        r -= 1
                    l = l + 1
                    r = r - 1
                elif sumThree > 0:
                    r -= 1
                else:
                    l += 1
        return res
