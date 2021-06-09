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
# 25. 最大正方形
# dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + xij
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0
        maxSide = 0
        rows, columns = len(matrix), len(matrix[0])
        dp = [[0] * columns for _ in range(rows)]
        for i in range(rows):
            for j in range(columns):
                if matrix[i][j] == '1':
                    if i == 0 or j == 0:
                        dp[i][j] = 1
                    else:
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    maxSide = max(maxSide, dp[i][j])        
        maxSquare = maxSide * maxSide
        return maxSquare
# 26. 滑动窗口最大值
# 优先队列, O(nlogn)
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        # 注意 Python 默认的优先队列是小根堆
        q = [(-nums[i], i) for i in range(k)]
        heapq.heapify(q)
        ans = [-q[0][0]]
        for i in range(k, n):
            heapq.heappush(q, (-nums[i], i))
            while q[0][1] <= i - k:
                heapq.heappop(q)
            ans.append(-q[0][0])
        return ans
# 单调队列, O(n)，如果左边小于右边的值，滑动时左边的值完全可以pop，因为只要右边在它就没用
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        q = collections.deque()
        for i in range(k):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)
        ans = [nums[q[0]]]
        for i in range(k, n):
            while q and nums[i] >= nums[q[-1]]:
                q.pop()
            q.append(i)
            while q[0] <= i - k:
                q.popleft()
            ans.append(nums[q[0]])
        return ans
# 分块 + 预处理，O(n)

class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        prefixMax, suffixMax = [0] * n, [0] * n
        for i in range(n):
            if i % k == 0:
                prefixMax[i] = nums[i]
            else:
                prefixMax[i] = max(prefixMax[i - 1], nums[i])
        for i in range(n - 1, -1, -1):
            if i == n - 1 or (i + 1) % k == 0:
                suffixMax[i] = nums[i]
            else:
                suffixMax[i] = max(suffixMax[i + 1], nums[i])
        ans = [max(suffixMax[i], prefixMax[i + k - 1]) for i in range(n - k + 1)]
        return ans
# 27. 完全平方数，一个数最少能表示为几个平方数之和
# 数学算法，每个自然数都可以由最多四个平方数之和表示
class Solution:
    def isSquare(self, n: int) -> bool:
        sq = int(math.sqrt(n))
        return sq*sq == n

    def numSquares(self, n: int) -> int:
        # four-square and three-square theorems
        while (n & 3) == 0:
            n >>= 2      # reducing the 4^k factor from number
        if (n & 7) == 7: # mod 8
            return 4

        if self.isSquare(n):
            return 1
        # check if the number can be decomposed into sum of two squares
        for i in range(1, int(n**(0.5)) + 1):
            if self.isSquare(n - i*i):
                return 2
        # bottom case from the three-square theorem
        return 3
# 38. 最长上升子序列
# DP，O(n^2)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        dp = []
        for i in range(len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)
# 贪心+二分，贪心寻找上升最慢的数列，更新时使用二分查找，O(logn)
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        d = []
        for n in nums:
            if not d or n > d[-1]:
                d.append(n)
            else:
                l, r = 0, len(d) - 1
                loc = r
                while l <= r:
                    mid = (l + r) // 2
                    if d[mid] >= n:
                        loc = mid
                        r = mid - 1
                    else:
                        l = mid + 1
                d[loc] = n
        return len(d)
# 39. 删除无效括号，返回所有结果
# 搜索，首先统计左右括号需要去掉多少，然后决定每一步是否去掉左/右括号，注意增加右括号需要判定是否左>右，否则会产生无效解
class Solution:
    @staticmethod
    def DFS(i, lc, rc, lrm, rrm, s, n, path, result):
        while i < n and s[i] != '(' and s[i] != ')':
            path += s[i]
            i += 1
        if i == n:
            if lrm == 0 and rrm == 0:
                result.add(path)
            return
        if s[i] == '(':
            if lrm > 0:
                Solution.DFS(i+1, lc, rc, lrm-1, rrm, s, n, path, result)
            Solution.DFS(i+1, lc+1, rc, lrm, rrm, s, n, path+'(', result)
        elif s[i] == ')':
            if rrm > 0:
                Solution.DFS(i+1, lc, rc, lrm, rrm-1, s, n, path, result)
            if lc > rc:
                Solution.DFS(i+1, lc, rc+1, lrm, rrm, s, n, path+')', result)
        
    def removeInvalidParentheses(self, s: str) -> List[str]:
        lrm, rrm = 0, 0
        for c in s:
            if c == '(':
                lrm += 1
            elif c == ')':
                if lrm > 0:
                    lrm -= 1
                else:
                    rrm += 1
        result = set()
        path = ''
        Solution.DFS(0, 0, 0, lrm, rrm, s, len(s), path, result)
        return list(result)

