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
# 14. 上台阶，一次可以上两个或者一个，求多少种走法
# DP, f(n) = f(n-1) + f(n-2), O(n)，超时
class Solution:
    def climbStairs(self, n: int) -> int:
        a, b = 1, 1
        for _ in range(n-1):
            a, b = b, a + b
        return b
# 快速幂，DP展示的公式可以由矩阵连乘表示，将中间的矩阵幂提取出来计算, O(logn)
class Solution:
    @staticmethod
    def mul(a, b):
        return [
            a[0] * b[0] + a[1] * b[2],
            a[0] * b[1] + a[1] * b[3],
            a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3]
        ]
    @staticmethod
    def pow(a, n):
        ret = [1, 0, 0, 1]
        while n > 0:
            if n & 1 == 1:
                ret = Solution.mul(ret, a)
            n >>= 1
            a = Solution.mul(a, a)
        return ret

    def climbStairs(self, n: int) -> int:
        q = [1, 1, 1, 0]
        res =  Solution.pow(q, n)
        return res[0]

# 通项公式，复杂度取决于 pow
class Solution:
    def climbStairs(self, n: int) -> int:
        sqrt5 = math.sqrt(5)
        fibn = math.pow((1 + sqrt5)/2, n+1) - math.pow((1 - sqrt5)/2, n+1)
        return round(fibn/sqrt5)
# 15. 编辑距离，一个单词经过至少几次增删改到达另一个单词
# 删相当于目标单词增加，因此三个动作：A增、B增、A改，规定只在单词的末尾增加或者改字母
# 因为增改的顺序无所谓，因此都规定在词末，方便
# DP, 二维数组 D[i][j] 表明 A[:i+1] 与 B[:j+1] 的编辑距离
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)
        if n * m == 0:
            return n + m
        D = [[0] * (m + 1) for _ in range(n + 1)]
        # 边界状态初始化
        for i in range(n + 1):
            D[i][0] = i # 对比空串，编辑距离就是增加若干字符
        for j in range(m + 1):
            D[0][j] = j
        # 计算所有 DP 值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = D[i - 1][j] + 1
                down = D[i][j - 1] + 1
                left_down = D[i - 1][j - 1] 
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                D[i][j] = min(left, down, left_down)
        return D[n][m]
# 15. 最小覆盖子串，给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。
# 滑动窗口
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        need = collections.defaultdict(int)
        tset = set(t)
        for c in tset:
            need[c] += 1
        needCnt = len(t)
        i = 0
        res = (0, float('inf'))
        for j, c in enumerate(s):
            if need[c] > 0:
                needCnt -= 1
            need[c] -= 1
            if needCnt == 0:       #步骤一：滑动窗口包含了所有T元素
                while True:      #步骤二：增加i，排除多余元素
                    c = s[i] 
                    if need[c] == 0:
                        break
                    need[c] += 1
                    i += 1
                if j - i < res[1] - res[0]:   #记录结果
                    res = (i, j)
                need[s[i]] += 1  #步骤三：i增加一个位置，寻找新的满足条件滑动窗口
                needCnt += 1
                i+=1
        return '' if res[1] > len(s) else s[res[0]:res[1]+1]

