# 41. 戳气球，每戳一个获得本气球*左气球*右气球个金币，没有记作1
# DP，区间内最后一个气球肯定只有自己，以他为中间，分为左右子区间
class Solution:
    def maxCoins(self, nums: List[int]) -> int:
        nums.insert(0,1)
        nums.insert(len(nums),1)
        store = [[0]*(len(nums)) for i in range(len(nums))]
        def range_best(i,j):
            m = 0 
            for k in range(i+1,j):
                left = store[i][k]
                right = store[k][j]
                a = left + nums[i]*nums[k]*nums[j] + right
                if a > m:
                    m = a
            store[i][j] = m
        for n in range(2,len(nums)):
            for i in range(0,len(nums)-n):
                range_best(i,i+n)
        return store[0][len(nums)-1]
# 42. 零钱兑换
# 其中lru_cache或者cache是缓存相同参数得函数
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        @functools.lru_cache(amount)
        def dp(rem) -> int:
            if rem < 0: return -1
            if rem == 0: return 0
            mini = int(1e9)
            for coin in self.coins:
                res = dp(rem - coin)
                if res >= 0 and res < mini:
                    mini = res + 1
            return mini if mini < int(1e9) else -1

        self.coins = coins
        if amount < 1: return 0
        return dp(amount)
# 43. 比特位计数，给定非负整数，记录所有比它小或等于它的非负整数的二进制形式中有多少1
# Brian Kernighan 算法，直接计算，O(nlogn)
# x = x & x - 1 将二进制最后一个1变成0
class Solution:
    def countBits(self, n: int) -> List[int]:
        def countOnes(x: int) -> int:
            ones = 0
            while x > 0:
                x &= (x - 1)
                ones += 1
            return ones
        bits = [countOnes(i) for i in range(n + 1)]
        return bits
# DP 最高有效位，可以根据比i小的数进行DP, O(n)
# 对 x 如果知道最大的 y 使得 y \leq x 且 y 是 2 的整数次幂，则称 y 是 x 的最高有效位
# z = x - y 则有 dp[x] = dp[z] + 1。而判断2的幂，当且仅当 y & y - 1 == 0
class Solution:
    def countBits(self, n: int) -> List[int]:
        bits = [0]
        highBit = 0
        for i in range(1, n + 1):
            if i & (i - 1) == 0:
                highBit = i
            bits.append(bits[i - highBit] + 1)
        return bits
# DP 最低有效位, O(n)
# 偶数 x, d[x] = d[x//2]，奇数 x, d[x] = d[x//2] + 1
class Solution:
    def countBits(self, n: int) -> List[int]:
        bits = [0]
        for i in range(1, n + 1):
            bits.append(bits[i >> 1] + (i & 1))
        return bits
# DP 最低设置位，O(n)
# y = x & x - 1，d[x] = d[y] + 1
class Solution:
    def countBits(self, n: int) -> List[int]:
        bits = [0]
        for i in range(1, n + 1):
            bits.append(bits[i & (i - 1)] + 1)
        return bits
# 44.  除法求值, 给一些分数，求查询的分数值，没有则返回-1 （路径正反向搜索或者并查集）
# 并查集
class UnionFind:
    def __init__(self):
        self.father = {}
        self.value = {}
    
    def find(self,x):
        root = x
        base = 1
        while self.father[root] != None:
            root = self.father[root]
            base *= self.value[root]
        while x != root:
            original_father = self.father[x]
            self.value[x] *= base
            base /= self.value[original_father]
            self.father[x] = root
            x = original_father
        return root
    
    def merge(self,x,y,val):
        root_x,root_y = self.find(x),self.find(y)
        if root_x != root_y:
            self.father[root_x] = root_y
            self.value[root_x] = self.value[y] * val / self.value[x]

    def is_connected(self,x,y):
        return x in self.value and y in self.value and self.find(x) == self.find(y)
    
    def add(self,x):
        if x not in self.father:
            self.father[x] = None
            self.value[x] = 1.0
class Solution:
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        uf = UnionFind()
        for (a,b),val in zip(equations,values):
            uf.add(a)
            uf.add(b)
            uf.merge(a,b,val)
        res = [-1.0] * len(queries)
        for i,(a,b) in enumerate(queries):
            if uf.is_connected(a,b):
                res[i] = uf.value[a] / uf.value[b]
        return res
# 45. 根据身高重建队列，给数列[[h, k]], h为当前身高，k 为前面有多少高于当前身高的，重建队列
# 从低到高考虑，O(n^2)
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (x[0], -x[1]))
        n = len(people)
        ans = [[] for _ in range(n)]
        for person in people:
            spaces = person[1] + 1
            for i in range(n):
                if not ans[i]:
                    spaces -= 1
                    if spaces == 0:
                        ans[i] = person
                        break
        return ans
# 从高到低考虑，矮的人不会影响前面高的排好的顺序，因此可以插入，O(n^2)
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        people.sort(key=lambda x: (-x[0], x[1]))
        n = len(people)
        ans = list()
        for person in people:
            ans[person[1]:person[1]] = [person]
        return ans
# 46. 分割等和子集，给定一组数，问能否正好分割成两个和相等的集合
# 0-1背包问题，恰好等于一半，O(n target)
# dij 指的是[0,...,i]满足和为j，当 j >= ni 时 dij = di-1j or di-1j-ni, j < ni 时，dij = di-1j
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        if n < 2:
            return False
        total = sum(nums)
        maxNum = max(nums)
        if total & 1:
            return False
        target = total // 2
        if maxNum > target:
            return False
        dp = [[0] * (target + 1) for _ in range(n)]
        for i in range(n):
            dp[i][0] = True
        dp[0][nums[0]] = True
        for i in range(1, n):
            num = nums[i]
            for j in range(1, target + 1):
                if j >= num:
                    dp[i][j] = dp[i - 1][j] | dp[i - 1][j - num]
                else:
                    dp[i][j] = dp[i - 1][j]
        return dp[n - 1][target]
# 47. 路径总和 III，一棵树上找到路径之和为目标的所有路径个数
# DFS，维持所有sumList
class Solution:
    def pathSum(self, root: TreeNode, sum: int) -> int:
        def dfs(root, sumlist):
            if root is None: return 0
            sumlist = [num + root.val for num in sumlist] + [root.val]
            return sumlist.count(sum) + dfs(root.left, sumlist) + dfs(root.right, sumlist)
        return dfs(root, [])
# 48. 目标和，给定数列，加入加减号，问有多少种方法达到目标和
# 动态规划，O(n*(sum-target))
# sum = neg + pos, pos - neg = target => neg = (sum - target) // 2
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        d = sum(nums) - target
        if d < 0 or d % 2 != 0:
            return 0
        neg = d // 2
        dp = [0] * (neg+1)
        dp[0] = 1
        for n in nums:
            for j in range(neg, n-1, -1):
                dp[j] += dp[j-n]
        return dp[neg]
