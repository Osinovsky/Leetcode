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
