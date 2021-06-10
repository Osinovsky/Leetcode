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
