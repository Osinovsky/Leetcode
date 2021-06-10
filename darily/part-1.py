# 1. is power of 4?
# log
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        if n <= 0:
            return False
        return math.log(n, 4).is_integer()
# 如果 n 是 4 的幂，那么 n 的二进制表示中有且仅有一个 1，并且这个 1 出现在从低位开始的第偶数个二进制位上
# n & n-1 == 0 判断是否为 2 的幂
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0 and (n & 0xaaaaaaaa) == 0
# 4^x (mod3) = (1 + 3)^x (mod3) = 1 mod 3
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0 and n % 3 == 1
# 2. 找零, DP
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        D = [1] * (amount+1)
        for coin in coins:
            for i in range(coin, amount+1):
                D[i] += D[i-coin]
        return D[amount]