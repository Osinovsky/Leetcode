# 51. 任务调度器，给一些任务，同样任务之间需要间隔n个时间，问最短多少时间能够执行完
# 模拟，O(task * |kind of task|)
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = collections.Counter(tasks)
        m = len(freq)
        nextValid = [1] * m
        rest = list(freq.values())
        time = 0
        for i in range(len(tasks)):
            time += 1
            minNextValid = min(nextValid[j] for j in range(m) if rest[j] > 0)
            time = max(time, minNextValid)
            best = -1
            for j in range(m):
                if rest[j] and nextValid[j] <= time:
                    if best == -1 or rest[j] > rest[best]:
                        best = j
            
            nextValid[best] = time + n + 1
            rest[best] -= 1
        return time
# 构造，O(task * |kind of task|)
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        freq = collections.Counter(tasks)
        maxExec = max(freq.values())
        maxCount = sum(1 for v in freq.values() if v == maxExec)
        return max((maxExec - 1) * (n + 1) + maxCount, len(tasks))
# 52. 回文子串数量
# Manacher 算法，O(n)
class Solution:
    def countSubstrings(self, s: str) -> int:
        t = '$#'
        for c in s:
            t += c
            t += '#'
        n = len(t)
        t += '!'
        f = [0] * n
        iMax, rMax, ans = 0, 0, 0
        for i in range(1, n):
            f[i] = min(rMax-i+1, f[2*iMax-i]) if i <= rMax else 1
            while t[i+f[i]] == t[i-f[i]]:
                f[i] += 1
            if i + f[i] - 1 > rMax:
                iMax = i
                rMax = i + f[i] - 1
            ans += (f[i] // 2)
        return ans
