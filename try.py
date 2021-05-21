import collections

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        distance = {}
        num = n = -1
        for i in range(len(s)):
            if distance.get(s[i], -1) >= n:
                n = distance.get(s[i], -1)
            distance[s[i]] = i

            if i - n > num:
                num = i - n
        return num

s = Solution()
print(s.lengthOfLongestSubstring("ohomn"))
