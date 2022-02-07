class Solution:

    SYMBOL_VALUES = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000,
    }

    def romanToInt(self, s: str):
        ans = 0
        n = len(s)
        for i in range(n):
            value = self.SYMBOL_VALUES[s[i]]
            if i < n - 1:
                value_next = self.SYMBOL_VALUES[s[i + 1]]
            if i < n - 1 and value < value_next:
                ans -= value
            else:
                ans += value

        return ans
