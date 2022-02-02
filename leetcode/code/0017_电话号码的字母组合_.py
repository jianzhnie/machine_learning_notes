import itertools
from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return list()

        phoneMap = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz',
        }

        groups = (phoneMap[digit] for digit in digits)
        return [
            ''.join(combination) for combination in itertools.product(*groups)
        ]


if __name__ == '__main__':
    solution = Solution()
    input = ['2', '6', '8']
    ans = solution.letterCombinations(input)
    print(ans)
