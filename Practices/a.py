from typing import List
import math

class Solution:
    def storeWater(bucket: List[int], vat: List[int]) -> int:
        n = len(bucket)
        cnt = []
        minCnt = 1e5+7

        for i in range(n):
            diVal = bucket[i] if bucket[i] != 0 else 1
            cnt.append(math.ceil(vat[i] / diVal))
            minCnt = min(minCnt, cnt[i])

        res = minCnt
        for i in range(n):
            tmp = bucket[i]
            while cnt[i] > minCnt:
                tmp += 1
                cnt[i] = vat[i] / tmp
                res += 1

        return res

    bucket1 = [9,0, 1]
    vat1 = [0, 2, 2]
    print(storeWater(bucket1, vat1))
