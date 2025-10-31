"""
Cuckoo, XOR, and Vacuum filter baselines for classical approximate membership.
"""
import numpy as np
import random

class CuckooFilter:
    def __init__(self, m, bucket_size=4, max_kicks=500):
        self.m = m
        self.bucket_size = bucket_size
        self.max_kicks = max_kicks
        self.buckets = [[] for _ in range(m)]

    def _hash(self, x, seed):
        return hash((x, seed)) % self.m

    def insert(self, x):
        i1 = self._hash(x, 0)
        i2 = self._hash(x, 1)
        for i in (i1, i2):
            if len(self.buckets[i]) < self.bucket_size:
                self.buckets[i].append(x)
                return True
        i = random.choice([i1, i2])
        for _ in range(self.max_kicks):
            if len(self.buckets[i]) < self.bucket_size:
                self.buckets[i].append(x)
                return True
            j = random.randrange(len(self.buckets[i]))
            x, self.buckets[i][j] = self.buckets[i][j], x
            i = self._hash(x, 0) if i == self._hash(x, 1) else self._hash(x, 1)
        return False

    def contains(self, x):
        i1 = self._hash(x, 0)
        i2 = self._hash(x, 1)
        return x in self.buckets[i1] or x in self.buckets[i2]

class XORFilter:
    # Placeholder: use a simple Bloom filter as a stand-in for now
    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.bits = [0] * m
        self.hashes = [lambda x, i=i: hash((x, i)) % m for i in range(k)]
    def insert(self, x):
        for h in self.hashes:
            self.bits[h(x)] = 1
    def contains(self, x):
        return all(self.bits[h(x)] for h in self.hashes)

class VacuumFilter:
    # Placeholder: use a counting Bloom filter as a stand-in for now
    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.counts = [0] * m
        self.hashes = [lambda x, i=i: hash((x, i)) % m for i in range(k)]
    def insert(self, x):
        for h in self.hashes:
            self.counts[h(x)] += 1
    def contains(self, x):
        return all(self.counts[h(x)] > 0 for h in self.hashes)
    def delete(self, x):
        for h in self.hashes:
            if self.counts[h(x)] > 0:
                self.counts[h(x)] -= 1
