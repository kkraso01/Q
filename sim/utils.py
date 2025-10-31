"""Utility functions for quantum data structures."""
import hashlib


def splitmix64(x, seed=0):
    """
    64-bit deterministic hash using splitmix algorithm.
    
    Args:
        x: Input value to hash
        seed: Seed for hash function independence
        
    Returns:
        64-bit hash value
    """
    z = (x + seed + 0x9e3779b97f4a7c15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def make_hash_functions(k):
    """
    Generate k independent deterministic hash functions.
    
    Args:
        k: Number of hash functions to generate
        
    Returns:
        List of k hash functions
    """
    return [lambda x, i=i: splitmix64(hash(x), seed=i) for i in range(k)]


def bitstring_to_int(s):
    """Convert bitstring to integer for hashing."""
    if isinstance(s, bytes):
        return int.from_bytes(s, byteorder='big')
    elif isinstance(s, str):
        return int.from_bytes(s.encode('utf-8'), byteorder='big')
    return s
