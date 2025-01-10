from functools import lru_cache

import chess

@lru_cache
def ctzll(x):
    return (x & -x).bit_length() - 1

def unpack_bits(packed_data, total_bits):
    unpacked = []
    for byte in packed_data:
        for i in range(7, -1, -1):
            unpacked.append((byte >> i) & 1)
    return unpacked[:total_bits]

def pack_bits(data):
    packed = bytearray()

    for i in range(0, len(data), 8):
        byte = 0
        for bit in data[i:i + 8]:
            byte = (byte << 1) | bit
        packed.append(byte)
    return packed