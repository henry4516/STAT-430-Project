class Xorshift32:
    '''
    Simple 32-bit xorshift pseudorandom number generator.

    The internal state is a 32-bit integer.
    Outputs are mapped uniformly to [1, 1024].
    '''

    MASK_32 = 0xFFFFFFFF

    def __init__(self, seed: int):
        if seed == 0:
            raise ValueError("Xorshift32 requires a nonzero seed.")

        self.state = seed & self.MASK_32

    def next_uint32(self) -> int:
        x = self.state

        x ^= (x << 13) & self.MASK_32
        x ^= x >> 17
        x ^= (x << 5) & self.MASK_32

        self.state = x & self.MASK_32
        return self.state

    def next_integer(self) -> int:
        '''
        Return an integer in [1, 1024].
        
        '''
        return (self.next_uint32() >> 22) + 1

    def generate(self, length: int) -> list[int]:
        '''
        Generate the sequence given seed.

        '''
        return [self.next_integer() for _ in range(length)]
    


if __name__ == '__main__':
    import numpy as np
    
    matrix = np.zeros((1000, 1500), dtype=np.int32)
    for i in np.arange(1000):
        seed = i + 1
        matrix[i] = Xorshift32(seed).generate(1500)

    header = ','.join(['No.' + str(index + 1) for index in np.arange(1500)])
    np.savetxt('430_XOR_1500.csv', matrix, delimiter=',', fmt='%d', header=header)

