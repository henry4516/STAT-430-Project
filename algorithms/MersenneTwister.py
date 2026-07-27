class SimpleMT19937:
    '''
    Mersenne Twister algorithm.

    '''
    # MT19937 parameters
    N = 624
    M = 397
    MATRIX_A = 0x9908B0DF
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7FFFFFFF
    MASK_32 = 0xFFFFFFFF

    def __init__(self, seed: int):
        self.state = [0] * self.N
        self.index = self.N

        # state array initialization
        self.state[0] = seed & self.MASK_32

        for i in range(1, self.N):
            previous = self.state[i - 1]
            self.state[i] = (
                1812433253 * (previous ^ (previous >> 30)) + i
            ) & self.MASK_32

    def twist(self) -> None:
        '''
        Generate the next state.

        '''
        for i in range(self.N):
            # Join the highest bit of state[i] with the lower
            # 31 bits of state[i + 1]
            x = (
                (self.state[i] & self.UPPER_MASK)
                | (self.state[(i + 1) % self.N] & self.LOWER_MASK)
            )

            x_shifted = x >> 1

            # apply MATRIX_A when x is odd
            if x & 1:
                x_shifted ^= self.MATRIX_A

            self.state[i] = (
                self.state[(i + self.M) % self.N] ^ x_shifted
            ) & self.MASK_32

        self.index = 0

    def random_uint32(self) -> int:
        '''
        Return one pseudorandom unsigned 32-bit integer.

        '''
        
        if self.index >= self.N:
            self.twist()

        y = self.state[self.index]
        self.index += 1

        # Tempering transformation
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18

        return y & self.MASK_32

    def random_1_to_1024(self) -> int:
        '''
        Map the output to [1,1024].

        '''
        y = self.random_uint32()
        return (y >> 22) + 1

    def generate(self, length: int) -> list[int]:
        '''
        Generate a sequence of the requested length.

        '''

        return [self.random_1_to_1024() for _ in range(length)]
    

if __name__ == '__main__':
    import numpy as np
    
    matrix = np.zeros((1000, 1500), dtype=np.int32)
    for i in np.arange(1000):
        seed = i + 1
        matrix[i] = SimpleMT19937(seed).generate(1500)

    header = ','.join(['No.' + str(index + 1) for index in np.arange(1500)])
    np.savetxt('430_Mersenne_1500.csv', matrix, delimiter=',', fmt='%d', header=header)

