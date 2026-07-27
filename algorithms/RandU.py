class RandU:
    a = 65539
    m = 2**31

    def __init__(self, seed: int):
        self.state = seed % self.m
        if self.state == 0:
            self.state = 1

    def next_int(self):
        '''
        Updates current state.

        '''
        self.state = (self.a * self.state) % self.m
        return self.state

    def next_mapped(self):
        '''
        Updates and map to number in [1,1024].
        '''
        return (self.next_int() >> 21) + 1

    def generate(self, length):
        return [self.next_mapped() for _ in range(length)]


if __name__ == '__main__':
    import numpy as np
    matrix = np.zeros((1000, 1500), dtype=np.int32)
    for i in np.arange(1000):
        seed = i + 1
        matrix[i] = RandU(seed).generate(1500)

    header = ','.join(['No.' + str(index + 1) for index in np.arange(1500)])
    np.savetxt('430_RANDU_1500.csv', matrix, delimiter=',', fmt='%d', header=header)
