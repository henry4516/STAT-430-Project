import numpy as np

# relations: the relation between x_(n-1) and x_(n).
def LCG(x):
    '''
    Relation: x_(n) = a * x_(n-1) + c (mod m)
    '''

    # parameters
    m = 2**10
    a = 205
    c = 1
    state = x - 1
    next = (a * state + c) % m

    return next + 1 # +1 to be between 1 and 1024

def BinaryShift(x):
    MASK = (1 << 10) - 1  # 1023

    x ^= (x << 1) & MASK
    x ^= x >> 5
    x ^= (x << 2) & MASK
    x = x & MASK

    return x + 1

def DecimalMethod(x):
    y = np.sqrt(x * 133 + 29)
    y /= 177
    z = int(y * 23198891881357) % 1024
    return z + 1

class RNG:
    '''
    We redesigned the logics. Since a seed gives a unique sequence, we don't need to repeat for lengths 100, 500, 1500 -
    just generate sequences of length 1500 and let them pick the length they want (5 levels).
    For seeds, we apply algorithms on 1-1000, and let them pick what they want (4 levels)
    We also ignored the separate U01 version - they can be obtained from division by 1024
    '''
    def __init__(self, name, length=1500, seedMax=1000):
        self.name = name
        self.length = length
        self.seedMax = seedMax

    def generator(self, f, seed):
        '''
        Recursively generates a random sequence given seed, with relation f such that
        f(x_(n-1))=f(x_(n))
        '''
        
        randseq = np.zeros(self.length, dtype=np.int32)

        # beginning of steps
        x_old = seed
        randseq[0] = seed # raw integer seed

        # generators
        for i in np.arange(1, self.length):
            x_new = f(x_old)
            randseq[i] = x_new
            x_old = x_new

        return randseq
    
    def sequences(self, f, algorithm='algorithm'):
        '''
        Generates a matrix based on all seeds 1-1000 such that:
        1. each row is a random sequence based on a seed, whose first one is the seed
        2. each column is a random number in sequence at that row
        3. level 0 is raw integer sequences; level 1 is the corresponding decimal sequences.
        3. size is seedList_length x generator_length x 2

        matrix saved to savePath
        '''

        matrix = np.zeros((self.seedMax, self.length), dtype=np.int32)
        savePath = self.name + '_' + algorithm + '_' + str(self.length) + '.csv'
        header = 'Seed,' + ','.join(['No.' + str(index) for index in np.arange(2, self.length + 1)])

        for i in np.arange(0, self.seedMax):
            seed = i + 1
            matrix[i] = self.generator(f, seed)

        np.savetxt(savePath, matrix, delimiter=',', fmt='%d', header=header)

        return matrix
    

if __name__ == '__main__':
    rng = RNG('430')
    #rng.sequences(LCG, 'LCG')
    #rng.sequences(BinaryShift, 'Mersenne')
    #rng.sequences(DecimalMethod, 'Decimal')
