# evaluate randomness of each sequence

import numpy as np
import RandTest as rt

class RandEva:
    def __init__(self, matrix):
        self.n_seeds, self.length = matrix.shape
        self.matrix = matrix

    def evaluate(self, length):
        seqs = self.matrix[:,:length]

        seeds = np.arange(1, 1001)
        length_col = np.repeat(length, 1000)

        # test scores
        entropy = np.apply_along_axis(rt.entropy, 1, seqs)
        runs = np.apply_along_axis(rt.runsTest, 1, seqs)
        pi = np.apply_along_axis(rt.piTest, 1, seqs)
        acf = np.apply_along_axis(rt.acfTest, 1, seqs)
        uni = np.apply_along_axis(rt.uniformityTest, 1, seqs)
        rep = np.apply_along_axis(rt.RepetitionScore, 1, seqs)

        columns = [seeds, length_col, entropy, runs, pi, acf, uni, rep]
        result = np.zeros((1000, len(columns)), dtype=np.float16)
        for j in range(len(columns)):
            result[:, j] = columns[j]

        print(f'length {length} done')
        return result
    
    def tests(self, savePath):
        '''
        savePath is False if no printing or str as save path
        '''

        print(f'Processing {savePath}...')

        lengths = [300,600,900,1200,1500]
        matrices = [self.evaluate(length) for length in lengths]

        final = np.concatenate(matrices, axis=0)

        if isinstance(savePath, str):
            header = 'seed,length,entropy,runs,pi,acf,uni,rep'
            np.savetxt(savePath + '.csv', final, delimiter=',', fmt='%f', header=header)

        return final

if __name__ == '__main__':
    LCG_seqs = np.loadtxt('430_LCG_1500.csv', delimiter=',', skiprows=1, dtype=np.int32)
    RANDU_seqs = np.loadtxt('430_RANDU_1500.csv', delimiter=',', skiprows=1, dtype=np.int32)
    MT_seqs = np.loadtxt('430_Mersenne_1500.csv', delimiter=',', skiprows=1, dtype=np.int32)
    XOR_seqs = np.loadtxt('430_XOR_1500.csv', delimiter=',', skiprows=1, dtype=np.int32)

    RandEva(LCG_seqs).tests('LCG_scores')
    RandEva(RANDU_seqs).tests('RANDU_scores')
    RandEva(MT_seqs).tests('MT_scores')
    RandEva(XOR_seqs).tests('XOR_scores')
