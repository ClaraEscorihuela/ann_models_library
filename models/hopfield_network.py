import numpy as np


class HopfieldNetwork:

    def __init__(self):
        self.W = None
        self.positions_to_update = []

    # Define a function for computing the weights matrix
    def train(self, patterns, synchronous=False, check=True):
        """
        Crete weight matrix
        :param patterns: base vectors to create the matrix
        :param synchronous: boolean for the sincronious/asncronious update of the patterns
        :param check: boolean to check if the patterns are stored/ fixed points in the matrix
        :return: number of patterns stored in the matrix, should be equal to number of patterns
        """

        # Create weigt matrix
        self.W = np.sum([p.reshape((-1, 1)) @ p.reshape((1, -1)) for p in patterns], axis=0)
        self.W = self.W.astype(np.float64).copy()
        self.W /= self.W.shape[0]


        # Check if all patterns are well stored by recalling (if required)
        if check:
            yes = 0
            for pattern in patterns:
                recalled, _ = self.recall(pattern, synchronous=synchronous)
                if np.array_equal(recalled, pattern):
                    yes += 1
                else:
                    print("Warning: input pattern is not a fixed point ", pattern, recalled)

            print(yes, " patterns memorized")
            return yes


    def update(self, pattern, synchronous=False, seed=0):
        """
        Update a new vector in syncronious or asyncronious mode
        :param pattern: vector
        :param synchronous: boolean
        :param seed: 0
        :return: new pattern
        """
        if synchronous:
            # All elements from the pattern are updated at the same time
            pattern_update = np.sign(self.W @ pattern)
            pattern_update[pattern_update == 0] = 1
        else:
            # List all patterns dimensions not updated yet
            if not self.positions_to_update:
                self.positions_to_update = [i for i in range(pattern.shape[0])]

            # Select randomly one position to update
            np.random.seed(seed)
            j = np.random.choice(self.positions_to_update)

            # Update the chosen position
            pattern_update = pattern.copy()
            value = np.sign(np.sum(self.W[j, :] * pattern))
            pattern_update[j] = 1 if value >= 0 else -1
            self.positions_to_update.remove(j)

        return pattern_update

    def recall(self, pattern, synchronous=False, max_iterations=1000):
        """
        Complete update depending on number of iterations
        :param pattern vector
        :param synchronious boolean for update mode
        :param max_iterations number of maximum iterations
        """
        iteration = 0
        while (iteration < max_iterations):
            if synchronous:
                pattern_update = self.update(pattern, synchronous=True)
            else:
                pattern_update = pattern
                for _ in range(pattern.shape[0]):
                    # Apply the function as many times as the update of the pattern
                    pattern_update = self.update(pattern_update, synchronous=False)
            if np.array_equal(pattern_update, pattern):
                break
            else:
                iteration += 1
                pattern = pattern_update

        return pattern, iteration


    def test_recall(self, p_distorted, p_real, synchronous=False):
        pattern_recall,_ = self.recall(p_distorted, synchronous=synchronous)
        sol = np.array_equal(pattern_recall, p_real)

        text = 'Yay!' if sol else 'Bad'
        print(f'pat dist = {p_distorted}, pat recall{pattern_recall}: {text}')

        return sol

