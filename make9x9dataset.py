import numpy as np


class Kernel:

    class Filter():
        def __init__(self,name,filter_list=None, dist=1):
            self.name = name
            self.filter_list = filter_list if filter_list else []
            self.dist = dist

        def add_filter(self,filter_):
            self.filter_list.append(filter_)

        def get_filter_score(self,matrix):
            scores = np.array([self.get_minikowsky_distance(filter_,matrix) for filter_ in self.filter_list])
            return np.min(scores)

        def get_minikowsky_distance(self,a,b):
            p = self.dist

            if p == None:
                p = 2

            a = a.reshape(1,9)
            b = b.reshape(1,9)

            return np.sum(np.abs(a - b)**p, axis=1)**(1/p)


    def __init__(self):
        self.filter_set = self.generate_filter_set()

    def generate_filter_set(self):
        # empty filter
        empty = self.Filter('empty')
        empty.add_filter(np.zeros(9,dtype='int').reshape(3,3))

        # full filter
        full = self.Filter('full')
        full.add_filter(np.ones(9,dtype='int').reshape(3,3))
        full.add_filter(np.array([
            [1,0,1],
            [0,1,0],
            [1,0,1]]))

        full.add_filter(np.array([
            [0,1,0],
            [1,0,1],
            [0,1,0]]))
        
        # horizontal and vertical fill matrix
        left_vertical = self.Filter('left_vertical')
        right_vertical = self.Filter('right_vertical')
        center_vertical = self.Filter('center_vertical')
        top_horizontal = self.Filter('top_horizontal')
        bottom_horizontal = self.Filter('bottom_horizontal')
        center_horizontal = self.Filter('center_horizontal')

        filter_set = [] 

        for i in range(2):
            matrix = np.zeros(9,dtype='int').reshape(3,3)
            matrix[i*2] = np.ones(3,dtype='int')
            filter_set.append(matrix)
            new_matrix = np.rot90(matrix)
            filter_set.append(new_matrix)

        for i in range(2):
            matrix = np.zeros(9,dtype='int').reshape(3,3)
            matrix[i] = np.ones(3,dtype='int')
            matrix[i+1] = np.ones(3,dtype='int')
            filter_set.append(matrix)
            new_matrix = np.rot90(matrix)
            filter_set.append(new_matrix)

        left_vertical.add_filter(filter_set[1])
        left_vertical.add_filter(filter_set[5])

        right_vertical.add_filter(filter_set[3])
        right_vertical.add_filter(filter_set[7])

        top_horizontal.add_filter(filter_set[0])
        top_horizontal.add_filter(filter_set[4])

        bottom_horizontal.add_filter(filter_set[2])
        bottom_horizontal.add_filter(filter_set[6])

        center_vertical.add_filter(np.array([
            [0,1,0],
            [0,1,0],
            [0,1,0]]))
        center_horizontal.add_filter(np.array([
            [0,0,0],
            [1,1,1],
            [0,0,0]]))

        # diagonal fill large matrix
        bottom_left_diagonal = self.Filter('bottom_left_diagonal')
        bottom_right_diagonal = self.Filter('bottom_right_diagonal')
        top_left_diagonal = self.Filter('top_left_diagonal')
        top_right_diagonal = self.Filter('top_right_diagonal')
        center_diagonal_up = self.Filter('center_diagonal_up')
        center_diagonal_down = self.Filter('center_diagonal_down')

        filter_set = []

        matrix = np.array([
            [1,1,0],
            [1,0,0],
            [0,0,0]])

        for i in range(4):
            filter_set.append(matrix)
            matrix = np.rot90(matrix)

        matrix = np.array([
            [1,1,1],
            [1,1,0],
            [1,0,0]])

        for i in range(4):
            filter_set.append(matrix)
            matrix = np.rot90(matrix)

        bottom_left_diagonal.add_filter(filter_set[1])
        bottom_left_diagonal.add_filter(filter_set[5])
        
        bottom_right_diagonal.add_filter(filter_set[2])
        bottom_right_diagonal.add_filter(filter_set[6])

        top_left_diagonal.add_filter(filter_set[0])
        top_left_diagonal.add_filter(filter_set[4])

        top_right_diagonal.add_filter(filter_set[3])
        top_right_diagonal.add_filter(filter_set[7])

        center_diagonal_up.add_filter(np.array([
            [0,0,1],
            [0,1,0],
            [1,0,0]]))
        center_diagonal_down.add_filter(np.array([
            [1,0,0],
            [0,1,0],
            [0,0,1]]))



        filter_set = []

        filter_set.append(empty)
        filter_set.append(full)

        filter_set.append(center_vertical)
        filter_set.append(center_horizontal)
        filter_set.append(center_diagonal_up)
        filter_set.append(center_diagonal_down)

        filter_set.append(left_vertical)
        filter_set.append(right_vertical)
        filter_set.append(top_horizontal)
        filter_set.append(bottom_horizontal)

        filter_set.append(bottom_left_diagonal)
        filter_set.append(bottom_right_diagonal)
        filter_set.append(top_left_diagonal)
        filter_set.append(top_right_diagonal)

        return filter_set

    def get_classification(self, number):
        return self.filter_set[number].name

    def apply_transformation(self, matrix):

        new_matrix = (matrix >= 50).astype('int')

        min_score = self.filter_set[0].get_filter_score(new_matrix)
        min_index = 0

        for i, filter_ in enumerate(self.filter_set):
            score = filter_.get_filter_score(new_matrix)
            if score <= min_score:
                min_score = score
                min_index = i

        return min_index

    def transform(self, matrix):
        # reshape the original matrix into a square
        cells = matrix.shape
        dimension = int(cells[0]**(1/2))
        square_matrix = matrix.reshape(dimension,dimension)

        # make the square sidelengths divisuble by three
        extra_dimension = dimension % 3
        dimension -= extra_dimension

        for d in range(extra_dimension):
            square_matrix = np.delete(square_matrix,0,0)
            square_matrix = np.delete(square_matrix,0,1)

        # make a new list to hold data
        new_matrix = np.zeros(int(dimension/3)**2, dtype='int')
        
        # fill matrix with transformed values
        for row in range(int(dimension/3)):
            for column in range(int(dimension/3)):

                # create a submatrix
                sub_matrix = np.array([
                        square_matrix[row*3    ][column*3:column*3 + 3],
                        square_matrix[row*3 + 1][column*3:column*3 + 3],
                        square_matrix[row*3 + 2][column*3:column*3 + 3]])

                # apply filter and save result in new matrix
                new_matrix[column + row*9] = self.apply_transformation(sub_matrix)

        return new_matrix

    def convert_to_string(self, matrix):
        classification_symbols = {
                0 :' ',
                1 :'#',
                2 :'|',
                3 :'=',
                4 :'/',
                5 :'\\',
                6 :'|',
                7 :'|',
                8 :'=',
                9 :'_',
                10:'\\',
                11:'/',
                12:'/',
                13:'\\'}

        return [classification_symbols[x] for x in matrix]
        
# dataset to write to a file
def main():
    import pandas as pd
    import numpy as np
    import time

    kernel = Kernel()
    
##########################################################
#    # uncomment for training data
#    train_path = '../Digit-Recognition-Tree/Data/train.csv'
#    train_data = pd.read_csv(train_path)
#    train_data.dropna(inplace=True)
#    
#    X_train = train_data.drop(columns=['label']).values
#    Y_train = train_data['label'].values
#
#    compressed = np.zeros((42000,81),dtype='int')
#
#    print(X_train.shape)
#
#    start = time.time()
#    for i, row in enumerate(X_train):
#        compressed[i] = kernel.transform(row)
#        if i % 420 == 0:
#            print(f'{i/420:>4}% complete')
#    stop = time.time()
#
#    all_data = np.concatenate((np.array(train_data.iloc[:, 0])[:, np.newaxis], compressed), axis=1)
#    np.savetxt("compressed_train.csv", all_data, delimiter=",",fmt='%d')
#
#############################################################
    # uncomment for testing data
    test_path = '../Digit-Recognition-Tree/Data/test.csv'
    test_data = pd.read_csv(test_path)
    test_data.dropna(inplace=True)

    X_test = test_data.values

    compressed = np.zeros((28000,81),dtype='int')
    
    print(test_data.shape)

    start = time.time()
    for i, row in enumerate(X_test):
        compressed[i] = kernel.transform(row)
        if i % 280 == 0:
            print(f'{i/280:>4}% complete')
    stop = time.time()

    np.savetxt("compressed_train.csv", compressed, delimiter=",",fmt='%d')


if __name__ == '__main__':
    main()
