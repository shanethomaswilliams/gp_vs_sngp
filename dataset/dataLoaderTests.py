import unittest
from makeDataLoaders import load_data_test, load_data_train
import torch

class test_loaders(unittest.TestCase):

    #Tests is the random seed actually yeilds reproducably train data sets
    def test_RndSeed_train(self):
        GP_train_A, GP_val_A, SNGP_train_A, SNGP_val_A = load_data_train("Data/Noisy/Sin",
                                                                         10_000,
                                                                         random_state=42,
                                                                         shuffle=False)
        GP_train_B, GP_val_B, SNGP_train_B, SNGP_val_B = load_data_train("Data/Noisy/Sin",
                                                                         10_000,
                                                                         random_state=42,
                                                                         shuffle=False)
        
        #Test equality for GP Train
        for x_A, y_A in GP_train_A:
            for x_B, y_B in GP_train_B:
                self.assertTrue(torch.equal(x_A, x_B))
                self.assertTrue(torch.equal(y_A, y_B))

        #Test equality for GP Val
        for x_A, y_A in GP_val_A:
            for x_B, y_B in GP_val_B:
                self.assertTrue(torch.equal(x_A, x_B))
                self.assertTrue(torch.equal(y_A, y_B))

        #Test equality for SNGP train
        for x_A, y_A in SNGP_train_A:
            for x_B, y_B in SNGP_train_B:
                self.assertTrue(torch.equal(x_A, x_B))
                self.assertTrue(torch.equal(y_A, y_B))

        #Test equality for SNGP Val
        for x_A, y_A in SNGP_val_A:
            for x_B, y_B in SNGP_val_B:
                self.assertTrue(torch.equal(x_A, x_B))
                self.assertTrue(torch.equal(y_A, y_B))

    #Test if the dataLoaders for GP and SNGP contain the same data for train load
    def test_SNGP_GP_Loads_equals_train(self):
        GP_train, GP_val, SNGP_train, SNGP_val = load_data_train("Data/Noisy/Sin",
                                                                  10_000,
                                                                  random_state=12345,
                                                                  shuffle=False)
        
        for x_GP, y_GP in GP_train:
            for x_SNGP, y_SNGP in SNGP_train:
                #data values in them should be the same
                self.assertTrue(torch.equal(x_GP, x_SNGP))
                #Have to add an extra dimension to SNGP y
                self.assertTrue(torch.equal(y_GP,y_SNGP.unsqueeze(-1)))

        for x_GP, y_GP in GP_val:
            for x_SNGP, y_SNGP in SNGP_val:
                #data values in them should be the same
                self.assertTrue(torch.equal(x_GP, x_SNGP))
                #Have to add an extra dimension to SNGP y
                self.assertTrue(torch.equal(y_GP,y_SNGP.unsqueeze(-1)))
                
    #Test that different seeds yeild different results
    def test_diff_seed_train(self):
        GP_train_A, GP_val_A, SNGP_train_A, SNGP_val_A = load_data_train("Data/Noisy/Sin",
                                                                         10_000,
                                                                         random_state=42)
        GP_train_B, GP_val_B, SNGP_train_B, SNGP_val_B = load_data_train("Data/Noisy/Sin",
                                                                         10_000,
                                                                         random_state=6789)
        
        #Test inequality for GP Train
        for x_A, y_A in GP_train_A:
            for x_B, y_B in GP_train_B:
                self.assertTrue(not torch.equal(x_A, x_B))
                self.assertTrue(not torch.equal(y_A, y_B))

        #Test inequality for GP Val
        for x_A, y_A in GP_val_A:
            for x_B, y_B in GP_val_B:
                self.assertTrue(not torch.equal(x_A, x_B))
                self.assertTrue(not torch.equal(y_A, y_B))

        #Test inequality for SNGP train
        for x_A, y_A in SNGP_train_A:
            for x_B, y_B in SNGP_train_B:
                self.assertTrue(not torch.equal(x_A, x_B))
                self.assertTrue(not torch.equal(y_A, y_B))

        #Test inequality for SNGP Val
        for x_A, y_A in SNGP_val_A:
            for x_B, y_B in SNGP_val_B:
                self.assertTrue(not torch.equal(x_A, x_B))
                self.assertTrue(not torch.equal(y_A, y_B))

    #Tests is the random seed actually yeilds reproducably test data sets
    def test_Rndseed_testSet(self):
        GP_test_A, SNGP_test_A = load_data_test("Data/Clean/CrazySin",
                                                10_000,
                                                random_state=1337,
                                                shuffle=False)
        GP_test_B, SNGP_test_B = load_data_test("Data/Clean/CrazySin",
                                                10_000,
                                                random_state=1337,
                                                shuffle=False)
        
        for x_A, y_A in GP_test_A:
            for x_B, y_B in GP_test_B:
                self.assertTrue(torch.equal(x_A,x_B))
                self.assertTrue(torch.equal(y_A,y_B))

        for x_A, y_A in SNGP_test_A:
            for x_B, y_B in SNGP_test_B:
                self.assertTrue(torch.equal(x_A,x_B))
                self.assertTrue(torch.equal(y_A,y_B))
    
    #Test if the dataLoaders for GP and SNGP contain the same data for test load
    def test_SNGP_GP_Loads_equals_testSet(self):
        GP_test, SNGP_test = load_data_test("Data/Clean/CrazySin",
                                            10_000,
                                            random_state=54321,
                                            shuffle=False)
        
        for x_GP, y_GP in GP_test:
            for x_SNGP, y_SNGP in SNGP_test:
                self.assertTrue(torch.equal(x_GP,x_SNGP))
                self.assertTrue(torch.equal(y_GP,y_SNGP.unsqueeze(-1)))

    #Test that different seeds yeild different results
    def test_diff_seed_testSet(self):
        GP_test_A, SNGP_test_A = load_data_test("Data/Clean/CrazySin",
                                                10_000,
                                                random_state=123,
                                                shuffle=False)
        GP_test_B, SNGP_test_B = load_data_test("Data/Clean/CrazySin",
                                                10_000,
                                                random_state=789,
                                                shuffle=False)
        
        for x_A, y_A in GP_test_A:
            for x_B, y_B in GP_test_B:
                self.assertTrue(not torch.equal(x_A,x_B))
                self.assertTrue(not torch.equal(y_A,y_B))

        for x_A, y_A in SNGP_test_A:
            for x_B, y_B in SNGP_test_B:
                self.assertTrue(not torch.equal(x_A,x_B))
                self.assertTrue(not torch.equal(y_A,y_B))
        

if __name__ == '__main__':
    unittest.main()
        