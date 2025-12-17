import unittest
from makeDataLoaders import load_data
import torch

class test_loaders(unittest.TestCase):

    #Tests is the random seed actually yeilds reproducably train data sets
    def test_RndSeed_train(self):
        GP_train_A, GP_test_A, SNGP_train_A, SNGP_val_A, SNGP_test_A = load_data("Data/Noisy/Sin",
                                                                         10_000,
                                                                         random_state=42,
                                                                         shuffle=False)
        GP_train_B, GP_test_B, SNGP_train_B, SNGP_val_B, SNGP_test_B = load_data("Data/Noisy/Sin",
                                                                         10_000,
                                                                         random_state=42,
                                                                         shuffle=False)
        
        #Test equality for GP Train
        for x_A, y_A in GP_train_A:
            for x_B, y_B in GP_train_B:
                self.assertTrue(torch.equal(x_A, x_B))
                self.assertTrue(torch.equal(y_A, y_B))

        #Test equality for GP Test
        for x_A, y_A in GP_test_A:
            for x_B, y_B in GP_test_B:
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

        #Test equality for SNGP Test
        for x_A, y_A in SNGP_test_A:
            for x_B, y_B in GP_test_B:
                self.assertTrue(torch.equal(x_A, x_B))
                self.assertTrue(torch.equal(y_A, y_B.unsqueeze(-1)))

    #Test if the dataLoaders for GP and SNGP contain the same data for train load
    def test_SNGP_GP_Loads_equals_train(self):
        GP_train, GP_test, SNGP_train, SNGP_val, SNGP_test = load_data("Data/Noisy/Sin",
                                                                  10_000,
                                                                  random_state=12345,
                                                                  shuffle=False)
        
        for x_GP, y_GP in GP_train:
            for x_SNGP_tr, y_SNGP_tr in SNGP_train:
                for x_SNGP_val, y_SNGP_val in SNGP_val:
                #data values in them should be the same
                    self.assertTrue(torch.equal(x_GP, torch.cat([x_SNGP_tr, x_SNGP_val])))
                #Have to add an extra dimension to SNGP y
                    self.assertTrue(torch.equal(y_GP.unsqueeze(-1),torch.cat([y_SNGP_tr, y_SNGP_val])))


        #Test that test is equal
        for x_GP, y_GP in GP_test:
            for x_SNGP, y_SNGP in SNGP_test:
                self.assertTrue(torch.equal(x_GP, x_SNGP))
                self.assertTrue(torch.equal(y_GP.unsqueeze(-1), y_SNGP))

                
    #Test that different seeds yeild different results
    def test_diff_seed_train(self):
        GP_train_A, GP_test_A, SNGP_train_A, SNGP_val_A, SNGP_test_A = load_data("Data/Noisy/Sin",
                                                                         10_000,
                                                                         random_state=42)
        GP_train_B, GP_test_B, SNGP_train_B, SNGP_val_B, SNGP_test_B = load_data("Data/Noisy/Sin",
                                                                         10_000,
                                                                         random_state=6789)
        
        #Test inequality for GP Train
        for x_A, y_A in GP_train_A:
            for x_B, y_B in GP_train_B:
                self.assertTrue(not torch.equal(x_A, x_B))
                self.assertTrue(not torch.equal(y_A, y_B))

        for x_A, y_A in GP_test_A:
            for x_B, y_B in GP_test_B:
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


        for x_A, y_A in SNGP_test_A:
            for x_B, y_B in SNGP_test_B:
                self.assertTrue(not torch.equal(x_A, x_B))
                self.assertTrue(not torch.equal(y_A, y_B))

if __name__ == '__main__':
    unittest.main()
        