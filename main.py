import argparse
import torch
import os
#Import code we wrote to for models
from sngp import train
from sngp import model
from gaussian_process.true_gp import FullGPCholesky, train_gp
from dataset.makeDataLoaders import load_data_test, load_data_train


def get_args_parser():
    parser = argparse.ArgumentParser(description = "Arguments for running experiment")

    #Add model specific arguments
    parser.add_argument("modelName", help="Name of model to train, SNGP or GP")
    parser.add_argument("--rank", type=int, help = "Rank to run for SNGP, not need for GP")
    parser.add_argument("--lenScale", type=float, default=0.25)
    parser.add_argument("--outScale", type=float, default = 1.0)

    #Add arguments for dataset
    parser.add_argument("dataset", choices=["Sin", "CrazySin", "Friedman"], 
                        help="name of dataset to run")
    parser.add_argument("num_examples", type=int, help="Number of examples to use for training")
    parser.add_argument("--tr_ratio", type = float, default= 0.7, help="Percent of data to use as training")

    #Other Useful Args
    parser.add_argument("--seed", type = int, default=42)
    parser.add_argument("--savePath", default="results")

    return parser
    

if __name__ == '__main__':

    #Get arguments
    parser = get_args_parser()
    args = parser.parse_args()

    #Check if cuda is avaliable
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA:", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Using CPU")


    #Load Datasets
    GP_train, SNGP_train, SNGP_val = load_data_train("dataset/Data/Noisy/"+args.dataset,
                                                             args.num_examples,
                                                             train_percentage= args.tr_ratio,
                                                             random_state=args.seed)
    GP_test, SNGP_test = load_data_test("dataset/Data/Clean/"+args.dataset,
                                         args.num_examples,
                                         random_state=args.seed)
    
    for x, y in SNGP_val:
        F = x.shape[1]



    #Make Correct Model and Train
    if args.modelName == "SNGP":

        saveFolder = args.savePath
        modelID = "SNGP_R%d_LS%.2f_OS%.2f_%d%s" %(args.rank,
                                                  args.lenScale,
                                                  args.outScale,
                                                  args.num_examples,
                                                  args.dataset)
        os.makedirs(saveFolder + "/SNGP/info", exist_ok=True)

        #Make SNGP Model
        sngp_model = model.RFFGP_Reg(in_features=F,
                                     out_features=1,
                                     rank = args.rank,
                                     lengthscale= args.lenScale,
                                     outputscale= args.outScale).to(device)
        
        #Trained SNGP Model
        #Note while playing around I found it is incredibly senstive to learning rate
        # I.E. 0.001 diverges but 0.0005 converges so be careful with it
        trained_model, info = train.train_model(sngp_model, device, 
                                                SNGP_train, SNGP_val, l2pen_mag=1.0,
                                                n_epochs=1_000, lr=0.0005, do_early_stopping=True)
        torch.save(info, saveFolder +"/SNGP/info/" + modelID)

    elif args.modelName == "GP":

        saveFolder = args.savePath
        modelID = "GP_%d%s" %(args.num_examples,
                              args.dataset)
        os.makedirs(saveFolder + "/GP/info", exist_ok=True)

        gp = FullGPCholesky(lengthscale=1.0, outputscale=1.0, noise=0.2, learn_hyperparams=True)
        for X_train, y_train in GP_train:
            info = train_gp(gp, X_train, y_train, n_iterations=400, lr=0.01)
            gp.fit(X_train, y_train)
            #Don't save model cause they big boyz
            #torch.save(gp, saveFolder + "/GP/model/" + modelID)
            torch.save(info, saveFolder + "/GP/info/" + modelID)


    

    
    
