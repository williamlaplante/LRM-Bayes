import sys
sys.path.append('../../../')

import torch
import numpy as np

from DiscreteFisherBayes.Source.Models import CMP
from DiscreteFisherBayes.Source.Posteriors import FDBayes, KSDBayes, Bayes


# In this file, for a fixed theta_1, theta_2, we produce 5 datasets from a CMP(theta_1, theta_2)
# Then, we produce posterior samples with Bayes, FDBayes and KSDBayes.
# The 5 datasets, and 3 times 5 posterior samples (one for each method and dataset), are saved.


theta1 = 4.0
theta2 = 1.25
dnum = 2000
pnum = 5000
numboot = 100
        

for i in range(5):
    #We first sample the data
    print("Sampling Data...")
    cmp = CMP()
    data = cmp.sample(torch.tensor([theta1, theta2]), 100*dnum, 10*dnum)[::100,]

    #We define the prior distribution
    prior = torch.distributions.MultivariateNormal(torch.tensor([3.0, 3.0]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    log_prior = lambda param: prior.log_prob(param).sum()

    #We define the three methods that are used in Takuo's paper
    posterior_FDBayes = FDBayes(cmp.ratio_m, cmp.ratio_p, cmp.stat_m, cmp.stat_p, log_prior)
    posterior_Bayes = Bayes(cmp.uloglikelihood, torch.arange(100).reshape(100, 1), log_prior)
    posterior_KSDBayes = KSDBayes(cmp.ratio_m, cmp.stat_m, cmp.shift_p, log_prior)

    posterior_FDBayes.set_X(data)
    posterior_Bayes.set_X(data)
    posterior_KSDBayes.set_X(data)


    #This function is the one used to fit beta (calibration). It returns the "optimal" beta.
    def get_beta_opt(posterior):
        
        Ps = np.zeros((10, pnum, 2))
        beta_opt = torch.tensor([1.0])

        p_init, _ = posterior.minimise(posterior.loss, prior.sample(), ite=50000, lr=0.1, loss_thin=100, progress=False)
        boot_minimisers, _ = posterior.bootstrap_minimisers(data, numboot, lambda: p_init)
        posterior.set_X(data)
        beta_opt = posterior.optimal_beta(posterior.loss, boot_minimisers)
        return beta_opt


    print("Optimizing beta...")
    beta_opt_FDBayes = get_beta_opt(posterior_FDBayes)
    #beta_opt_KSDBayes = get_beta_opt(posterior_KSDBayes) NOTE: this takes literally 1h... 

    transit_p = torch.distributions.Normal(torch.zeros(2), 0.1*torch.ones(2))

    #We get posterior samples
    print("Posterior Sampling...")
    post_sample_Bayes = posterior_Bayes.sample(pnum, pnum, transit_p, prior.sample(), beta=torch.tensor([1.0]))

    post_sample_KSDBayes = posterior_KSDBayes.sample(pnum, pnum, transit_p, prior.sample(), beta=torch.tensor([1.0]))

    post_sample_FDBayes = posterior_FDBayes.sample(pnum, pnum, transit_p, prior.sample(), beta=beta_opt_FDBayes)

    #We save the posterior samples
    print("Saving Samples and Dataset...")
    for type, sample in zip(["Bayes", "KSDBayes", "FDBayes"], [post_sample_Bayes, post_sample_KSDBayes, post_sample_FDBayes]):

        File_ID = type + '_theta1=' + str(theta1) + '_theta2=' + str(theta2) + '_numboot=' + str(numboot) + '_dnum=' + str(dnum) + '_pnum=' + str(pnum) + '_data=dataset' + str(i)
        np.save("./outputs/"+File_ID+"_samples.npy", sample)
    
    #We save the dataset so we can use it later
    np.save(f"./outputs/dataset{i}_theta1={theta1}_theta2={theta2}.npy", data.numpy())