import numpy as np
import matplotlib.pyplot as plt
import iminuit
import probfit

import pyximport; pyximport.install()
from randomgaussian import *

# We will use the same data as in the previous example
np.random.seed(0)
data = np.random.randn(10000) * 4 + 1
# sigma = 4 and mean = 1
plt.hist(data, bins=100, histtype='step');

# Define the unbinned likelihood cost function 
unbinned_likelihood = probfit.UnbinnedLH(gauss_pdf_cython, data)
minuit = iminuit.Minuit(unbinned_likelihood, sigma=2, pedantic=False, print_level=0)
# Remember: minuit.errordef is automatically set to 0.5
# as required for likelihood fits (this was explained above)
minuit.migrad() # yes: amazingly fast
unbinned_likelihood.show(minuit)
minuit.print_fmin()
minuit.print_matrix() 
