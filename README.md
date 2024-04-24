---------------------------------------------------------------------------------------------------
### Introduction
---------------------------------------------------------------------------------------------------

This document gives a rough guide on how to use the `DDE-Find` library. If you have any questions 
that I haven't answered here, please email me (Robert) at `rrs254@cornell.edu`. 

The `DDE-Find` repository consists of a few `.py` files and (by popular demand!) a user-friendly 
`Jupyter` notebook to drive them. For the most part, you should only need to interact with the 
`Jupyter` Notebook (`Experiment.ipynb`). The repository also includes a script and config file 
(`Experiments.py` and `Experiments.conf`, respectively) to run repeated experiments at specified
noise levels. We used these files to generate most of the results in the `DDE-Find` paper.



---------------------------------------------------------------------------------------------------
### Dependencies 
---------------------------------------------------------------------------------------------------

The `DDE-Find` library uses a number of external libraries. We recommend setting up a virtual 
environment for the project (alternatively, you can manually install the packages listed below). 
Below is a guide on how to do that (note that you must already have `conda` installed).

First, open a command line in the `DDE-Find` library and make a new virtual environment using 
`python 3.10`:

`conda create --name DDE_Find python=3.10`

Say yes if conda prompts you. Once you've built the virtual environment, activate it:

`conda activate DDE_Find`

Now, let's add the packages we ned to run the `DDE-Find` code (note: the versions are optional; 
the versions I list are the ones I used when developing this library. If you encounter any packages 
errors while running the code, try re-installing the packages with the specified versions):

`conda install numpy=1.26.3`
`conda install torch=2.1.2`
`conda install matplotlib=3.8.2`
`conda install seaborn=0.13.2`
`conda install scipy=1.12.9`
`conda install jupyter`
`conda install tensorboard=2.16.2`
`conda install pyyaml=6.0.1`

The virtual environment is now built and ready to run! 

In the future, to activate the environment (and thus, gain access to all the packages we need to
run the `DDE-Find` library), simply activate the environment with `conda activate DDE_Find`.



---------------------------------------------------------------------------------------------------
### Library contents
---------------------------------------------------------------------------------------------------

All of our code has extensive comments. However, to give an overview, our library is structured 
as follows:

`Model.py`: This file houses class definitions for the the various models (structured and neural)
in the `DDE-Find` library. Objects of these classes are callable, parameterized `torch.nn.Module` 
objects that act as the right-hand side of a DDE. 

If you want to extend `DDE-Find` to a new structured model, you can do so by defining a new 
class in `Model.py`. For a Model class to work with the rest of `DDE-Find`, it must subclass 
`torch.nn.Module` and its `forward` method must accept four arguments, $x(t), x(t - \tau), \tau$, 
and $t$. Further, each learnable parameter must be a `torch.nn.parameter.Parameter` object 
(otherwise, the optimizer will not see them). 

If you want to make your Model compatible with the `Experiments.py` script, each component of 
$\theta$ must get its own `torch.nn.parameter.Parameter` object. For example, if 
$\theta = \theta_1 e_1 + \theta_2 e_2 + \theta_3 e_3$, then you should store each component 
of $\theta$ in a separate `torch.nn.parameter.Parameter` object. See any of the existing model 
classes for an example. This step is necessary for `Experiments.py` to fetcheach of the model's 
parameters and construct the $\theta$ table at the end.

`X0.py`: This file houses our class definitions for the various kinds of initial conditions. 
Objects of these classes are callable, parameterized `torch.nn.Module` objects that act as the 
initial condition function in a DDE IVP.

If you want to extend `DDE-Find` to a new kind of initial condition, you can do so by defining 
a new class in `X0.py`. For a X0 class to work with the rest of `DDE-Find`, it must subclass 
`torch.nn.Module` and its `forward` method must accept one argument, $t$. Further, each 
learnable parameter must be a `torch.nn.parameter.Parameter` object (otherwise, the optimizer 
will not see them). 

If you want to make your X0 compatible with the `Experiments.py` script, each component of 
$\phi$ must get its own `torch.nn.parameter.Parameter` object. For example, if 
$\phi = \phi_1 e_1 + \phi_2 e_2$, then you should store each component of $\phi$ in a separate 
`torch.nn.parameter.Parameter` object. See any of the existing X0 classes for an example.
This step is necessary for `Experiments.py` to fetch each of the IC's's parameters and construct 
the $\phi$ table at the end.

`Loss.py`: This file houses our cost and loss functions. Currently, we have two loss functions: 
`SSE_Loss` and `Integral_Loss`. `SSE_Loss` is the sum of squares error between the target and 
predicted trajectory. The `Integral_Loss`, by contrast, implements a numerical approximation to the 
loss function $\int_{0}^{T} l(x(t)) dt$. We use the trapezoidal rule to approximate the integral in 
this loss. All of our experiments in the paper use the `Integral_Loss`, but our algorithm works 
with both loss functions. There are two cost functions, `L2_cost` and `L1_Cost`. These implement 
the $L^1$ and $L^2$ norms, respectively. You should define $\ell$ and $G$ using these costs. Note 
that one of the arguments to the `Integral_Loss` is cost function, `l`, which acts as $\ell$.

`Experiment.ipynb`: The main Jupyter Notebook file. This file drives the rest of the code. I'll talk 
about this file in more detail below. 

`NDDE.py`: This file houses two classes: `NDDE` and `DDE_adjoint_Backward`. `NDDE` is a wrapper
for a DDE IVP. It includes two pieces, $F$ (a Model object) and $X_0$ (an IC object). This class
calls `DDE_adjoint_Backward` to make predictions and update the model's parameters. 
`DDE_adjoint_Backward` is a `torch.autograd.Function` subclass which is designed to implement the 
forward and backward passes in `DDE-Find`. The forward method takes a model, initial condition, 
$\tau$ estimate, $N_{\tau}$, $T$, $\ell$, $G$, $\theta$, and $\phi$. It uses these values to compute 
the forward trajectory by solving the forward DDE using one of our solvers (see below). It then 
stores the values we need for the backward step and returns the predicted trajectory. 

The backward method is more involved but essentially solves the adjoint equation and then uses this 
solution to compute the gradient of $L$ with respect to $\theta$, $\tau$, and $\phi$ (it returns 
these quantities). To do this, the backward method first fetches the data from the forward pass, 
sets up some tensors, and then solves the adjoint equation backward in time. We do this either 
using the forward Euler or RK2 solvers (The code is set up to the RK2 solver, though both solvers 
work). Once we have the discretized adjoint solution, we compute $\partial_{\theta} \mathcal{L}$, 
$\partial \mathcal{L} / \partial \tau$, and $\partial_{\phi} \mathcal{L}$. We do this using the
equations in theorem 1 of the paper. Many of these steps involve computing integrals. We use the 
trapezoidal rule to evaluate all integrals. 

`Solver.py`: This file houses two DDE solvers: A forward Euler solver and a basic Runge Kutta 
solver. We use these solvers for the forward pass in the `DDE_adjoint_Backward` class. 



---------------------------------------------------------------------------------------------------
### `Experiment.ipynb`
---------------------------------------------------------------------------------------------------

With all that established, let's talk about how to use the Jupyter Notebook. The first code cell 
imports the relevant files/libraries. If you want to change the solver or loss function, you
should do that here. Note that if you want to use a different solver, you will also need to change 
the corresponding import statement in the `NDDE.py` file.

The cell titled "Generate Target Trajectory" uses the selected solver to get the target trajectory 
(by solving the true DDE, which is also set in this cell). Note that the code includes several 
pre-set target DDEs. These are commented out by default. To select one, uncomment the model you
want to use (beginning with the name of the model and ending with the `T_Target` definition). You 
should uncomment the corresponding segment of code in the "Train the Model" code bock.

Next, the cell titled "Train the Model" initializes the model and runs the Epochs. This cell 
is where you change the initial guess for the parameters and $\tau$ (which - hopefully - will train 
to match the values you set for the true trajectory in the "Generate True Solution" cell). The code 
includes several pre-set segments of code, one for each model type. You should uncomment one of 
these segments (beginning with the name of the model and ending with the `Param_List` definition).
The segment of code you uncomment should match the segment you uncommented in the "Generate Target 
Trajectories" code cell (if these don't match, weird things will happen). 

For each epoch, we first compute the predicted trajectory using the current parameter, $\tau$ 
values. We then interpolate the target trajectory so we can evaluate the forward and adjoint 
trajectories at the same time values (in case the two use different step sizes). Next, we compute 
the loss between the predicted and target trajectories, perform backprop, and then update $\theta$, 
$\tau$, and $\phi$. Note that the forward and backward passes use the `DDE_adjoint_Backward` class 
in `NDDE.py`. 

There are also code blocks to plot the true, target, and predicted trajectories after training.



---------------------------------------------------------------------------------------------------
### `Experiment.conf`
---------------------------------------------------------------------------------------------------

While `Experiments.ipynb` gives you a user-friendly jupyter notebook, it is not the best for 
running multiple experiments and obtaining statistics. For that, we recommend using `Experiment.py`
and `Experiment.conf`. The former is a script and the later is a configuration file to drive it. 
Before running, select an appropriate set of settings in `Experiment.conf` (the configuration file 
includes comments telling you what each setting does). Once you have set your settings, run the 
script (make sure you have the `DDE_Find` virtual environment enabled):

`python ./Experiments.py`



---------------------------------------------------------------------------------------------------
### Conclusion 
---------------------------------------------------------------------------------------------------

Hopefully, this is enough to get you started. If you have more specific questions about our 
implementation, our code's comments should be able to answer them. Otherwise, feel free to 
email me (`rrs254@cornell.edu`) with additional questions.

This package and README was developed by Robert Stephany.

`__/\\\\\\\\\\\\_____/\\\\\\\\\\\\_____/\\\\\\\\\\\\\\\________________/\\\\\\\\\\\\\\\_____________________________/\\\__        `
` _\/\\\////////\\\__\/\\\////////\\\__\/\\\///////////________________\/\\\///////////_____________________________\/\\\__       `
`  _\/\\\______\//\\\_\/\\\______\//\\\_\/\\\___________________________\/\\\______________/\\\______________________\/\\\__      `
`   _\/\\\_______\/\\\_\/\\\_______\/\\\_\/\\\\\\\\\\\______/\\\\\\\\\\\_\/\\\\\\\\\\\_____\///___/\\/\\\\\\__________\/\\\__     `
`    _\/\\\_______\/\\\_\/\\\_______\/\\\_\/\\\///////______\///////////__\/\\\///////_______/\\\_\/\\\////\\\____/\\\\\\\\\__    `
`     _\/\\\_______\/\\\_\/\\\_______\/\\\_\/\\\___________________________\/\\\_____________\/\\\_\/\\\__\//\\\__/\\\////\\\__   `
`      _\/\\\_______/\\\__\/\\\_______/\\\__\/\\\___________________________\/\\\_____________\/\\\_\/\\\___\/\\\_\/\\\__\/\\\__  `
`       _\/\\\\\\\\\\\\/___\/\\\\\\\\\\\\/___\/\\\\\\\\\\\\\\\_______________\/\\\_____________\/\\\_\/\\\___\/\\\_\//\\\\\\\/\\_ `
`        _\////////////_____\////////////_____\///////////////________________\///______________\///__\///____\///___\///////\//__`
