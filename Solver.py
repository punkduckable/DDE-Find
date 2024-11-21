import  torch;
from    typing  import Tuple;



def Forward_Euler(F : torch.nn.Module, X0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor, N_tau : int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function uses the Forward Euler method to approximately solve
        x'(t)   = F(x(t), x(t - \tau), \tau, t)     t \in [0, T]
        x(t)    = X0(t)                             t \in [-\tau, 0]
    Here, x \in \mathbb{R}^d.
    
    --------------------------------------------------------------------------------------------
    Arguments:

    F : This is a torch.nn.Module object which represents the right-hand side of the DDE (See 
    above). 

    X0: This is a module which gives the initial condition in [-\tau, 0]. In particular, at
    time t \in [-\tau, 0], we set the initial condition at time t to X0(t). X0 should be a
    torch.nn.Module object which takes a tensor (of shape S) of time values and returns a S x d
    tensor[s, :] element holds the value of the IC at the s'th element of the input tensor.

    tau : This is a single element tensor whose lone element represents the time delay.

    T : this is a single element tensor whose lone element represents the final time. 

    N_tau: The number of time steps in the interval [0, \tau]. This decides the resolution of
    the numerical solve.

    --------------------------------------------------------------------------------------------
    Returns:

    A two element tuple. The first element holds a 2D tensor whose kth row holds the state of 
    the system (as solved by the solver) at the kth time value. The second is another 1D tensor 
    whose kth element holds the kth time value. Note: We only approximate the solution in the 
    interval [0, T].
    """

    # Checks
    assert(tau.numel()      == 1);
    assert(tau.item()       >  0);
    assert(T.numel()        == 1);
    assert(isinstance(N_tau, int));
    assert(N_tau            >  0);

    # Define the time-step to be tau/N_tau of the delay
    dt  : float         = tau.item()/N_tau;

    # Find the number of time steps (of size dt) to get to T. i.e. min{ N : N*dt >= T}.
    N   : int           = int(torch.ceil(T/dt).item());

    # Now, get the IC. We also use this chance to recover d.
    X0_0    : torch.Tensor  = X0(torch.tensor(0, dtype = torch.float32));
    d       : int           = X0_0.shape[1];

    # Set up tensors to hold the solution, time steps. In general, we want the solution at N + 1 
    # times: 0, dt, 2*dt, ... , N*dt. 
    x_Trajectory    : torch.Tensor  = torch.empty([N + 1, d],   dtype = torch.float32);
    t_Trajectory    : torch.Tensor  = torch.linspace(start = 0, end = N*dt, steps = N + 1);

    # Set the first column of x to the IC at time 0 (the last element of t_Values_X0).
    x_Trajectory[0, :]             = X0_0;
    
    # Compute the solution!
    for i in range(0, N):
        # Fetch t, x(t), and x(t - tau) at the i'th time value. Note that if t < tau (equivalently, 
        # i < N_tau), then t - \tau < 0, which means that x(t - \tau) = X0(t - \tau)
        t_i     : torch.Tensor  = t_Trajectory[i];                                                          # t
        x_i     : torch.Tensor  = x_Trajectory[i, :];                                                       # x(t)
        y_i     : torch.Tensor  = x_Trajectory[i - N_tau, :] if i > N_tau else X0(t_i - tau).reshape(-1);   # x(t - tau)
        F_i     : torch.Tensor  = F(x_i, y_i, tau, t_i);

        # Now... compute x at the next time step.
        x_Trajectory[i + 1, :] = x_i + dt*F_i;
       
    # All done!
    return (x_Trajectory, t_Trajectory);



def RK2(F : torch.nn.Module, X0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor, N_tau : int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function uses a 2nd order Runge-Kutta method to approximately solve
        x'(t)   = F(x(t), x(t - \tau), \tau, t) t \in [0, T]
        x(t)    = X0(t)                         t \in [-\tau, 0]
    Here, x \in \mathbb{R}^d.
    
    --------------------------------------------------------------------------------------------
    Arguments:

    F : This is a torch.nn.Module object which represents the right-hand side of the DDE (See 
    above). 

    X0: This is a module which gives the initial condition in [-\tau, 0]. In particular, at
    time t \in [-\tau, 0], we set the initial condition at time t to X0(t). X0 should be a
    torch.nn.Module object which takes a tensor (of shape S) of time values and returns a S x d
    tensor[s, :] element holds the value of the IC at the s'th element of the input tensor.

    tau : This is a single element tensor whose lone element represents the time delay.

    T : this is a single element tensor whose lone element represents the final time. 
    
    N_tau: The number of time steps in the interval [0, \tau]. This decides the resolution of
    the numerical solve.
    
    --------------------------------------------------------------------------------------------
    Returns:

    A two element tuple. The first element holds a 2D tensor whose kth row holds the state of 
    the system (as solved by the solver) at the kth time value. The second is another 1D tensor 
    whose kth element holds the kth time value. Note: We only approximate the solution in the 
    interval [0, N*dt], where N = min{ N : N*dt >= T }.
    """

    # Checks
    assert(tau.numel()      == 1);
    assert(tau.item()       >  0);
    assert(T.numel()        == 1);
    assert(isinstance(N_tau, int));
    assert(N_tau            >  0);


    # Define the time-step to be tau/N_tau.
    dt  : float         = tau.item()/N_tau;

    # Find the number of time steps (of size dt) to get to T. i.e. min{ N : N*dt >= T}.
    N   : int           = int(torch.ceil(T/dt).item());

    # Now, get the IC. We also use this chance to recover d.
    X0_0    : torch.Tensor  = X0(torch.tensor(0, dtype = torch.float32));
    d       : int           = X0_0.shape[1];

    # Set up tensors to hold the solution, time steps. In general, we want the solution at N + 1 
    # times: 0, dt, 2*dt, ... , N*dt. 
    x_Trajectory    : torch.Tensor  = torch.empty([N + 1, d],   dtype = torch.float32);
    t_Trajectory    : torch.Tensor  = torch.linspace(start = 0, end = N*dt, steps = N + 1);

    # Set the first column of x to the IC at time 0 (the last element of t_Values_X0).
    x_Trajectory[0, :]             = X0_0;
    
    # Compute the solution!
    for i in range(0, N):
        # Find x at the i+1'th time value. We do this using a 2 step RK method. Note that if 
        # t < tau (equivalently, i < N_tau), then t - \tau < 0, which means that x(t - \tau) = x0. 
        t_i     : torch.Tensor  = t_Trajectory[i];                                                          # t
        x_i     : torch.Tensor  = x_Trajectory[i, :];                                                       # x(t)
        y_i     : torch.Tensor  = x_Trajectory[i - N_tau, :] if i > N_tau else X0(t_i - tau).reshape(-1);   # x(t - tau)
        k1      : torch.Tensor  = F(x_i, y_i, tau, t_i);

        t_ip1   : torch.Tensor  = t_Trajectory[i + 1];                                                      # t + dt
        x_ip1   : torch.Tensor  = x_i + dt*k1;                                                              # x(t) + dt*k1
        y_ip1   : torch.Tensor  = x_Trajectory[i + 1 - N_tau, :] if i + 1 > N_tau else X0(t_i + dt - tau).reshape(-1);  # x(t + dt - tau)
        k2      : torch.Tensor  = F(x_ip1, y_ip1, tau, t_ip1);

        x_Trajectory[i + 1, :]  = x_i + dt*0.5*(k1 + k2);

    # All done!
    return (x_Trajectory, t_Trajectory);



def RK4(F : torch.nn.Module, X0 : torch.Tensor, tau : torch.Tensor, T : torch.Tensor, N_tau : int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function uses a 4th order Runge-Kutta method to approximately solve
        x'(t)   = F(x(t), x(t - \tau), \tau, t) t \in [0, T]
        x(t)    = X0(t)                         t \in [-\tau, 0]
    Here, x \in \mathbb{R}^d. Specifically, we use the following method:
        y_{n + 1} = y_{n} + (h/6)(k_1^n + 2 k_2^n + 2 k_3^n + k_4^n)
        t_{n + 1} = t_n + h

        k_1^n = F(x(t_n),   x(t_n - N_tau*h),       \tau,   t_n)

        g_2^n = x(t_n) + (h/2) k_1^n
        k_2^n = F(g_2^n,    \psi_2^n,               \tau,   t_n + h/2)

        g_3^n = x(t_n) + (h/2) k_2^n
        k_3^n = F(g_3^n,    \psi_3^n,               \tau,   t_n + h/2)

        g_4^n = x(t_n) + h k_3^n
        k_4^n = F(g_4^n,    x(t_n + h - N_tau*h),   \tau,   t_n + h)

    Where, h is the step size and 
        \psi_i^n =  { X_0(t_n + c_i h - N_tau h)        if n < N_tau    i = 2, 3
                    { g_i^{n - N_tau}                   otherwise
    Though it may look weird, this is just the DDE adaptation of the standard RK4 method. 
    In our implementation, we keep logs of the past g_k^n values.

      
    --------------------------------------------------------------------------------------------
    Arguments:

    F : This is a torch.nn.Module object which represents the right-hand side of the DDE (See 
    above). 

    X0: This is a module which gives the initial condition in [-\tau, 0]. In particular, at
    time t \in [-\tau, 0], we set the initial condition at time t to X0(t). X0 should be a
    torch.nn.Module object which takes a tensor (of shape S) of time values and returns a S x d
    tensor[s, :] element holds the value of the IC at the s'th element of the input tensor.

    tau : This is a single element tensor whose lone element represents the time delay.

    T : this is a single element tensor whose lone element represents the final time. 
    
    N_tau: The number of time steps in the interval [0, \tau]. This decides the resolution of
    the numerical solve.
    
    --------------------------------------------------------------------------------------------
    Returns:

    A two element tuple. The first element holds a 2D tensor whose kth row holds the state of 
    the system (as solved by the solver) at the kth time value. The second is another 1D tensor 
    whose kth element holds the kth time value. Note: We only approximate the solution in the 
    interval [0, N*dt], where N = min{ N : N*dt >= T }.
    """

    # Checks
    assert(tau.numel()      == 1);
    assert(tau.item()       >  0);
    assert(T.numel()        == 1);
    assert(isinstance(N_tau, int));
    assert(N_tau            >  0);


    # Define the time-step to be tau/N_tau.
    h       : float         = tau.item()/N_tau;

    # Find the number of time steps (of size h) to get to T. i.e. min{ N : N*h >= T}.
    N       : int           = int(torch.ceil(T/h).item());

    # Now, get the IC. We also use this chance to recover d.
    X0_0    : torch.Tensor  = X0(torch.tensor(0, dtype = torch.float32));
    d       : int           = X0_0.shape[1];

    # Make trajectories to hold the past g_2^n and g_3^n values.
    g_2_n   : torch.Tensor  = torch.empty((N, d), dtype = torch.float32);
    g_3_n   : torch.Tensor  = torch.empty((N, d), dtype = torch.float32);

    # Set up tensors to hold the solution, time steps. In general, we want the solution at N + 1 
    # times: 0, dt, 2*dt, ... , N*dt. 
    x_Trajectory    : torch.Tensor  = torch.empty([N + 1, d],   dtype = torch.float32);
    t_Trajectory    : torch.Tensor  = torch.linspace(start = 0, end = N*h, steps = N + 1);

    # Set the first column of x to the IC at time 0 (the last element of t_Values_X0).
    x_Trajectory[0, :]             = X0_0;

    # Compute the solution!
    for i in range(0, N):
        # Compute k_1.
        t_1     : torch.Tensor  = t_Trajectory[i];                                                              # t
        g_1     : torch.Tensor  = x_Trajectory[i, :];                                                           # x(t)
        psi_1   : torch.Tensor  = X0(t_1 - tau).reshape(-1) if i < N_tau else x_Trajectory[i - N_tau, :]        # x(t - \tau).
        k_1     : torch.Tensor  = F(g_1, psi_1, tau, t_1);

        # Compute k_2.
        t_2     : torch.Tensor  = t_Trajectory[i] + h/2;                                                        # t + h/2
        g_2     : torch.Tensor  = x_Trajectory[i] + (h/2)*k_1;                                                  # x(t) + (h/2)k_1
        psi_2   : torch.Tensor  = X0(t_2 - tau).reshape(-1) if i < N_tau else g_2_n[i - N_tau, :];              # x(t - \N_tau) + (h/2)k_1
        k_2     : torch.Tensor  = F(g_2, psi_2, tau, t_2);                                                      # ~F(t + h/2)
        g_2_n[i, :]             = g_2;

        # Compute k_3.
        t_3     : torch.Tensor  = t_Trajectory[i] + h/2;                                                        # t + h/2
        g_3     : torch.Tensor  = x_Trajectory[i] + (h/2)*k_2;                                                  # x(t) + (h/2)k_2
        psi_3   : torch.Tensor  = X0(t_3 - tau).reshape(-1) if i < N_tau else g_3_n[i - N_tau, :];              # x(t - N_tau) + (h/2)k_2
        k_3     : torch.Tensor  = F(g_3, psi_3, tau, t_3)                                                       # ~F(t + h/2)
        g_3_n[i, :]             = g_3;

        # Compute k_4                               
        t_4     : torch.Tensor  = t_Trajectory[i] + h;                                                          # t + h
        g_4     : torch.Tensor  = x_Trajectory[i] + h*k_3                                                       # x(t - N_tau) + h k_3
        psi_4   : torch.Tensor  = X0(t_4 - tau).reshape(-1) if i < N_tau else x_Trajectory[i - N_tau + 1, :]    # ~F(t + h)
        k_4     : torch.Tensor  = F(g_4, psi_4, tau, t_4);
        
        # Take the step!
        x_Trajectory[i + 1, :]  = x_Trajectory[i, :] + (h/6.)*(k_1 + 2*k_2 + 2*k_3 + k_4);

    # All done!
    return (x_Trajectory, t_Trajectory);