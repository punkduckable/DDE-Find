from    typing      import  List;

import  torch; 

# Logger setup 
import  logging;
LOGGER : logging.Logger = logging.getLogger(__name__);



class Constant(torch.nn.Module):
    """
    This class implements a constant IC function:
        X0(t) = x0
    In this case, \phi = x0.
    """
    def __init__(self, x0 : torch.Tensor) -> None:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        x0 : This should be a single element tensor whose lone value holds the constant we want to 
        set the IC to.
        """

        # Run the super class initializer. 
        super().__init__();

        # Run checks
        assert(len(x0.shape) == 1);

        # Set d.
        self.d = x0.shape[0];

        # Store the constant x0 value as a parameter object.
        self.x0 = torch.nn.Parameter(x0, requires_grad = True);


    def forward(self, t : torch.Tensor) -> torch.Tensor:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        t : This should be a 1D torch.Tensor whose i'th value holds the i'th t value.

        
        -------------------------------------------------------------------------------------------
        Returns: 

        A 1D torch.Tensor object whose i'th value holds x0.
        """

        # The IC is ALWAYS x0... it's a constant!
        return self.x0.repeat(list(t.shape) + [1]);



class Affine(torch.nn.Module):
    """
    This class implements a simple affine IC:
        X0(t) = a*t + b
    In this case, \phi = (a, b).
    """
    def __init__(self, a : torch.Tensor, b : torch.Tensor) -> None:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        a, b : These should be 1D tensors whose holding the constants in the function t -> a*t + b.
        """

        # Run the super class initializer. 
        super().__init__();

        # Run checks
        assert(len(a.shape) == 1);
        assert(len(b.shape) == 1);
        assert(a.shape[0]   == b.shape[0]);

        # Set d.
        self.d = a.shape[0];

        # Store the constants a, b as parameters.
        self.a = torch.nn.Parameter(a.reshape(1, -1), requires_grad = True);
        self.b = torch.nn.Parameter(b.reshape(1, -1), requires_grad = True);



    def forward(self, t : torch.Tensor) -> torch.Tensor:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        t : This should be a 1D torch.Tensor whose i'th value holds the i'th t value.

        
        -------------------------------------------------------------------------------------------
        Returns: 

        A 1D torch.Tensor object whose i'th value holds x0.
        """

        # Reshape t.
        t = t.reshape(-1, 1);

        # Compute the IC!
        return (self.a)*t + self.b;



class Periodic(torch.nn.Module):
    """
    This class implements a simple periodic IC:
        X0(t) = A*cos(w*t) + b
    In this case, \phi = (A, w, b).
    """
    def __init__(self, A : torch.Tensor, w : torch.Tensor, b : torch.Tensor) -> None:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        A, w, b: These should be 1D tensors whose k'th components define the k'th component of the 
        initial condition: X0_k(t) = A_k * sin(w_k * t) + b_k.
        """

        # Run the super class initializer. 
        super().__init__();

        # Run checks
        assert(len(A.shape) == 1);
        assert(len(w.shape) == 1);
        assert(len(b.shape) == 1);
        assert(A.shape[0]   == w.shape[0]);
        assert(A.shape[0]   == b.shape[0]);

        # set d.
        self.d = A.shape[0];

        # Store the constants A, w as parameters.
        self.A = torch.nn.Parameter(A.reshape(1, -1), requires_grad = True);
        self.w = torch.nn.Parameter(w.reshape(1, -1), requires_grad = True);
        self.b = torch.nn.Parameter(b.reshape(1, -1), requires_grad = True);



    def forward(self, t : torch.Tensor) -> torch.Tensor:
        """
        -------------------------------------------------------------------------------------------
        Arguments:
        
        t : This should be a 1D torch.Tensor whose i'th value holds the i'th t value.

        
        -------------------------------------------------------------------------------------------
        Returns: 

        A 1D torch.Tensor object whose i'th value holds X0(t[i]).
        """

        # Reshape t.
        t = t.reshape(-1, 1);

        # Compute the IC!
        return torch.mul(self.A, torch.sin(torch.mul(t, self.w))) + self.b;





class Neural(torch.nn.Module):
    """ 
    This class implements a neural network IC:
        X0(t) = F(t)
    where F is a neural network. In this case, \phi is the parameters (weights/biases) of F.
    """

    def __init__(self, Widths : List[int]):
        """ 
        This class defines IC of a DDE as a neural network. We use Widths to define the widths of 
        the layers in the network. We also use the softplus activation function after each hidden 
        layer.


        
        --------------------------------------------------------------------------------------------
        Arguments:
        --------------------------------------------------------------------------------------------

        Widths: This should be a list of N + 1 integers, where N is the number of layers in the 
        neural network. Widths[0] represents the dimension of the domain, while Widths[-1] 
        represents the dimension of the co-domain. For i \in {1, 2, ... , N - 2}, Widths[i] 
        represents the width of the i'th hidden layer. Because a Neural object takes in t as its 
        input, Widths[0] must be 2d + 2. Finally, Widths[1] must be d. 
        """
        
        # Call the super class initializer. 
        super(Neural, self).__init__();

        # Make sure Widths is a list of ints.
        self.N_Layers = len(Widths) - 1;
        for i in range(self.N_Layers + 1):
            assert(isinstance(Widths[i], int));
        
        # Find d, make sure 2*Widths[-1] + 2 == Widths[0].
        self.d = Widths[-1];
        assert(Widths[0] == 1);
        self.Widths = Widths;

        # Set up the network's layers.
        self.Layers     = torch.nn.ModuleList();
        for i in range(self.N_Layers):
            self.Layers.append(torch.nn.Linear(in_features = Widths[i], out_features = Widths[i + 1]));
            torch.nn.init.xavier_normal_(self.Layers[i].weight);

            torch.nn.init.zeros_(self.Layers[i].bias);
        
        # Finally, set the activation function.
        self.Activation = torch.nn.Softplus();



    def forward(self, t : torch.Tensor) -> torch.Tensor:
        """
        This function passes t through the neural network and returns X0(t) (see class doc string).
        
        --------------------------------------------------------------------------------------------
        Arguments:

        t : This should be a 1D torch.Tensor whose i'th value holds the i'th t value.
        """

        t = t.reshape(-1, 1);

        # Set up the input to the network.
        X : torch.Tensor = t;

        # Compute, return the output.  
        for i in range(self.N_Layers - 1):
            X = self.Activation(self.Layers[i](X));
        Output : torch.Tensor = self.Layers[-1](X);
        return Output;
