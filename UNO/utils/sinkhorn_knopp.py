import torch


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    @torch.no_grad()
    def forward(self, logits):
        Q = torch.exp(logits / self.epsilon).t()
        B = Q.shape[1]  # Number of features in each prototype
        K = Q.shape[0]  # Number of prototypes

        # Make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        # For each iteration
        for it in range(self.num_iters):
            # Normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # Normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        # The columns must sum to 1 so that Q is an assignment, then return Q transposed
        Q *= B
        return Q.t()
    
