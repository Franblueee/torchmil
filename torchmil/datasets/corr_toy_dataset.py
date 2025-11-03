import math
import torch
from tensordict import TensorDict


def normal_pdf(x, mu=0.0, sigma=1.0):
    """Compute the probability density function of a normal distribution."""
    norm_const = 1 / (sigma * math.sqrt(2.0 * math.pi))
    quad_term = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return norm_const * quad_term


class ShiftedMeanMILDataset(torch.utils.data.Dataset):
    r"""
    Shifted Mean MIL Dataset for correlated instance labels.

    This dataset generates synthetic MIL bags where positive bags contain a contiguous
    sequence of R instances with shifted mean features. This creates correlation between
    instance labels within positive bags, which is particularly useful for testing MIL
    algorithms for their ability to detect local instance-to-instance patterns within bags.

    This dataset was introduced in the paper:
    "Synthetic Data Reveals Generalization Gaps in Correlated Multiple Instance Learning"
    arXiv preprint: https://arxiv.org/abs/2510.25759

    **Bag generation:**
    - Each bag has a variable number of instances (between S_low and S_high)
    - Features are drawn from a normal distribution N(mu, sigma^2)
    - For positive bags (Y=1), a contiguous sequence of R instances has their first K features shifted by Delta
    - Instance labels reflect whether an instance has shifted features (1 if shifted, 0 otherwise)

    Each bag is returned as a TensorDict with the following keys:
    - X: The bag's feature matrix of shape `(bag_size, M)`
    - Y: The bag's label (1 for positive, 0 for negative)
    - y_inst: The instance-level labels within the bag
    - bag_size: The number of instances in the bag

    **Example usage:**
    ```python
    from torchmil.datasets import ShiftedMeanMILDataset

    # Create dataset with 100 bags
    dataset = ShiftedMeanMILDataset(N=100, R=3, S_low=15, S_high=45, K=1, M=768,
                                     p_y1=0.5, Delta=1.0, seed=42)

    # Get a bag
    bag = dataset[0]
    X, Y, y_inst = bag['X'], bag['Y'], bag['y_inst']
    print(f"Bag shape: {X.shape}")
    print(f"Bag label: {Y}")
    print(f"Instance labels: {y_inst}")
    ```

    Arguments:
        N: Number of bags to generate
        R: Number of contiguous instances with shifted mean in positive bags
        S_low: Minimum bag size
        S_high: Maximum bag size
        K: Number of features to shift in positive instances
        M: Total number of features
        p_y1: Probability of generating a positive bag
        Delta: Shift amount for positive instances
        mu: Mean of the normal distribution
        sigma: Standard deviation of the normal distribution
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        N,
        R=3,
        S_low=15,
        S_high=45,
        K=1,
        M=768,
        p_y1=0.5,
        Delta=1.0,
        mu=0.0,
        sigma=1.0,
        seed=42,
    ):
        super().__init__()

        self.N = N
        self.R = R
        self.S_low = S_low
        self.S_high = S_high
        self.K = K
        self.M = M
        self.p_y1 = p_y1
        self.Delta = Delta
        self.mu = mu
        self.sigma = sigma
        self.seed = seed

        self.generator = torch.Generator().manual_seed(self.seed)

        # Generate bag sizes
        self.lengths = tuple(
            torch.randint(
                self.S_low, self.S_high + 1, (self.N,), generator=self.generator
            ).tolist()
        )

        # Generate all features at once
        self.H = self.mu + self.sigma * torch.randn(
            sum(self.lengths), self.M, generator=self.generator
        )

        # Generate starting positions for shifted sequences in positive bags
        self.u = torch.cat(
            [
                torch.randint(0, S_i - self.R + 1, (1,), generator=self.generator)
                for S_i in self.lengths
            ]
        )

        # Generate bag labels
        self.y = torch.bernoulli(
            self.p_y1 * torch.ones(size=(self.N,)), generator=self.generator
        ).long()

        # Split features into bags
        self.H_split = torch.split(self.H, self.lengths)

        # Create instance labels for each bag and apply feature shifts
        self.y_inst_list = []
        for i, H_i in enumerate(self.H_split):
            inst_labels = torch.zeros(self.lengths[i], dtype=torch.long)
            if self.y[i] == 1:
                # Shift features for positive bags
                start_idx = self.u[i].item()
                end_idx = start_idx + self.R
                H_i[start_idx:end_idx, 0 : self.K] += self.Delta
                # Mark shifted instances with label 1
                inst_labels[start_idx:end_idx] = 1
            self.y_inst_list.append(inst_labels)

    def __len__(self):
        """Returns the number of bags in the dataset."""
        return self.N

    def __getitem__(self, index: int) -> TensorDict:
        """
        Arguments:
            index: Index of the bag to retrieve

        Returns:
            bag_dict: TensorDict containing the following keys:
                - X: Bag features of shape `(bag_size, M)`
                - Y: Label of the bag
                - y_inst: Instance labels of the bag
                - bag_size: Number of instances in the bag
        """
        if index >= self.N:
            raise IndexError(f"Index {index} out of range (max: {self.N - 1})")

        bag_dict = TensorDict(
            {
                "X": self.H_split[index].float(),
                "Y": self.y[index],
                "y_inst": self.y_inst_list[index],
                "bag_size": torch.tensor(self.lengths[index], dtype=torch.long),
            }
        )

        return bag_dict

    def p_y1_given_h(self, index: int):
        """
        Compute the posterior probability P(Y=1|h) for a given bag.

        This method computes the Bayes optimal classifier prediction
        for the bag at the given index based on the known data generation process.

        Arguments:
            index: Index of the bag

        Returns:
            p_y1_given_h: Posterior probability P(Y=1|h)
        """
        h = self.H_split[index][:, 0 : self.K]
        S_i = self.lengths[index]

        # Compute P(h|Y=0)
        p_h_given_y0 = torch.prod(
            torch.stack([normal_pdf(h[j]) for j in range(S_i)])
        ) * (1.0 - self.p_y1)

        # Compute P(h|Y=1)
        p_u = (1 / (S_i - self.R + 1)) * torch.ones(size=(S_i - self.R + 1,))
        # p_h_given_u_y1.shape = (S_i - R + 1, S_i, K)
        p_h_given_u_y1 = torch.stack(
            [
                torch.stack(
                    [
                        torch.stack(
                            [
                                normal_pdf(h[j, k], self.mu + self.Delta)
                                if j >= u and j < (u + self.R)
                                else normal_pdf(h[j, k])
                                for k in range(self.K)
                            ]
                        )
                        for j in range(S_i)
                    ]
                )
                for u in range(S_i - self.R + 1)
            ]
        )
        p_h_given_y1 = (
            torch.sum(
                torch.prod(torch.prod(p_h_given_u_y1, dim=-1), dim=-1) * p_u, dim=-1
            )
            * self.p_y1
        )

        # Compute posterior P(Y=1|h) using Bayes' rule
        p_y1_given_h = p_h_given_y1 / (p_h_given_y0 + p_h_given_y1)

        return p_y1_given_h
