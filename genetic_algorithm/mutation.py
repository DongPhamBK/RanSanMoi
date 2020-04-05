# đã xong mục này
import numpy as np
from typing import List, Union, Optional
from .individual import Individual

#đột biến
def gaussian_mutation(chromosome: np.ndarray, prob_mutation: float, 
                      mu: List[float] = None, sigma: List[float] = None,
                      scale: Optional[float] = None) -> None:
    """
    Thực hiện một đột biến gaussian cho mỗi gen trong một cá thể có xác suất, prob_muting.
     Nếu mu và sigma được định nghĩa thì phân phối gauss sẽ được rút ra từ đó,
     nếu không, nó sẽ được rút ra từ N (0, 1) cho hình dạng của cá nhân.
    """
    # Xác định gen nào sẽ bị đột biến
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    # Nếu mu và sigma được xác định, tạo phân phối gaussian xung quanh mỗi một
    if mu and sigma:
        gaussian_mutation = np.random.normal(mu, sigma)
    # Nếu không xung quanh trung tâm N(0,1)
    else:
        gaussian_mutation = np.random.normal(size=chromosome.shape)
    
    if scale:
        gaussian_mutation[mutation_array] *= scale

    # Update
    chromosome[mutation_array] += gaussian_mutation[mutation_array]

def random_uniform_mutation(chromosome: np.ndarray, prob_mutation: float,
                            low: Union[List[float], float],
                            high: Union[List[float], float]
                            ) -> None:
    """
    Đột biến ngẫu nhiên mỗi gen trong một cá thể với xác suất, prob_muting.
    Nếu một gen được chọn để đột biến, nó sẽ được gán một giá trị với xác suất đồng nhất giữa [low, high).

    @Note [low, high) được xác định cho mỗi gen để giúp có được đầy đủ các giá trị có thể
    @TODO: Eq 11.4
    """
    assert type(low) == type(high), 'low and high must have the same type'
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    if isinstance(low, list):
        uniform_mutation = np.random.uniform(low, high)
    else:
        uniform_mutation = np.random.uniform(low, high, size=chromosome.shape)
    chromosome[mutation_array] = uniform_mutation[mutation_array]

def uniform_mutation_with_respect_to_best_individual(chromosome: np.ndarray, best_chromosome: np.ndarray, prob_mutation: float) -> None:
    """
    Đột biến ngẫu nhiên mỗi gen trong một cá thể có xác suất, đột biến đầu dò. Nếu một gen được
     chọn để đột biến, nó sẽ di chuyển về phía gen từ cá thể tốt nhất.

    @TODO: Eq 11.6
    """
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    uniform_mutation = np.random.uniform(size=chromosome.shape)
    chromosome[mutation_array] += uniform_mutation[mutation_array] * (best_chromosome[mutation_array] - chromosome[mutation_array])

def cauchy_mutation(individual: np.ndarray, scale: float) -> np.ndarray:
    pass

def exponential_mutation(chromosome: np.ndarray, xi: Union[float, np.ndarray], prob_mutation: float) -> None:
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    # Fill xi if necessary
    if not isinstance(xi, np.ndarray):
        xi_val = xi
        xi = np.empty(chromosome.shape)
        xi.fill(xi_val)

    # Thay xi vì vậy chúng ta nhận được E(0, 1), thay vì E(0, xi)
    xi_div = 1.0 / xi
    xi.fill(1.0)
    
    # Eq 11.17
    y = np.random.uniform(size=chromosome.shape)
    x = np.empty(chromosome.shape)
    x[y <= 0.5] = (1.0 / xi[y <= 0.5]) * np.log(2 * y[y <= 0.5])
    x[y > 0.5] = -(1.0 / xi[y > 0.5]) * np.log(2 * (1 - y[y > 0.5]))

    # Eq 11.16
    delta = np.empty(chromosome.shape)
    delta[mutation_array] = (xi[mutation_array] / 2.0) * np.exp(-xi[mutation_array] * np.abs(x[mutation_array]))

    # Cập nhật delta sao cho E(0, xi) = (1 / xi) * E(0 , 1)
    delta[mutation_array] = xi_div[mutation_array] * delta[mutation_array]

    # Cập nhật cá nhân
    chromosome[mutation_array] += delta[mutation_array]

def mmo_mutation(chromosome: np.ndarray, prob_mutation: float) -> None:
    from scipy import stats
    mutation_array = np.random.random(chromosome.shape) < prob_mutation
    normal = np.random.normal(size=chromosome.shape)  # Eq 11.21
    cauchy = stats.cauchy.rvs(size=chromosome.shape)  # Eq 11.22
    
    # Eq 11.20
    delta = np.empty(chromosome.shape)
    delta[mutation_array] = normal[mutation_array] + cauchy[mutation_array]

    # Cập nhật cá nhân
    chromosome[mutation_array] += delta[mutation_array]