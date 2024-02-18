from abc import ABC, abstractmethod 

class W(ABC):

    @abstractmethod
    def kernels(self):
        pass
    
    @abstractmethod
    def W_eta_segments(self):
        pass
    
    @abstractmethod
    def loss_KL_segments(self):
        pass

    @abstractmethod
    def gradient_cov_segments(self):
        pass
