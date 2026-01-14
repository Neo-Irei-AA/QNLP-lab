# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Loss Functions
==============
Module containing loss functions to train lambeq's quantum models.

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jax import numpy as jnp
    from types import ModuleType


class LossFunction(ABC):
    """Loss function base class.

    Attributes
    ----------
    backend : ModuleType
        The module to use for array numerical functions.
         Either numpy or jax.numpy.

    """

    def __init__(self, use_jax: bool = False) -> None:
        """Initialise a loss function.

        Parameters
        ----------
        use_jax : bool, default: False
            Whether to use the Jax variant of numpy as `backend`.

        """

        self.backend: ModuleType

        if use_jax:
            from jax import numpy as jnp
            self.backend = jnp
        else:
            self.backend = np

    def _match_shapes(self,
                      y1: np.ndarray | jnp.ndarray,
                      y2: np.ndarray | jnp.ndarray) -> None:
        if y1.shape != y2.shape:
            raise ValueError('Provided arrays must be of equal shape. Got '
                             f'arrays of shape {y1.shape} and {y2.shape}.')

    def _smooth_and_normalise(self,
                              y: np.ndarray | jnp.ndarray,
                              epsilon: float
                              ) -> np.ndarray | jnp.ndarray:

        y_smoothed = y + epsilon

        l1_norms: np.ndarray | jnp.ndarray = self.backend.linalg.norm(
                                                y_smoothed,
                                                ord=1,
                                                axis=1,
                                                keepdims=True)

        return y_smoothed / l1_norms

    @abstractmethod
    def calculate_loss(self,
                       y_pred: np.ndarray | jnp.ndarray,
                       y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of loss function."""

    def __call__(self,
                 y_pred: np.ndarray | jnp.ndarray,
                 y_true: np.ndarray | jnp.ndarray) -> float:
        return self.calculate_loss(y_pred, y_true)

class CELoss(LossFunction):
    """Multiclass cross-entropy loss function.

    Parameters
    ----------
    y_pred: np.ndarray or jnp.ndarray
        Predicted labels from model. Expected to be of shape
        [batch_size, n_classes], where each row is a probability
        distribution.
    y_true: np.ndarray or jnp.ndarray
        Ground truth labels. Expected to be of shape
        [batch_size, n_classes], where each row is a one-hot vector.
    
    """
    
    def __init__(self,
                 use_jax: bool = False,
                 epsilon: float = 1e-9) -> None:
        """Initialise a multiclass cross-entropy loss function.
        
        Parameters
        ----------
        use_jax : bool, default: False
            Whether to use the Jax variant of numpy.
        epsilon : float, default: 1e-9
            Smoothing constant used to prevent calculating log(0).

        """

        self._epsilon = epsilon
        super().__init__(use_jax)
    
    def calculate_loss(self,
                y_pred: np.ndarray | jnp.ndarray,
                y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of CE loss function."""

        self._match_shapes(y_pred, y_true)

    # y_pred_smoothed = self._smooth_and_normalise(y_pred, self._epsilon)
    
    # Lambdify
    #np_circuits = [c.lambdify(*parameters)(*tensors) for c in train_circuits]
        ptrue= []
        for s  in range(len(y_true)):
            s0=y_true[s,0,0]
            s1=y_true[s,0,1]
            s2=y_true[s,1,0]
            s3=y_true[s,1,1]
            #ptrue.append(np.array([s0,s1,s2,s3]))
            ptrue.append(np.array([s0*s2,s0*s3,s1*s2,s1*s3]))

        qtrue=np.array(ptrue)

        ppredic= [ ]
        for s in range(len(y_pred)):
            p0=abs(y_pred[s,0,0])**2
            p1=abs(y_pred[s,0,1])**2
            p2=abs(y_pred[s,1,0])**2
            p3=abs(y_pred[s,1,1])**2
            q0=p0*p2
            q1=p0*p3
            q2=p1*p2
            q3=p1*p3
            pp=p0+p1+p2+p3
            p0n=p0/pp
            p1n=p1/pp
            p2n=p2/pp
            p3n=p3/pp
            ppredic.append(np.array([p0n,p1n,p2n,p3n]))

        ppre=np.array(ppredic)

        CrossEntropy=0.0

        for s  in range(len(y_true)):
            p=ppre[s,0]*qtrue[s,0]+ppre[s,1]*qtrue[s,1]+ppre[s,2]*qtrue[s,2]+ppre[s,3]*qtrue[s,3]
            #p=numpy.dot(ppre[s,:], ptrue[s,:])
            CrossEntropy= CrossEntropy+ self.backend.log(p)
            CE=-CrossEntropy/len(y_true)
        
        return CE


class CELoss8(LossFunction):
    """Multiclass cross-entropy loss function.
    
    Parameters
    ----------
    y_pred: np.ndarray or jnp.ndarray
        Predicted labels from model. Expected to be of shape
        [batch_size, n_classes], where each row is a probability
        distribution.
    y_true: np.ndarray or jnp.ndarray
        Ground truth labels. Expected to be of shape
        [batch_size, n_classes], where each row is a one-hot vector.
    
    """

    def __init__(self,
                 use_jax: bool = False,
                 epsilon: float = 1e-9) -> None:
        """Initialise a multiclass cross-entropy loss function.
        
        Parameters
        ----------
        use_jax : bool, default: False
            Whether to use the Jax variant of numpy.
        epsilon : float, default: 1e-9
            Smoothing constant used to prevent calculating log(0).
        
        """

        self._epsilon = epsilon
        
        super().__init__(use_jax)
    
    def calculate_loss(self,
                y_pred: np.ndarray | jnp.ndarray,
                y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of CE loss function."""

        self._match_shapes(y_pred, y_true)
    
    # y_pred_smoothed = self._smooth_and_normalise(y_pred, self._epsilon)
    
    # Lambdify
    #np_circuits = [c.lambdify(*parameters)(*tensors) for c in train_circuits]
        ptrue= []
        for s  in range(len(y_true)):
            s0=y_true[s,0,0,0]
            s1=y_true[s,0,0,1]
            s2=y_true[s,0,1,0]
            s3=y_true[s,0,1,1]
            s4=y_true[s,1,0,0]
            s5=y_true[s,1,0,1]
            s6=y_true[s,1,1,0]
            s7=y_true[s,1,1,1]
            ptrue.append(np.array([s0, s1, s2, s3, s4, s5, s6, s7]))

        qtrue=np.array(ptrue)
        
        ppredic= [ ]
        for s in range(len(y_pred)):
            p0=y_pred[s,0,0,0]
            p1=y_pred[s,0,0,1]
            p2=y_pred[s,0,1,0]
            p3=y_pred[s,0,1,1]
            p4=y_pred[s,1,0,0]
            p5=y_pred[s,1,0,1]
            p6=y_pred[s,1,1,0]
            p7=y_pred[s,1,1,1]
            q0=abs(p0)**2
            q1=abs(p1)**2
            q2=abs(p2)**2
            q3=abs(p3)**2
            q4=abs(p4)**2
            q5=abs(p5)**2
            q6=abs(p6)**2
            q7=abs(p7)**2
            qq=q0+q1+q2+q3+q4+q5+q6+q7
            q0n=q0/qq
            q1n=q1/qq
            q2n=q2/qq
            q3n=q3/qq
            q4n=q4/qq
            q5n=q5/qq
            q6n=q6/qq
            q7n=q7/qq
            ppredic.append(np.array([q0n,q1n,q2n,q3n,q4n,q5n,q6n,q7n]))

        ppre=np.array(ppredic)

        CrossEntropy=0.0

        for s  in range(len(y_true)):
            p=ppre[s,0]*qtrue[s,0]+ppre[s,1]*qtrue[s,1]+ppre[s,2]*qtrue[s,2]+ppre[s,3]*qtrue[s,3]+ \
            ppre[s,4]*qtrue[s,4]+ppre[s,5]*qtrue[s,5]+ppre[s,6]*qtrue[s,6]+ppre[s,7]*qtrue[s,7]
            #p=numpy.dot(ppre[s,:], ptrue[s,:])
            CrossEntropy= CrossEntropy+ self.backend.log(p)
            CE=-CrossEntropy/len(y_true)

        return CE


class CELoss4(LossFunction):

    def __init__(self,
                 use_jax: bool = False,
                 epsilon: float = 1e-9) -> None:
        """Initialise a multiclass cross-entropy loss function.
        
        Parameters
        ----------
        use_jax : bool, default: False
            Whether to use the Jax variant of numpy.
        epsilon : float, default: 1e-9
            Smoothing constant used to prevent calculating log(0).
        
        """
        
        self._epsilon = epsilon
        
        super().__init__(use_jax)
    
    def calculate_loss(self,
                y_pred: np.ndarray | jnp.ndarray,
                y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of CE loss function."""

        self._match_shapes(y_pred, y_true)
    
    # y_pred_smoothed = self._smooth_and_normalise(y_pred, self._epsilon)
    
    # Lambdify
    #np_circuits = [c.lambdify(*parameters)(*tensors) for c in train_circuits]
        ptrue= []
        for s  in range(len(y_true)):
            s0=y_true[s,0,0]
            s1=y_true[s,0,1]
            s2=y_true[s,1,0]
            s3=y_true[s,1,1]
            ptrue.append(np.array([s0, s1, s2, s3]))

        qtrue=np.array(ptrue)

        ppredic= [ ]
        for s in range(len(y_pred)):
            p0=y_pred[s,0,0]
            p1=y_pred[s,0,1]
            p2=y_pred[s,1,0]
            p3=y_pred[s,1,1]
            q0=abs(p0)**2
            q1=abs(p1)**2
            q2=abs(p2)**2
            q3=abs(p3)**2
            qq=q0+q1+q2+q3
            q0n=q0/qq
            q1n=q1/qq
            q2n=q2/qq
            q3n=q3/qq
            ppredic.append(np.array([q0n,q1n,q2n,q3n]))

        ppre=np.array(ppredic)

        CrossEntropy=0.0

        for s  in range(len(y_true)):
            p=ppre[s,0]*qtrue[s,0]+ppre[s,1]*qtrue[s,1]+ppre[s,2]*qtrue[s,2]+ppre[s,3]*qtrue[s,3]
            #p=numpy.dot(ppre[s,:], ptrue[s,:])
            CrossEntropy= CrossEntropy+ self.backend.log(p)
            CE=-CrossEntropy/len(y_true)
        
        return CE


class CrossEntropyLoss(LossFunction):
    """Multiclass cross-entropy loss function.

    Parameters
    ----------
    y_pred: np.ndarray or jnp.ndarray
        Predicted labels from model. Expected to be of shape
        [batch_size, n_classes], where each row is a probability
        distribution.
    y_true: np.ndarray or jnp.ndarray
        Ground truth labels. Expected to be of shape
        [batch_size, n_classes], where each row is a one-hot vector.

    """

    def __init__(self,
                 use_jax: bool = False,
                 epsilon: float = 1e-9) -> None:
        """Initialise a multiclass cross-entropy loss function.

        Parameters
        ----------
        use_jax : bool, default: False
            Whether to use the Jax variant of numpy.
        epsilon : float, default: 1e-9
            Smoothing constant used to prevent calculating log(0).

        """

        self._epsilon = epsilon

        super().__init__(use_jax)

    def calculate_loss(self,
                       y_pred: np.ndarray | jnp.ndarray,
                       y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of CE loss function."""

        self._match_shapes(y_pred, y_true)

        y_pred_smoothed = self._smooth_and_normalise(y_pred, self._epsilon)

        entropies = y_true * self.backend.log(y_pred_smoothed)
        loss_val: float = -self.backend.sum(entropies) / len(y_true)

        return loss_val


class BinaryCrossEntropyLoss(CrossEntropyLoss):
    """Binary cross-entropy loss function.

    Parameters
    ----------
    y_pred: np.ndarray or jnp.ndarray
        Predicted labels from model. When `sparse` is `False`,
        expected to be of shape [batch_size, 2], where each row is a
        probability distribution. When `sparse` is `True`, expected to
        be of shape [batch_size, ] where each element indicates P(1).
    y_true: np.ndarray or jnp.ndarray
        Ground truth labels. When `sparse` is `False`, expected
        to be of shape [batch_size, 2], where each row is a one-hot
        vector. When `sparse` is `True`, expected to be of shape
        [batch_size, ] where each element is an integer indicating
        class label.

    """

    def __init__(self,
                 sparse: bool = False,
                 use_jax: bool = False,
                 epsilon: float = 1e-9) -> None:
        """Initialise a binary cross-entropy loss function.

        Parameters
        ----------
        sparse : bool, default: False
            If True, each input element indicates P(1), else the
             probability distribution over classes is expected.
        use_jax : bool, default: False
            Whether to use the Jax variant of numpy.
        epsilon : float, default: 1e-9
            Smoothing constant used to prevent calculating log(0).

        """

        self._sparse = sparse
        super().__init__(use_jax, epsilon)

    def calculate_loss(self,
                       y_pred: np.ndarray | jnp.ndarray,
                       y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of BCE loss function."""

        if self._sparse:
            # For numerical stability, it is convenient to reshape the
            #  sparse input to a dense representation.

            self._match_shapes(y_pred, y_true)

            y_pred_dense = self.backend.stack((1 - y_pred, y_pred)).T
            y_true_dense = self.backend.stack((1 - y_true, y_true)).T

            return super().calculate_loss(y_pred_dense, y_true_dense)
        else:
            return super().calculate_loss(y_pred, y_true)


class MSELoss(LossFunction):
    """Mean squared error loss function.

    Parameters
    ----------
    y_pred: np.ndarray or jnp.ndarray
        Predicted values from model. Shape must match y_true.
    y_true: np.ndarray or jnp.ndarray
        Ground truth values.

    """

    def calculate_loss(self,
                       y_pred: np.ndarray | jnp.ndarray,
                       y_true: np.ndarray | jnp.ndarray) -> float:
        """Calculate value of MSE loss function."""

        self._match_shapes(y_pred, y_true)

        return float(self.backend.mean((y_pred - y_true) ** 2))
