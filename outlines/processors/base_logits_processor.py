from abc import abstractmethod
from typing import List, Protocol, Union

import numpy as np
import torch
from numpy.typing import NDArray


def is_mlx_array(logits):
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(logits, mx.array)


class OutlinesLogitsProcessor(Protocol):
    """
    Base class for logits processors which normalizes types of logits:
    - ndarray (used by llama-cpp-python), converted to torch.Tensor
    - torch.Tensor (used by everything else)

    Normalization of types and conversion to torch.Tensor
    doesn't move memory, it just casts the type.

    Normalizing the types allows all logits processors inheriting from this class
    to implement a single method for all the business logit: `process_logits()`
    """

    @abstractmethod
    def process_logits(
        self, input_ids: List[int], logits: torch.Tensor
    ) -> torch.Tensor:
        ...

    def _batch_process_logits(self, input_ids, logits: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
             input_ids: List[int] | List[List[int]]
             logits: 1D tensor | 2D tensor
        Returns:
             Augmented logits: 1D tensor | 2D tensor

        If given a list of input sequences and a 2D tensor, handle batch request,
        otherwise handle request normally.
        """
        logits_in_batch_mode = len(logits.shape) > 1
        input_ids_in_batch_mode = isinstance(input_ids[0], list)

        if logits_in_batch_mode != input_ids_in_batch_mode:
            raise TypeError(
                f"Logits and input_ids incompatible: "
                f"logits bach mode={logits_in_batch_mode}, "
                f"input ids in batch mode={input_ids_in_batch_mode}"
            )

        if logits_in_batch_mode:
            for i in range(logits.shape[0]):
                single_input_ids: List[int] = input_ids[i]
                logits[i, :] = self.process_logits(single_input_ids, logits[i, :])
            return logits
        else:
            return self.process_logits(input_ids, logits)

    def __call__(
        self,
        input_ids: Union[NDArray[np.int64], List[int], torch.Tensor],
        logits: Union[NDArray[np.float32], torch.Tensor],
    ) -> Union[NDArray[np.int64], torch.Tensor]:
        """
        Apply logits processor
        Unify type
        - convert input_ids: either ndarray, List[int], or Tensor -> List[int]
        - convert logits: either ndarray, mlx array, Tensor -> Tensor
        Call process_logits() to perform business logic
        """
        with torch.no_grad():
            if not isinstance(input_ids, list):
                input_ids = input_ids.tolist()

            if isinstance(logits, np.ndarray):
                # Unify type, convert numpy array to Tensor
                # from_numpy and .numpy() don't copy the data, it uses the same memory address
                torch_logits = torch.from_numpy(logits)
                processed_torch_logits = self._batch_process_logits(
                    input_ids, torch_logits
                )
                return processed_torch_logits.detach().numpy()

            elif isinstance(logits, torch.Tensor):
                return self._batch_process_logits(input_ids, logits)

            elif is_mlx_array(logits):
                # mlx -> torch -> mlx conversion docs:
                # https://ml-explore.github.io/mlx/build/html/usage/numpy.html
                import mlx.core as mx

                torch_logits = torch.from_dlpack(logits)
                processed_torch_logits = self._batch_process_logits(
                    input_ids, torch_logits
                )

                # numpy doesn't support bfloat16, mlx doesn't support direct conversion from torch
                logits_float32_numpy = processed_torch_logits.float().numpy()
                return mx.array(logits_float32_numpy)

            else:
                raise TypeError(
                    "LogitsProcessor must be called with either np.NDArray"
                    ", torch.Tensor, or mlx.core.array typed logits"
                )
