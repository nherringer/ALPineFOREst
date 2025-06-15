from pathlib import Path

from alpfore.encoder import SystemEncoder
from alpfore.loaders import LAMMPSDumpLoader, COLVARLoader
from alpfore.core.trajectory_interface import Trajectory
from alpfore.evaluators import CGDNAHybridizationEvaluator, DeltaDeltaGEvaluator
from typing import Callable, Iterable, Tuple, List
import numpy as np

class Pipeline:
    def __init__(self, encoder_config_path: str, candidate_list: List[Tuple]):
        """
        Central object to orchestrate the ALPineFOREst pipeline stages.

        Parameters:
        - encoder_config_path: path to JSON file with scaling and vocabulary
        - candidate_list: List of (seq, ssl, lsl, sgd) Tuples
        """
        self.encoder = SystemEncoder.from_json(encoder_config_path)
        self.candidate_list = candidate_list
        self.trajectories: List[Trajectory] = []

    def evaluate_candidate_list_ddg(
        self,
        candidate_list: List[Tuple],
        cand_list_trjs: Iterable[Trajectory],
        cv_cutoffs: List[np.ndarray],
        ddg_eval_factory: Callable[..., DeltaDeltaGEvaluator],
    ):
        results = []
        for key, traj, cutoff in zip(self.candidate_list, cand_list_trjs, cv_cutoffs):
            ddg_eval = ddg_eval_factory(key, traj.run_dir, cutoff)
            ddg, sem = ddg_eval.evaluate(traj, cutoff)
            results.append((ddg, sem))
        return results

    def encode_and_load(self, loader_type="lammps", **loader_kwargs):
        print("loader_type =", loader_type)
        print("kwargs =", loader_kwargs)

        if loader_type == "lammps":
            from alpfore.loaders.lammps_loader import LAMMPSDumpLoader as loader_cls

            required_keys = ["struct_pattern", "traj_pattern"]
            optional_keys = ["stride", "n_equil_drop", "use_parallel", "n_jobs"]

        elif loader_type == "colvar":
            from alpfore.loaders.colvar_loader import COLVARLoader as loader_cls

            required_keys = ["colvar_pattern"]
            optional_keys = ["names"]

        else:
            raise ValueError(f"Unknown loader_type: {loader_type}")

        # Filter only keys relevant to the selected loader
        selected_kwargs = {
            k: loader_kwargs[k]
            for k in required_keys + optional_keys
            if k in loader_kwargs
        }

        # Sanity check: raise if any required key is missing
        for k in required_keys:
            if k not in selected_kwargs:
                raise ValueError(
                    f"Missing required argument for loader_type='{loader_type}': {k}"
                )

        # Final dispatch
        self.trajectories = list(
            loader_cls.from_candidate_list(
                self.candidate_list, encoder=self.encoder, **selected_kwargs
            )
        )

    def evaluate_ddg(self, cv_cutoffs=20, walker_ids=[0, 1, 2], bandwidth=2.5):
        """Evaluate ddG and SEM values for each candidate system."""
        return evaluate_candidate_list_ddg(
            candidate_list=self.candidate_list,
            cand_list_trjs=self.trajectories,
            cv_cutoffs=cv_cutoffs,
            ddg_eval_factory=lambda key, run_dir, cv_cutoffs: DeltaDeltaGEvaluator(
                key,
                run_dir,
                walker_ids=walker_ids,
                cv_cutoff=cv_cutoffs,
                bandwidth=bandwidth,
            ),
        )
