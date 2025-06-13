from pathlib import Path

from alpfore.encoder import SystemEncoder
from alpfore.loaders import LAMMPSDumpLoader
from alpfore.core.loader import Trajectory
from alpfore.evaluators import CGDNAHybridizationEvaluator, DeltaDeltaGEvaluator
from typing import Callable, Iterable, Tuple, List

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
        candidate_list: List[Tuple],
        cand_list_trjs: Iterable[Trajectory],
        hybrid_eval_factory: Callable[..., CGDNAHybridizationEvaluator],
        ddg_eval_factory: Callable[..., DeltaDeltaGEvaluator],
    ):
        results = []
        for key, traj in zip(candidate_list, cand_list_trjs):
            hybrid_eval = hybrid_eval_factory(*key)
            hybrid_ratios = hybrid_eval.evaluate(traj)
            ddg_eval = ddg_eval_factory(key, traj.run_dir, hybrid_ratios)
            ddg, sem = ddg_eval.evaluate(traj, hybrid_ratios)
            results.append((ddg, sem))
        return results

    def encode_and_load(self, **loader_kwargs):
        """Lazy-load and store trajectories for each candidate system."""
        self.trajectories = list(
            LAMMPSDumpLoader.from_candidate_list(
                self.candidate_list,
                encoder=self.encoder,
                **loader_kwargs
            )
        )

    def evaluate_ddg(self, walker_ids=[0, 1, 2], ratio_cutoff=0.8, bandwidth=2.5):
        """Evaluate ddG and SEM values for each candidate system."""
        return evaluate_candidate_list_ddg(
            candidate_list=self.candidate_list,
            cand_list_trjs=self.trajectories,
            hybrid_eval_factory=lambda seq, ssl, lsl, sgd, *_: CGDNAHybridizationEvaluator(
                1, sgd, ssl, lsl, len(seq)
            ),
            ddg_eval_factory=lambda key, run_dir, ratios: DeltaDeltaGEvaluator(
                key,
                run_dir,
                ratios,
                walker_ids=walker_ids,
                ratio_cutoff=ratio_cutoff,
                bandwidth=bandwidth
            )
        )

