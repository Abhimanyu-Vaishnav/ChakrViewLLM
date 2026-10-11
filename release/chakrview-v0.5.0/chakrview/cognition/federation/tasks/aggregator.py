"""
Deterministic Result Aggregator for Distributed Tasks (Step 38).

Provides typed, deterministic result aggregation strategies to merge
discrete WorkUnit outputs into a single coherent task result.
"""

from typing import Dict, List, Any, Callable, Optional

from chakrview.cognition.federation.tasks.models import (
    DistributedTask,
    WorkUnit,
    WorkUnitState,
    AggregationStrategy,
)
from chakrview.cognition.federation.tasks.errors import (
    ResultAggregationError,
)


class TaskResultAggregator:
    """
    Merges validated WorkUnit results according to the declared AggregationStrategy.
    """

    def __init__(self) -> None:
        self._custom_strategies: Dict[str, Callable[[List[WorkUnit]], Any]] = {}

    def register_custom_strategy(
        self,
        name: str,
        handler: Callable[[List[WorkUnit]], Any],
    ) -> None:
        """Register a custom result aggregation handler."""
        self._custom_strategies[name] = handler

    def aggregate_results(self, task: DistributedTask) -> Any:
        """
        Merge results of all completed work units into a single final task result.
        
        Raises:
            ResultAggregationError if units are incomplete, types mismatch, or strategy fails.
        """
        if not task.work_units:
            return None

        # Verify all units have completed
        uncompleted = [u.unit_id for u in task.work_units if u.state != WorkUnitState.COMPLETED]
        if uncompleted:
            raise ResultAggregationError(
                f"Cannot aggregate results for task {task.task_id}: units {uncompleted} are not COMPLETED"
            )

        # Sort units by sequence
        sorted_units = sorted(task.work_units, key=lambda u: u.sequence)
        strategy = task.aggregation_strategy

        try:
            if strategy == AggregationStrategy.CONCATENATE:
                return self._aggregate_concatenate(sorted_units)
            elif strategy == AggregationStrategy.MERGE_DICT:
                return self._aggregate_merge_dict(sorted_units)
            elif strategy == AggregationStrategy.REDUCE_SUM:
                return self._aggregate_reduce_sum(sorted_units)
            elif strategy == AggregationStrategy.FIRST_SUCCESS:
                return self._aggregate_first_success(sorted_units)
            elif strategy == AggregationStrategy.CUSTOM:
                if task.name in self._custom_strategies:
                    return self._custom_strategies[task.name](sorted_units)
                raise ResultAggregationError(
                    f"No custom aggregation strategy registered for task name '{task.name}'"
                )
            else:
                raise ResultAggregationError(f"Unsupported aggregation strategy '{strategy}'")
        except Exception as e:
            if isinstance(e, ResultAggregationError):
                raise
            raise ResultAggregationError(f"Result aggregation failed for task {task.task_id}: {str(e)}") from e

    def _aggregate_concatenate(self, units: List[WorkUnit]) -> Any:
        """Concatenate strings or lists in sequence order."""
        first_data = units[0].result.result_data if units[0].result else None
        if isinstance(first_data, str):
            parts = []
            for u in units:
                data = u.result.result_data if u.result else ""
                parts.append(str(data))
            return "".join(parts)
        elif isinstance(first_data, list):
            merged_list = []
            for u in units:
                data = u.result.result_data if u.result else []
                if isinstance(data, list):
                    merged_list.extend(data)
                else:
                    merged_list.append(data)
            return merged_list
        else:
            # Fallback to list of all results
            return [u.result.result_data if u.result else None for u in units]

    def _aggregate_merge_dict(self, units: List[WorkUnit]) -> Dict[str, Any]:
        """Merge dict results across units."""
        merged: Dict[str, Any] = {}
        for u in units:
            data = u.result.result_data if u.result else {}
            if isinstance(data, dict):
                merged.update(data)
            else:
                raise ResultAggregationError(
                    f"Expected dict result for MERGE_DICT, got {type(data)} in unit {u.unit_id}"
                )
        return merged

    def _aggregate_reduce_sum(self, units: List[WorkUnit]) -> float:
        """Sum numeric results across units."""
        total = 0.0
        for u in units:
            data = u.result.result_data if u.result else 0
            if isinstance(data, (int, float)):
                total += float(data)
            else:
                raise ResultAggregationError(
                    f"Expected numeric result for REDUCE_SUM, got {type(data)} in unit {u.unit_id}"
                )
        return total

    def _aggregate_first_success(self, units: List[WorkUnit]) -> Any:
        """Return the result of the first successful unit."""
        for u in units:
            if u.result and u.result.status == "SUCCESS":
                return u.result.result_data
        return None
