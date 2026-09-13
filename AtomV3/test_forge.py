"""Small synthetic checks for extraction, independent verification, and execution.

These fixtures contain no trained teacher results. They can run before the main
experiment without using held-out experiment outcomes to choose its protocol.
Run from the repository root: python -m unittest AtomV3.test_forge -v
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from AtomV3.compiler import (
    apply_program,
    fit_program,
    program_from_bytes,
    program_to_bytes,
)
from AtomV3.runtime import OperatorCache
from AtomV3.verify_archive import permutation_closure, verify_program


def arbitrary_operator(seed=19073):
    """Independent fixture: arbitrary permutations, not named task recipes."""
    rng = np.random.default_rng(seed)
    source = np.array([5, 2, 0, 4, 1, 3], dtype=np.uint8)
    table = np.stack([rng.permutation(10) for _ in range(6)]).astype(np.uint8)
    return {"source": source, "table": table}


def fixture_behavior(operator, inputs):
    """A slow, explicit fixture oracle, separate from the production executor."""
    return np.array(
        [[operator["table"][j][int(row[int(operator["source"][j])])]
          for j in range(6)] for row in inputs], dtype=np.uint8
    )


def samples(seed, count):
    return np.random.default_rng(seed).integers(0, 10, (count, 6), dtype=np.uint8)


class ExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.operator = arbitrary_operator()
        cls.calibration = samples(12031, 2048)
        cls.validation = samples(91841, 1024)

    def test_recovers_arbitrary_non_affine_bijective_tables_and_permutation(self):
        # Prove the fixture is more general than affine digit transformations.
        digits = np.arange(10)
        affine = [(a * digits + b) % 10
                  for a in (1, 3, 7, 9) for b in range(10)]
        self.assertTrue(any(
            not any(np.array_equal(row, candidate) for candidate in affine)
            for row in self.operator["table"]
        ))
        result = fit_program(
            self.calibration, fixture_behavior(self.operator, self.calibration),
            self.validation, fixture_behavior(self.operator, self.validation),
        )
        self.assertTrue(result["accepted_candidate"], result)
        np.testing.assert_array_equal(result["source"], self.operator["source"])
        np.testing.assert_array_equal(result["table"], self.operator["table"])
        independent_inputs = samples(55291, 257)
        np.testing.assert_array_equal(
            apply_program(result, independent_inputs),
            fixture_behavior(self.operator, independent_inputs),
        )

    def test_cross_coordinate_nonlinearity_fails_teacher_validation(self):
        def interacting(inputs):
            output = fixture_behavior(self.operator, inputs)
            # Depends nonlinearly on two independently varying input positions.
            output[:, 0] = (inputs[:, 0].astype(np.int64)
                            + inputs[:, 1].astype(np.int64) ** 2) % 10
            return output

        result = fit_program(
            self.calibration, interacting(self.calibration),
            self.validation, interacting(self.validation),
        )
        self.assertFalse(result["accepted_candidate"], result)
        self.assertLess(result["validation_agreement"], 0.5)

    def test_shuffled_teacher_labels_are_rejected(self):
        rng = np.random.default_rng(7937)
        calibration_targets = fixture_behavior(self.operator, self.calibration)
        validation_targets = fixture_behavior(self.operator, self.validation)
        result = fit_program(
            self.calibration, calibration_targets[rng.permutation(len(self.calibration))],
            self.validation, validation_targets,
        )
        self.assertFalse(result["accepted_candidate"], result)
        self.assertLess(result["validation_agreement"], 0.02)

    def test_validation_labels_cannot_change_the_fitted_operator(self):
        calibration_targets = fixture_behavior(self.operator, self.calibration)
        honest_validation = fixture_behavior(self.operator, self.validation)
        contradictory_validation = (honest_validation + 1) % 10
        honest = fit_program(self.calibration, calibration_targets,
                             self.validation, honest_validation)
        contradictory = fit_program(self.calibration, calibration_targets,
                                    self.validation, contradictory_validation)
        np.testing.assert_array_equal(honest["source"], contradictory["source"])
        np.testing.assert_array_equal(honest["table"], contradictory["table"])
        self.assertTrue(honest["accepted_candidate"])
        self.assertFalse(contradictory["accepted_candidate"])

    def test_systematic_wrong_teacher_is_not_rejected_by_extraction_alone(self):
        # The extractor must learn what a teacher does, without consulting the
        # independent specification. This teacher swaps 0/1 in the final digit;
        # it is internally consistent and representable, but is not identity.
        wrong = {"source": np.arange(6, dtype=np.uint8),
                 "table": np.tile(np.arange(10, dtype=np.uint8), (6, 1))}
        wrong["table"][5, [0, 1]] = wrong["table"][5, [1, 0]]
        result = fit_program(
            self.calibration, fixture_behavior(wrong, self.calibration),
            self.validation, fixture_behavior(wrong, self.validation),
        )
        self.assertTrue(result["accepted_candidate"], result)
        self.assertEqual(result["validation_agreement"], 1.0)
        # A separate verifier test below checks correctness against identity.
        np.testing.assert_array_equal(result["source"], wrong["source"])
        np.testing.assert_array_equal(result["table"], wrong["table"])


class ExecutionTests(unittest.TestCase):
    def test_exact_binary_roundtrip_preserves_arbitrary_behavior(self):
        original = arbitrary_operator(9081)
        payload = program_to_bytes(original)
        self.assertEqual(len(payload), 66)
        # An independently assembled byte fixture pins the portable format.
        expected = bytes(original["source"].tolist()) + bytes(
            value for row in original["table"].tolist() for value in row)
        self.assertEqual(payload, expected)
        restored = program_from_bytes(payload)
        self.assertEqual(program_to_bytes(restored), payload)
        x = samples(443, 101)
        np.testing.assert_array_equal(
            apply_program(restored, x), fixture_behavior(original, x))

    def test_corrupt_binary_payload_is_rejected(self):
        payload = program_to_bytes(arbitrary_operator())
        for invalid in (payload[:-1], payload + b"\x00", b"\xff" + payload[1:]):
            with self.subTest(length=len(invalid), first=invalid[0]):
                with self.assertRaises(ValueError):
                    program_from_bytes(invalid)

    def test_one_operator_cache_evicts_and_reloads_without_changing_execution(self):
        a, b = arbitrary_operator(111), arbitrary_operator(222)
        initial = samples(991, 73)
        expected = fixture_behavior(a, fixture_behavior(b, fixture_behavior(a, initial)))
        with tempfile.TemporaryDirectory(prefix="atom_forge_test_") as tmp:
            pa, pb = Path(tmp) / "a.bin", Path(tmp) / "b.bin"
            pa.write_bytes(program_to_bytes(a))
            pb.write_bytes(program_to_bytes(b))
            cache = OperatorCache(capacity=1)
            actual = initial
            for path in (pa, pb, pa):
                actual = cache.apply(path, actual)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(cache.counters["loads"], 3)
            self.assertEqual(cache.counters["evictions"], 2)
            self.assertEqual(cache.counters["bytes_read"], 198)
            self.assertEqual(cache.counters["peak_loaded_operator_payload_bytes"], 66)
            self.assertEqual(cache.counters["current_loaded_operator_payload_bytes"], 66)

    def test_repeated_cached_operator_is_a_hit(self):
        operator = arbitrary_operator(320)
        initial = samples(199, 17)
        with tempfile.TemporaryDirectory(prefix="atom_forge_test_") as tmp:
            path = Path(tmp) / "operator.bin"
            path.write_bytes(program_to_bytes(operator))
            cache = OperatorCache(capacity=1)
            first = cache.apply(path, initial)
            second = cache.apply(path, first)
            np.testing.assert_array_equal(
                second, fixture_behavior(operator, fixture_behavior(operator, initial)))
            self.assertEqual(cache.counters["loads"], 1)
            self.assertEqual(cache.counters["cache_hits"], 1)


class ClosureTests(unittest.TestCase):
    def test_reversal_is_absent_from_rotation_and_half_swap_subgroup(self):
        rotation = [1, 2, 3, 4, 5, 0]
        half_swap = [3, 4, 5, 0, 1, 2]
        closure = permutation_closure([rotation, half_swap])
        self.assertEqual(closure["count"], 6)
        self.assertFalse(closure["contains_reversal"])

    def test_adding_reversal_expands_to_dihedral_group(self):
        closure = permutation_closure([[1, 2, 3, 4, 5, 0], [5, 4, 3, 2, 1, 0]])
        self.assertEqual(closure["count"], 12)
        self.assertTrue(closure["contains_reversal"])


class IndependentVerificationTests(unittest.TestCase):
    def test_faithful_extraction_of_wrong_teacher_fails_independent_specification(self):
        x_cal, x_val = samples(775, 2048), samples(776, 1024)
        wrong = {"source": np.arange(6, dtype=np.uint8),
                 "table": np.tile(np.arange(10, dtype=np.uint8), (6, 1))}
        wrong["table"][5, [0, 1]] = wrong["table"][5, [1, 0]]
        candidate = fit_program(x_cal, fixture_behavior(wrong, x_cal),
                                x_val, fixture_behavior(wrong, x_val))
        self.assertTrue(candidate["accepted_candidate"])
        # Only this separate verifier sees the independent identity specification.
        verification = verify_program(
            candidate["source"], candidate["table"], "synthetic_identity",
            chunk_size=17, total_inputs=100, truth_fn=lambda inputs: inputs.copy(),
        )
        self.assertFalse(verification["exhaustive_pass"])
        self.assertFalse(verification["checked_inputs_pass"])
        self.assertEqual(verification["inputs_checked"], 100)
        self.assertEqual(verification["mismatches"], 20)
        self.assertIsNotNone(verification["first_counterexample"])

    def test_partial_verification_cannot_claim_full_domain_certificate(self):
        identity = {"source": np.arange(6, dtype=np.uint8),
                    "table": np.tile(np.arange(10, dtype=np.uint8), (6, 1))}
        verification = verify_program(
            identity["source"], identity["table"], "synthetic_identity",
            chunk_size=17, total_inputs=100, truth_fn=lambda inputs: inputs.copy(),
        )
        self.assertTrue(verification["checked_inputs_pass"])
        self.assertEqual(verification["mismatches"], 0)
        self.assertFalse(verification["is_complete_domain"])
        self.assertFalse(verification["exhaustive_pass"])


if __name__ == "__main__":
    unittest.main()
