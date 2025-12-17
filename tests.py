"""
unittest.py

Unit tests for the Yaman FWM / OPA toy project.

How to run:
    python -m unittest -v tests.py

Notes:
- These are *unit* tests: they validate small pieces (config/params/integrators/model RHS).
- A tiny pipeline test is included to ensure modules connect without crashing.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import config
import constants
import integrators
import parameters
import simulation
import yaman_model


class TestConfig(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        cfg = config.default_simulation_config()
        # Should not raise
        config.validate_config(cfg)

        self.assertGreater(cfg.z_max, 0.0)
        self.assertGreater(cfg.dz, 0.0)
        self.assertEqual(cfg.integrator.lower(), "rk4")
        self.assertGreater(cfg.save_every, 0)

    def test_validate_config_rejects_invalid(self) -> None:
        base = config.default_simulation_config()

        with self.assertRaises(ValueError):
            config.validate_config(config.SimulationConfig(
                z_max=0.0,
                dz=base.dz,
                integrator=base.integrator,
                save_every=base.save_every,
                check_nan=base.check_nan,
                verbose=base.verbose,
            ))

        with self.assertRaises(ValueError):
            config.validate_config(config.SimulationConfig(
                z_max=base.z_max,
                dz=0.0,
                integrator=base.integrator,
                save_every=base.save_every,
                check_nan=base.check_nan,
                verbose=base.verbose,
            ))

        with self.assertRaises(ValueError):
            config.validate_config(config.SimulationConfig(
                z_max=1.0,
                dz=2.0,
                integrator=base.integrator,
                save_every=base.save_every,
                check_nan=base.check_nan,
                verbose=base.verbose,
            ))

        with self.assertRaises(ValueError):
            config.validate_config(config.SimulationConfig(
                z_max=base.z_max,
                dz=base.dz,
                integrator="euler",
                save_every=base.save_every,
                check_nan=base.check_nan,
                verbose=base.verbose,
            ))

        with self.assertRaises(ValueError):
            config.validate_config(config.SimulationConfig(
                z_max=base.z_max,
                dz=base.dz,
                integrator=base.integrator,
                save_every=0,
                check_nan=base.check_nan,
                verbose=base.verbose,
            ))


class TestConstants(unittest.TestCase):
    def test_speed_of_light_positive(self) -> None:
        self.assertIsInstance(constants.c, float)
        self.assertGreater(constants.c, 0.0)


class TestParameters(unittest.TestCase):
    def test_make_fiber_params_validation(self) -> None:
        beta = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        fp = parameters.make_fiber_params(gamma=1.3, alpha=0.0, beta=beta)
        self.assertEqual(fp.beta.shape, (4,))
        self.assertTrue(np.issubdtype(fp.beta.dtype, np.floating))

        with self.assertRaises(ValueError):
            parameters.make_fiber_params(gamma=1.3, alpha=0.0, beta=np.array([1.0, 2.0, 3.0]))

    def test_make_wave_params_validation(self) -> None:
        omega = np.array([1.0, 1.1, 1.2, 1.3], dtype=float)
        p_in = np.array([0.5, 0.5, 0.0, 0.0], dtype=float)
        wp = parameters.make_wave_params(omega=omega, P_in=p_in)
        self.assertEqual(wp.omega.shape, (4,))
        self.assertEqual(wp.P_in.shape, (4,))

        with self.assertRaises(ValueError):
            parameters.make_wave_params(omega=np.array([1.0, 2.0]), P_in=p_in)

        with self.assertRaises(ValueError):
            parameters.make_wave_params(omega=omega, P_in=np.array([0.1, 0.2, 0.3]))

        with self.assertRaises(ValueError):
            parameters.make_wave_params(omega=omega, P_in=np.array([0.1, -0.2, 0.3, 0.4]))

    def test_make_initial_conditions_are_sqrt_power(self) -> None:
        p_in = np.array([0.0, 1e-6, 1.0, 4.0], dtype=float)
        ic = parameters.make_initial_conditions(p_in)

        self.assertEqual(ic.A0.shape, (4,))
        self.assertTrue(np.iscomplexobj(ic.A0))

        recovered_p = np.abs(ic.A0) ** 2
        np.testing.assert_allclose(recovered_p, p_in, rtol=0.0, atol=1e-15)

    def test_make_model_params_container(self) -> None:
        fiber = parameters.make_fiber_params(1.3, 0.0, np.ones(4))
        waves = parameters.make_wave_params(np.ones(4), np.zeros(4))
        ic = parameters.make_initial_conditions(np.zeros(4))
        mp = parameters.make_model_params(fiber=fiber, waves=waves, ic=ic)

        self.assertIsInstance(mp, parameters.ModelParams)
        self.assertIs(mp.fiber, fiber)
        self.assertIs(mp.waves, waves)
        self.assertIs(mp.ic, ic)


class TestIntegrators(unittest.TestCase):
    def test_rk4_step_matches_exp_for_simple_ode(self) -> None:
        # y' = y, y(0)=1 => y(z)=exp(z)
        def f(z: float, y: np.ndarray, _params: object) -> np.ndarray:
            return y

        y0 = np.array([1.0], dtype=float)
        z0 = 0.0
        dz = 0.1

        y1 = integrators.rk4_step(f, z0, y0, dz, params=None)
        expected = np.array([math.exp(dz)], dtype=float)

        np.testing.assert_allclose(y1, expected, rtol=1e-7, atol=0)

    def test_integrate_interval_shapes_and_saving(self) -> None:
        def f(z: float, y: np.ndarray, _params: object) -> np.ndarray:
            return y  # y' = y

        z_max = 1.0
        dz = 0.1
        y0 = np.array([1.0], dtype=float)

        z_out, y_out = integrators.integrate_interval(
            f=f, z_max=z_max, dz=dz, y0=y0, params=None, save_every=2, check_nan=True
        )

        # n_steps = 10, save_every=2 => saved at steps 0,2,4,6,8,10 => 6 points
        self.assertEqual(z_out.shape, (6,))
        self.assertEqual(y_out.shape, (6, 1))
        np.testing.assert_allclose(z_out, np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), rtol=0.0, atol=1e-15)

        expected = np.exp(z_out)
        np.testing.assert_allclose(y_out[:, 0], expected, rtol=0.0, atol=3e-6)

    def test_integrate_fixed_step_rejects_bad_inputs(self) -> None:
        def f(z: float, y: np.ndarray, _params: object) -> np.ndarray:
            return y

        with self.assertRaises(ValueError):
            integrators.integrate_fixed_step(
                f=f,
                z_grid=np.array([[0.0, 0.1]]),
                y0=np.array([1.0]),
                params=None,
            )

        with self.assertRaises(ValueError):
            integrators.integrate_fixed_step(
                f=f,
                z_grid=np.array([0.0, 0.1]),
                y0=np.array([1.0]),
                params=None,
                save_every=0,
            )

    def test_check_nan_raises(self) -> None:
        def f(_z: float, _y: np.ndarray, _params: object) -> np.ndarray:
            return np.array([np.nan], dtype=float)

        with self.assertRaises(FloatingPointError):
            integrators.integrate_interval(
                f=f,
                z_max=0.2,
                dz=0.1,
                y0=np.array([0.0], dtype=float),
                params=None,
                save_every=1,
                check_nan=True,
            )

        # If check_nan=False, it should not raise (but output will contain NaNs).
        z_out, y_out = integrators.integrate_interval(
            f=f,
            z_max=0.2,
            dz=0.1,
            y0=np.array([0.0], dtype=float),
            params=None,
            save_every=1,
            check_nan=False,
        )
        self.assertTrue(np.isnan(y_out).any())


class TestYamanModel(unittest.TestCase):
    def _make_minimal_params(self) -> parameters.ModelParams:
        fiber = parameters.make_fiber_params(
            gamma=1.3,
            alpha=0.0,
            beta=np.array([10.0, 11.0, 12.0, 13.0], dtype=float),
        )
        waves = parameters.make_wave_params(
            omega=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
            P_in=np.array([0.5, 0.5, 0.0, 0.0], dtype=float),
        )
        ic = parameters.make_initial_conditions(waves.P_in)
        return parameters.make_model_params(fiber=fiber, waves=waves, ic=ic)

    def test_rhs_shape_and_dtype(self) -> None:
        mp = self._make_minimal_params()
        a = mp.ic.A0.copy()
        da = yaman_model.rhs_yaman_simplified(z=0.0, a_arr=a, params=mp)

        self.assertEqual(da.shape, (4,))
        self.assertTrue(np.iscomplexobj(da))

    def test_rhs_rejects_wrong_shape(self) -> None:
        mp = self._make_minimal_params()
        with self.assertRaises(ValueError):
            yaman_model.rhs_yaman_simplified(z=0.0, a_arr=np.zeros(3, dtype=np.complex128), params=mp)

    def test_kerr_stub_matches_vector_formula(self) -> None:
        gamma = 2.0
        rng = np.random.default_rng(0)
        a = (rng.standard_normal(4) + 1j * rng.standard_normal(4)).astype(np.complex128)

        # What the "vector form" claims:
        powers = np.abs(a) ** 2
        total = powers.sum()
        other = total - powers
        factor = (2.0 / 3.0) * powers + (4.0 / 3.0) * other
        expected = 1j * gamma * factor * a

        got = yaman_model._kerr_terms_stub(z=0.0, a_arr=a, gamma=gamma)
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-14)

    def test_fwm_terms_vanish_when_signal_and_idler_zero(self) -> None:
        gamma = 1.0
        betas = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
        # pump1, pump2 non-zero; signal, idler exactly 0
        a = np.array([1.0 + 0.0j, 2.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

        got = yaman_model._fwm_terms_stub(z=0.123, a_arr=a, _alpha=0.0, betas=betas, gamma=gamma)
        np.testing.assert_allclose(got, np.zeros(4, dtype=np.complex128), rtol=0.0, atol=0.0)


class TestSimulationPipeline(unittest.TestCase):
    def test_run_single_simulation_runs_and_shapes(self) -> None:
        cfg = config.SimulationConfig(
            z_max=0.5,
            dz=0.1,
            integrator="rk4",
            save_every=1,
            check_nan=True,
            verbose=False,
        )

        gamma = 1.3
        alpha = 0.0
        beta = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)
        omega = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

        # Expected wave order across the project:
        # [pump1, pump2, signal, idler]
        p_in = np.array([0.5, 0.5, 0.0, 0.0], dtype=float)

        z, A = simulation.run_single_simulation(
            cfg,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            omega=omega,
            p_in=p_in,
            phase_in=None,
        )

        self.assertEqual(z[0], 0.0)
        self.assertAlmostEqual(z[-1], cfg.z_max, places=15)
        self.assertEqual(A.shape[1], 4)
        self.assertEqual(z.shape[0], A.shape[0])
        self.assertTrue(np.all(np.isfinite(A.real)))
        self.assertTrue(np.all(np.isfinite(A.imag)))

    def test_example_zero_signal_wave_order_is_consistent(self) -> None:
        # This test documents the intended wave ordering.
        # If it fails, the example likely uses a different index convention than yaman_model.
        z, A = simulation.example_zero_signal()
        self.assertEqual(A.shape[1], 4)

        # Intended order: [pump1, pump2, signal, idler]
        # In "zero signal" example, signal and idler should be exactly zero at z=0.
        # If the example uses a different ordering, these indices will not be zero.
        np.testing.assert_allclose(A[0, 2], 0.0 + 0.0j, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(A[0, 3], 0.0 + 0.0j, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
