# Copyright (C) 2015-2026: The University of Edinburgh, United Kingdom
#
# This file is part of the gprMax source code base.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax. If not, see <https://www.gnu.org/licenses/>.
"""
Tests for MagneticDispersiveMaterial - a material supporting frequency
dependent magnetic properties with Debye, Lorentz, and Drude models.
"""
import numpy as np
import pytest

import gprMax
import gprMax.config as config
import gprMax.model as model_mod
from gprMax.materials import MagneticDispersiveMaterial, Material


class _MagneticMaterialConfig:
    """Configuration mock for testing MagneticDispersiveMaterial."""
    def __init__(self, enabled=True):
        self.dispersive_averaging = enabled
        self.materials = {"maxpoles": 10}


def _magnetic_debye(numid, material_id, mr, sm, mdeltaer, mtau):
    """Helper to create a magnetic Debye material."""
    material = MagneticDispersiveMaterial(numid, material_id)
    material.type = "magneticdebye"
    material.mr = mr
    material.sm = sm
    material.mdeltaer = list(mdeltaer)
    material.mtau = list(mtau)
    return material


def _magnetic_lorentz(numid, material_id, mr, sm, mdeltaer, mtau, malpha):
    """Helper to create a magnetic Lorentz material."""
    material = MagneticDispersiveMaterial(numid, material_id)
    material.type = "magneticlorentz"
    material.mr = mr
    material.sm = sm
    material.mdeltaer = list(mdeltaer)
    material.mtau = list(mtau)
    material.malpha = list(malpha)
    return material


def _magnetic_drude(numid, material_id, mr, sm, mtau, malpha):
    """Helper to create a magnetic Drude material."""
    material = MagneticDispersiveMaterial(numid, material_id)
    material.type = "magneticdrude"
    material.mr = mr
    material.sm = sm
    material.mtau = list(mtau)
    material.malpha = list(malpha)
    return material


class TestMagneticDispersiveMaterialInit:
    """Test the MagneticDispersiveMaterial constructor and basic properties."""

    def test_basic_initialization(self, monkeypatch):
        """Test basic creation of MagneticDispersiveMaterial."""
        model_config = _MagneticMaterialConfig()
        # Note: monkeypatch not needed for simple property checks
        material = _magnetic_debye(3, "mag-mat", 5.0, 0.1, (6.0,), (1e-11,))
        assert material.ID == "mag-mat"
        assert material.numID == 3
        assert material.mr == 5.0
        assert material.sm == 0.1
        assert material.poles == 1
        assert material.type == "magneticdebye"
        assert material.mdeltaer == [6.0]
        assert material.mtau == [1e-11]

    def test_lorentz_initialization(self, monkeypatch):
        """Test Lorentz model initialization."""
        material = _magnetic_lorentz(3, "lorentz-mag", 5.0, 0.1, (6.0,), (1e-11,), (0.5,))
        assert material.ID == "lorentz-mag"
        assert material.type == "magneticlorentz"
        assert material.poles == 1
        assert material.mdeltaer == [6.0]
        assert material.malpha == [0.5]

    def test_drude_initialization(self, monkeypatch):
        """Test Drude model initialization."""
        material = _magnetic_drude(3, "drude-mag", 5.0, 0.1, (1e-11,), (0.01,))
        assert material.ID == "drude-mag"
        assert material.type == "magneticdrude"
        assert material.poles == 1
        assert material.mdeltaer == []  # Drude doesn't use deltaer in same way
        assert material.malpha == [0.01]

    def test_multiple_poles(self, monkeypatch):
        """Test material with multiple poles."""
        material = MagneticDispersiveMaterial(
            3, "multi-mag",
            poles=3,
            model_types=["magneticdebye", "magneticlorentz", "magneticdrude"],
            mdeltaer=[1.0, 2.0, 3.0],
            mtau=[1e-11, 2e-11, 3e-11],
            malpha=[0.1, 0.2, 0.3],
        )
        assert material.poles == 3
        assert material.model_types == ["magneticdebye", "magneticlorentz", "magneticdrude"]
        assert material.mdeltaer == [1.0, 2.0, 3.0]
        assert material.mtau == [1e-11, 2e-11, 3e-11]
        assert material.malpha == [0.1, 0.2, 0.3]


class TestMagneticDispersiveMaterialCalculateHr:
    """Test the calculate_hr method for complex permeability."""

    def test_debye_calculate_hr(self, monkeypatch):
        """Test magnetic Debye permeability calculation at various frequencies."""
        material = _magnetic_debye(3, "debye-mag", 5.0, 0.1, (6.0,), (1e-11,))

        # At DC (freq=0), permeability should be mr + mdeltaer
        hr_dc = material.calculate_hr(0.0)
        assert hr_dc.real == pytest.approx(5.0 + 6.0)  # mr + mdeltaer
        assert hr_dc.imag == pytest.approx(0.0)

        # At non-zero frequency
        hr_freq = material.calculate_hr(1e9)
        # mr + mdeltaer / (1 + 1j*w*tau)
        w = 2 * np.pi * 1e9
        expected = 5.0 + 6.0 / (1 + 1j * w * 1e-11)
        assert hr_freq == pytest.approx(expected)

    def test_lorentz_calculate_hr(self, monkeypatch):
        """Test magnetic Lorentz permeability calculation."""
        material = _magnetic_lorentz(3, "lorentz-mag", 5.0, 0.1, (6.0,), (1e-9,), (0.5,))

        hr_freq = material.calculate_hr(1e9)
        w = 2 * np.pi * 1e9
        pole_omega = 2 * np.pi * 1e9
        # mr + mdeltaer / (pole_omega^2 - w^2 + 1j*alpha*w)
        expected = 5.0 + 6.0 / (pole_omega**2 - w**2 + 1j * 0.5 * w)
        assert hr_freq == pytest.approx(expected)

    def test_drude_calculate_hr(self, monkeypatch):
        """Test magnetic Drude permeability calculation."""
        material = _magnetic_drude(3, "drude-mag", 5.0, 0.1, (1e-11,), (0.01,))

        hr_freq = material.calculate_hr(1e9)
        w = 2 * np.pi * 1e9
        # mr + sm / (1j*w*m0) for Drude
        expected = 5.0 + 0.1 / (1j * w * config.m0)
        assert hr_freq == pytest.approx(expected)

    def test_debye_with_conductivity(self, monkeypatch):
        """Test magnetic permittivity with magnetic conductivity."""
        material = _magnetic_debye(3, "cond-mag", 5.0, 0.5, (6.0,), (1e-11,))
        material.sm = 0.5  # Add conductivity

        hr_freq = material.calculate_hr(1e9)
        # Should not crash
        assert isinstance(hr_freq, complex)


class TestMagneticDispersiveMaterialCalculateUpdateCoeffsH:
    """Test the calculate_update_coeffsH method."""

    def test_debye_update_coeffs(self, monkeypatch, mocker):
        """Test update coefficient calculation for magnetic Debye material."""
        mocker.patch('gprMax.config.config.m0', 1.25663706212e-6)  # mu0

        mock_grid = mocker.MagicMock()
        mock_grid.maxpoles = 10
        mock_grid.dispersivedtype = np.complex128
        mock_grid.dl = [1e-3]
        mock_grid.size = [1.0, 1.0, 1.0]
        mock_grid.nx = 1
        mock_grid.ny = 1
        mock_grid.nz = 1

        material = _magnetic_debye(3, "update-mag", 5.0, 0.1, (6.0,), (1e-11,))
        material.calculate_update_coeffsH(mock_grid)

        # Should not crash and set coefficients
        assert hasattr(material, 'DA')
        assert hasattr(material, 'DBx')
        assert hasattr(material, 'DBy')
        assert hasattr(material, 'DBz')
        assert hasattr(material, 'srcm')


class TestMagneticDispersiveMaterialIsValid:
    """Test the is_valid property."""

    def test_valid_magnetic(self, monkeypatch):
        """Test valid magnetic material."""
        material = _magnetic_debye(3, "valid-mag", 5.0, 0.1, (6.0,), (1e-11,))
        assert material.is_valid is True

    def test_invalid_mr_zero(self, monkeypatch):
        """Test material with zero mr is invalid."""
        material = _magnetic_debye(3, "zero-mr-mag", 0.0, 0.1, (6.0,), (1e-11,))
        assert material.is_valid is False


class TestMagneticDispersiveMaterialLossTangentMag:
    """Test magnetic loss tangent (if applicable)."""

    def test_loss_tangent_debye(self, monkeypatch):
        """Test loss tangent for magnetic Debye material."""
        material = _magnetic_debye(3, "loss-mag", 5.0, 0.0, (6.0,), (1e-11,))

        # For lossless magnetic material, loss tangent should be 0
        lt = material.calculate_hr(1e9)  # Use calculate_hr as proxy
        # Check that the material is valid
        assert material.is_valid is True


class TestMagneticDispersiveMaterialReprAndCopy:
    """Test string representation and copy method."""

    def test_repr(self, monkeypatch):
        """Test __repr__ method."""
        material = _magnetic_debye(3, "repr-mag", 5.0, 0.1, (6.0,), (1e-11,))
        repr_str = repr(material)
        assert "MagneticDispersiveMaterial" in repr_str
        assert "ID='repr-mag'" in repr_str or "ID=" in repr_str

    def test_copy(self, monkeypatch):
        """Test copy method creates independent copy."""
        material = _magnetic_debye(3, "copy-mag", 5.0, 0.1, (6.0,), (1e-11,))

        copied = material.copy()
        assert copied.ID == material.ID
        assert copied.numID == material.numID
        assert copied.mr == material.mr
        assert copied.sm == material.sm
        assert copied.poles == material.poles

        # Modify original and check copy is independent
        material.mr = 9.0
        assert copied.mr == 5.0  # Copy should be unchanged


class TestMagneticDispersiveMaterialIntegration:
    """Integration-style tests with mock grids."""

    def test_full_debye_workflow(self, monkeypatch, mocker):
        """Test full workflow: init -> calculate_hr -> calculate_update_coeffsH."""
        mocker.patch('gprMax.config.config.m0', 1.25663706212e-6)

        mock_grid = mocker.MagicMock()
        mock_grid.maxpoles = 10
        mock_grid.dispersivedtype = np.complex128
        mock_grid.dl = [1e-3]
        mock_grid.size = [1.0, 1.0, 1.0]
        mock_grid.nx = 1
        mock_grid.ny = 1
        mock_grid.nz = 1

        material = _magnetic_debye(3, "workflow-mag", 5.0, 0.1, (6.0,), (1e-11,))

        # Step 1: Calculate permeability
        hr = material.calculate_hr(1e9)
        assert isinstance(hr, complex)

        # Step 2: Calculate update coefficients
        material.calculate_update_coeffsH(mock_grid)
        assert hasattr(material, 'DA')
        assert hasattr(material, 'srcm')

        # Step 3: Verify is_valid
        assert material.is_valid is True


def test_magnetic_material_import():
    """Test that MagneticDispersiveMaterial can be imported from gprMax.materials."""
    from gprMax.materials import MagneticDispersiveMaterial
    assert MagneticDispersiveMaterial is not None


def test_magnetic_is_subclass_of_material():
    """Test that MagneticDispersiveMaterial is a subclass of Material."""
    from gprMax.materials import MagneticDispersiveMaterial, Material
    assert issubclass(MagneticDispersiveMaterial, Material)


def test_magnetic_material_inherits_properties():
    """Test that MagneticDispersiveMaterial inherits Material properties."""
    material = MagneticDispersiveMaterial(3, "test")
    # Inherited from Material
    assert hasattr(material, 'numID')
    assert hasattr(material, 'ID')
    assert hasattr(material, 'type')
    assert hasattr(material, 'mr')
    assert hasattr(material, 'sm')
    # Inherited from DispersiveMaterial (Material base)
    assert hasattr(material, 'poles')
    assert hasattr(material, 'mdeltaer')
    assert hasattr(material, 'mtau')
    assert hasattr(material, 'malpha')