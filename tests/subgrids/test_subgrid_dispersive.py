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
Tests for subgrid dispersive material averaging.

Extends the existing subgrid functionality to support Debye, Lorentz, and
Drude dispersive materials in subgrids. Enables opt-in dispersive interface
averaging for subgrid-boundary materials, similar to the magnetic-dispersive
averaging for H-field components.

When enabled, the compound material at subgrid boundaries retains every
distinct inclusive pole and scales each residue by its cell weight.
"""