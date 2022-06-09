
from mala.common.parallelizer import printout
from mala.descriptors.descriptor import Descriptor
from mala.targets.target import Target


class TargetDescriptor(Descriptor):
    """
    Descriptor class that encapsulates a Target object.

    This is useful for advanced mappings, e.g. temperature improvement
    networks or two step schemes.
    """

    def __init__(self, params):
        super(TargetDescriptor, self).__init__(params)
        self.target_calculator = Target(params)

    def convert_units(self, array, in_units="1/eV"):
        """
        Convert descriptors from a specified unit into the ones used in MALA.

        Parameters
        ----------
        array : numpy.array
            Data for which the units should be converted.

        in_units : string
            Units of array.

        Returns
        -------
        converted_array : numpy.array
            Data in MALA units.

        """
        return self.target_calculator.convert_units(array, in_units=in_units)

    def backconvert_units(self, array, out_units):
        """
        Convert descriptors from MALA units into a specified unit.

        Parameters
        ----------
        array : numpy.array
            Data in MALA units.

        out_units : string
            Desired units of output array.

        Returns
        -------
        converted_array : numpy.array
            Data in out_units.

        """
        return self.target_calculator.backconvert_units(array,
                                                        out_units=out_units)

    def calculate_from_qe_out(self, qe_out_file, qe_out_directory):
        """
        Calculate the descriptors based on a Quantum Espresso outfile.

        Parameters
        ----------
        qe_out_file : string
            Name of Quantum Espresso output file for snapshot.

        qe_out_directory : string
            Path to Quantum Espresso output file for snapshot.

        Returns
        -------
        descriptors : numpy.array
            Numpy array containing the descriptors with the dimension
            (x,y,z,descriptor_dimension)

        """
        raise Exception("Calculation from QE.out not supported for this"
                        " Descriptor type.")

    def calculate_from_atoms(self, atoms, grid_dimensions):
        """
        Calculate the descriptors based on the atomic configurations.

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object holding the atomic configuration.

        grid_dimensions : list
            Grid dimensions to be used, in the format [x,y,z].

        Returns
        -------
        descriptors : numpy.array
            Numpy array containing the descriptors with the dimension
            (x,y,z,descriptor_dimension)
        """
        raise Exception("Calculation from atoms not supported for this"
                        " Descriptor type.")

