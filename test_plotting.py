import unittest
import matplotlib.pyplot as plt
from nano_star_simulation import plot_fractal_tree, Proton, proton_lineage

class TestPlotting(unittest.TestCase):

    def tearDown(self):
        proton_lineage.clear()

    def test_plot_fractal_tree_index_error(self):
        """
        Tests that the plot_fractal_tree function does not raise an IndexError
        when a proton with an empty lineage is present.
        """
        fig, ax = plt.subplots()
        initial_proton = Proton("P0", [])
        proton_lineage.append(initial_proton)
        
        # This will fail the test automatically if IndexError is raised
        plot_fractal_tree(ax, initial_proton, 0, 0, 90, 1, 1)

if __name__ == '__main__':
    unittest.main()
