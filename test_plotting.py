import unittest
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for tests
import matplotlib.pyplot as plt
from nano_star_simulation import plot_fractal_tree, Proton, proton_lineage

class TestPlotting(unittest.TestCase):

    def test_plot_fractal_tree_index_error(self):
        """
        Tests that the plot_fractal_tree function does not raise an IndexError
        when a proton with an empty lineage is present.
        """
        fig, ax = plt.subplots()
        initial_proton = Proton("P0", [])
        proton_lineage.append(initial_proton)

        try:
            plot_fractal_tree(ax, initial_proton, 0, 0, 90, 1, 1)
        except IndexError:
            self.fail("plot_fractal_tree() raised IndexError unexpectedly!")
        finally:
            # cleanup so other tests aren't affected
            try:
                proton_lineage.remove(initial_proton)
            except ValueError:
                pass

if __name__ == '__main__':
    unittest.main()