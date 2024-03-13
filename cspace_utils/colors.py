import random
import colorsys
from fractions import Fraction
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def generate_maximally_different_colors(n):
    """
    Generate n maximally different random colors for matplotlib.

    Parameters:
        n (int): Number of colors to generate.

    Returns:
        List of RGB tuples representing the random colors.
    """
    if n <= 0:
        raise ValueError("Number of colors (n) must be greater than zero.")

    # Define a list to store the generated colors
    colors = []

    # Generate n random hues, ensuring maximally different colors
    hues = [i / n for i in range(n)]

    # Shuffle the hues to get random order of colors
    random.shuffle(hues)
   
    # Convert each hue to RGB
    for hue in hues:
        # We keep saturation and value fixed at 0.9 and 0.8 respectively
        saturation = 0.9
        value = 0.8
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)

    return colors

def infinite_hues():
    yield Fraction(0)
    for k in itertools.count():
        i = 2**k # zenos_dichotomy
        for j in range(1,i,2):
            yield Fraction(j,i)


def hue_to_hsvs(h: Fraction):
    # tweak values to adjust scheme
    for s in [Fraction(6,10)]:
        for v in [Fraction(6,10), Fraction(9,10)]:
            yield (h, s, v)


def rgb_to_css(rgb) -> str:
    uint8tuple = map(lambda y: int(y*255), rgb)
    return tuple(uint8tuple)


def css_to_html(css):
    return f"<text style=background-color:{css}>&nbsp;&nbsp;&nbsp;&nbsp;</text>"


def n_colors(n=33, rgbs_ret = False):
    hues = infinite_hues()
    hsvs = itertools.chain.from_iterable(hue_to_hsvs(hue) for hue in hues)
    rgbs = (colorsys.hsv_to_rgb(*hsv) for hsv in hsvs)
    csss = (rgb_to_css(rgb) for rgb in rgbs)
    to_ret = list(itertools.islice(csss, n)) if rgbs_ret else list(itertools.islice(csss, n))
    return to_ret

def generate_distinct_colors(n, rgb = False):
    cmap = plt.cm.get_cmap('hsv', n)  # Choose a colormap
    colors = [mcolors.rgb2hex(cmap(i)[:3]) for i in range(n)]  # Convert colormap to hexadecimal colors
    if rgb:
        return [hex_to_rgb(c) for c in colors]
    else:
        return colors