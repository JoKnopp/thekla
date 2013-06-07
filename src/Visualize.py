from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

import logging
import os
import errno


def mkdir_p(path):
    path = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

#the code for radar_factory is not from me, I found it here:
#http://matplotlib.org/examples/api/radar_chart.html
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = 2*np.pi * np.linspace(0, 1-1./num_vars, num_vars)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(theta * 180/np.pi, labels,y=-0.08)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


def create_radar_chart(title, clusters, column_names, fname='', plot=True):
    """
    Plots a radar chart.
    
    title: string
    clusters: dictionary {cluster_name:scipy.ndarray}
        centroids of clusters
    column_names: list of strings
        labels for the axes
    """
    #this is a fix for bigger fonts at the topN topic words
    plt.rcParams.update({'font.size': 20})

    N = clusters.values()[0].shape[1] #number of axes for the radar chart
    theta = radar_factory(N, frame='polygon')

    fig = plt.figure(figsize=(11, 11))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.9, bottom=0.05)

    colors = ['b', 'r', 'g', 'm', 'y']
    cols = cycle(colors)

    ax = fig.add_subplot(111, projection='radar')
    ax.set_title(title, weight='bold', size='26', position=(0.08, 1.11),
                 horizontalalignment='center', verticalalignment='center')

    labels = []
    
    for cluster_name, topic_vector in clusters.items():
        #flatten to make sure there is a vector, not a matrix with 1 entry
        topic_vector = topic_vector.flatten()
        color = cols.next()
        ax.plot(theta, topic_vector, color=color)
        ax.fill(theta, topic_vector, facecolor=color, alpha=0.10)
        labels.append(cluster_name)
    ax.set_varlabels(column_names)

    legend = plt.legend(labels, loc=(0.78, 1.00), labelspacing=0.07)
    plt.setp(legend.get_texts(), fontsize='small')
    if fname:
        print('saving "{name}"'.format(name=fname))
        mkdir_p(fname)
        plt.savefig(fname)
    if plot:
        plt.show()

def plot_radar_chart(title, clusters, column_names):
    create_radar_chart(title,clusters, column_names)

def save_radar_chart(title, clusters, column_names, fname, plot=False):
    create_radar_chart(title, clusters, column_names, fname=fname, plot=plot)
