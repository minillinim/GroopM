#!/usr/bin/env python3

import numpy as np
np.seterr(all='raise')

class DensityKernel(object):

    def __init__(self,
        resolution,
        kernel_type='linear',
        span_ratio=.075):
        self.kernel_stamp = None
        self.resolution = resolution
        self.step = 1. / self.resolution
        self.kernel_type = kernel_type

        if self.kernel_type == 'linear':

            span = int(self.resolution * span_ratio)
            if span % 2 == 0:
                span -= 1
            assert(span > 2)
            half_span = int((span - 1) / 2)
            step = 1. / (half_span + 1)

            self.kernel_stamp = np.pad(
                np.pad(
                    [1.],
                    (half_span, half_span),
                    'linear_ramp',
                    end_values=(step, step)),
                (self.resolution-half_span-1, self.resolution-half_span-1),
                'constant',
                constant_values=(0.))

    def to_kidx(self, value):
        return max(0, self.resolution - int((1. - value) / self.step) -1)

    def get(self, value):
        kidx = self.to_kidx(value)
        return self.kernel_stamp[kidx:kidx+self.resolution]

class DensityStore(object):

    def __init__(self, num_columns, kernel, plot_journey=False, build_store=True):
        self.num_columns = num_columns
        self.kernel = kernel
        self.resolution = kernel.resolution
        if build_store:
            self.num_rows = 0
            self.store = np.zeros((self.resolution, self.num_columns))

        self.plot_journey=plot_journey
        if self.plot_journey:
            if build_store:
                self.density_graph = DensityGraph(
                    self.num_columns,
                    self.kernel)
        else:
            self.density_graph = None

    def duplicate(self):
        density_store =  DensityStore(
            self.num_columns,
            self.kernel,
            plot_journey=self.plot_journey,
            build_store=False)
        density_store.num_rows = self.num_rows
        density_store.store = np.array(self.store)

        if self.plot_journey:
            density_store.density_graph = self.density_graph.duplicate()

    def add(self, row):
        for idx, value in enumerate(row):
            self.store[:, idx] += self.kernel.get(value)
        self.num_rows += 1

        if self.density_graph is not None:
            self.density_graph.add(row)

    def remove(self, row):
        for idx, value in enumerate(row):
            self.store[:, idx] -= self.kernel.get(value)
        self.num_rows -= 1

        if self.density_graph is not None:
            self.density_graph.remove(row)

    def find_brightest(self, masked_columns):
        hotspots = np.argmax(self.store, axis=0)
        brightness = self.store[
            hotspots,
            range(self.num_columns)]
        brightness[masked_columns] = 0
        brightest_column = np.argmax(brightness)
        brightest_value = brightness[brightest_column]
        target_value = 1 - (hotspots[brightest_column] / self.resolution)

        return (
            brightest_column,
            brightest_value,
            target_value)

    def plot(self, ax):
        if self.density_graph is not None:
            return self.density_graph.plot_max_line(
                self.density_graph.plot_dens_line(
                    self.density_graph.plot(ax),
                    np.argmax(self.store, axis=0)))
        return ax

class DensityGraph(object):

    def __init__(self, num_columns, kernel, step_between=200, build_store=True):
        self.num_columns = num_columns
        self.kernel = kernel
        self.resolution = kernel.resolution
        self.step_between = step_between
        if build_store:
            self.store = np.zeros(
                (self.resolution, self.step_between * (self.num_columns - 1)))
            self.back_mult = np.array(list(range(self.step_between))[::-1])/(self.step_between - 1.)
            self.forward_mult = np.array(list(range(self.step_between)))/(self.step_between - 1.)

    def duplicate(self):
        density_graph =  DensityGraph(
            self.num_columns,
            self.kernel,
            step_between=self.step_between,
            build_store=False)
        density_graph.store = np.array(self.store)
        density_graph.back_mult = self.back_mult
        density_graph.forward_mult = self.forward_mult

        return density_graph

    def add(self, row):
        last_val = 0
        for idx, value in enumerate(row):
            if idx > 0:
                start_step_idx = self.step_between * (idx-1)
                for mult_idx, dg_idx in enumerate(
                    range(
                        start_step_idx,
                        start_step_idx + self.step_between)):

                    self.store[:,dg_idx] += self.kernel.get(
                        value * self.forward_mult[mult_idx] +
                        last_val * self.back_mult[mult_idx])

            last_val = value

    def remove(self, row):
        last_val = 0
        for idx, value in enumerate(row):
            if idx > 0:
                start_step_idx = self.step_between * (idx-1)
                for mult_idx, dg_idx in enumerate(
                    range(
                        start_step_idx,
                        start_step_idx + self.step_between)):

                    self.store[:,dg_idx] -= self.kernel.get(
                        value * self.forward_mult[mult_idx] +
                        last_val * self.back_mult[mult_idx])

            last_val = value

    def plot(self, ax):
        ax.imshow(
            np.sqrt(np.abs(self.store)),
            interpolation='nearest',
            extent=(-0.5, ((self.num_columns-1) * 1000) -0.5, -0.5, 3000-0.5))

        return ax

    def plot_max_line(self, ax):
        ax.plot(
            [(self.num_columns - 1) * i for i in range(np.shape(self.store)[1])],
            [3 * (1000 - i) for i in np.argmax(self.store, axis=0)],
            'r')

        return ax

    def plot_dens_line(self, ax, max_line):
        ax.plot(
            [i * 1000 for i in range(self.num_columns)],
            [3 * (1000 - i) for i in max_line],
            'k')

        return ax
