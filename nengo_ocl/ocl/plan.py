import pyopencl as cl

class Plan(object):

    def __init__(self, queue, kern, gsize, lsize, *kwargs):
        self.queue = queue
        self.kern = kern
        self.gsize = gsize
        self.lsize = lsize
        self.kwargs = kwargs

    def __call__(self):
        cl.enqueue_nd_range_kernel(
            self.queue, self.kern, self.gsize, self.lsize)
        self.queue.finish()


class Prog(object):
    def __init__(self, plans):
        self.plans = plans
        self.queues = [p.queue for p in self.plans]
        self.kerns = [p.kern for p in self.plans]
        self.gsize = [p.gsize for p in self.plans]
        self.lsize = [p.lsize for p in self.plans]

    def __call__(self):
        return self.call_n_times(1)

    def call_n_times(self, n):
        self.enqueue_n_times(n)
        self.queues[-1].finish()

    def enqueue_n_times(self, n):
        clrk = cl.enqueue_nd_range_kernel
        qs, ks, gs, ls = self.queues, self.kerns, self.gsize, self.lsize
        for ii in range(n):
            map(clrk, qs, ks, gs, ls)