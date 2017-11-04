#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
#from PIL import Image
import pyopencl as cl
from collections import OrderedDict

"""
Reading materials:
1. https://documen.tician.de/pyopencl/runtime_memory.html#buffer
2. http://www.drdobbs.com/open-source/easy-opencl-with-python/240162614
"""

"""
source_image - numpy array.astype(np.float32)
destination_segments - numpy array.astype(np.float32)
abst - level of abstraction - must satisfy NxN modulo abstxabst == 0
num_transforms
"""
def segment_and_transform(source_image, destination_segments, abst=92):

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    # ------------------------------------------------------------------------------

    # Create the transformations - the order of these transforms are:
    # 0 identity, 1 rot 90, 2 rot 180, 3 rot 270, 4 ref y=x, 5 ref y=-x, 6 ref x=0, 7 ref y=0
    A = transform_image(ctx, queue, abst, source_image)
    B = transform_image(ctx, queue, abst, destination_segments)
    return A, B

def transform_image(ctx, queue, abst, image):
    # We want to transform an array of images over all affine transformations
    #  and abstraction levels. We can do almost any abstraction level because
    #  we can use pillow to adjust the image width and height. (Note that
    # this might not work on real-life input data). Lets first do transforms
    # for one image.
    # --------------------------------------------------------------------------
    mf = cl.mem_flags
    image_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
    output_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, 8 * image.shape[0] * image.shape[1] * np.dtype('float32').itemsize)

    # Load program kernel.
    kernel = open("./opencl_kernels/parallel-transforms.c", "r")

    prg = cl.Program(ctx, kernel.read()).build(options=[
        '-D', 'width=%s' % image.shape[0],
        '-D', 'abst=%s' % abst,
        '-D', 'half_width=%f' % ((abst - 1) / 2.0)])

    # TODO Either allow OpenCL to do something clever or calculate local size yourself.
    trans = prg.parallel_transforms(queue, (184, 184), None, image_buffer, output_buffer)

    # --------------------------------------------------------------------------
    trans.wait()
    elapsed = 1e-9 * (trans.profile.end - trans.profile.start)
    print("Execution time: %g s" % elapsed)

    # There are 8 transformations.
    transformed = np.empty((8, image.shape[0], image.shape[1])).astype(np.float32)
    cl.enqueue_copy(queue, transformed, output_buffer)
    return transformed

def print_info(cl, ctx):
    print("MAX_WORK_GROUP_SIZE: %s" % ctx.devices[0].get_info(
        cl.device_info.MAX_WORK_GROUP_SIZE))
    print("MAX_WORK_ITEM_DIMENSIONS: %s " % ctx.devices[0].get_info(
        cl.device_info.MAX_WORK_ITEM_DIMENSIONS))
    print("MAX_WORK_ITEM_SIZES: %s" % ctx.devices[0].get_info(
        cl.device_info.MAX_WORK_ITEM_SIZES))
    print("LOCAL_MEM_SIZE: %s" % ctx.devices[0].get_info(
        cl.device_info.LOCAL_MEM_SIZE))
    print("GLOBAL_MEM_SIZE: %s" % ctx.devices[0].get_info(
        cl.device_info.GLOBAL_MEM_SIZE))
    print("Float size: %s" % np.dtype('float32').itemsize)

"""
Orthonormal Transformation Matrices:
"""
def get_ortho():

    # Identity - for completeness.
    identity = np.asarray([[1, 0], [0, 1]]).astype(np.int32)

    # Rotations:
    rot90 = np.asarray([[math.cos(math.radians(90)),  math.sin(math.radians(90))],
                        [-math.sin(math.radians(90)), math.cos(math.radians(90))]]).astype(np.int32)
    rot180 = np.asarray([[math.cos(math.radians(180)), math.sin(math.radians(180))],
                         [-math.sin(math.radians(180)), math.cos(math.radians(180))]]).astype(np.int32)
    rot270 = np.asarray([[math.cos(math.radians(270)), math.sin(math.radians(270))],
                         [-math.sin(math.radians(270)), math.cos(math.radians(270))]]).astype(np.int32)

    # Reflection about y = x.
    ref_yx = np.asarray([[0, 1], [1, 0]]).astype(np.int32)

    # Reflection about y = -x.
    ref_yMx = np.asarray([[0, -1], [-1, 0]]).astype(np.int32)

    # Reflection about y axis, x=0.
    ref_x0 = np.asarray([[-1, 0], [0, 1]]).astype(np.int32)

    # Reflection about x axis, y = 0.
    ref_y0 = np.asarray([[1, 0], [0, -1]]).astype(np.int32)

    return np.asarray([identity, rot90, rot180, rot270, ref_yx, ref_yMx, ref_x0,
                ref_y0]).astype(np.int32)

    # Ordered dict should preserve the order of the values.
    """return OrderedDict({'identity': identity,
           'rot90': rot90,
           'rot180': rot180,
           'rot270': rot270,
           'ref_yx': ref_yx,
           'ref_yMx': ref_yMx,
           'ref_x0': ref_x0,
           'ref_y0': ref_y0})
    """