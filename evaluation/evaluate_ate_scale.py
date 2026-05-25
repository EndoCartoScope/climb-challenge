# Modified by Raul Mur-Artal
# Automatically compute the optimal scale factor for monocular VO/SLAM.

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""
import numpy as np


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """
    model = np.asarray(model, dtype=float)
    data = np.asarray(data, dtype=float)

    assert model.shape == data.shape, "model and data must have the same shape as data"
    assert model.shape[0] == 3, "model and data must be 3xN"

    model_mean = model.mean(axis=1, keepdims=True)  # (3,1)
    data_mean = data.mean(axis=1, keepdims=True)  # (3,1)

    model_zerocentered = model - model_mean  # (3,N)
    data_zerocentered = data - data_mean  # (3,N)

    W = np.zeros((3, 3))
    for i in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, i], data_zerocentered[:, i])

    # SVD en numpy "normal"
    U, d, Vt = np.linalg.svd(W.T)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    rot = U @ S @ Vt  # (3,3)

    # Scale
    rotmodel = rot @ model_zerocentered
    dots = np.sum(np.sum(data_zerocentered * rotmodel, axis=0))
    norms = np.sum(np.sum(model_zerocentered ** 2, axis=0))
    s = float(dots / norms)

    # Translation with and without scale (keeping (3,1) dimension to avoid broadcasting (3,N))
    transGT = data_mean - s * rot @ model_mean  # (3,1)
    trans = data_mean - rot @ model_mean  # (3,1)

    # Modelos alineados
    model_alignedGT = s * rot @ model + transGT  # (3,N)
    model_aligned = rot @ model + trans  # (3,N)

    alignment_errorGT = model_alignedGT - data  # (3,N)
    alignment_error = model_aligned - data  # (3,N)

    trans_errorGT = np.sqrt(np.sum(alignment_errorGT ** 2, axis=0))  # (N,)
    trans_error = np.sqrt(np.sum(alignment_error ** 2, axis=0))  # (N,)

    return rot, transGT, trans_errorGT, trans, trans_error, s


def build_correspondence_matrices(slam_centers, slam_ids,
                                  gt_centers, gt_ids):
    """
    Build 'model' and 'data' for Horn (closed-form).

    slam_centers -- (Ns, 3)
    slam_ids -- (Ns,)
    gt_centers -- (Nc, 3)
    gt_ids -- (Nc,)

    Return:
      model: (3, N) -- SLAM trajectory
      data:  (3, N) -- Ground Truth trajectory
      common_ids: (N,) -- matched IDs
    """
    slam_dict = {int(i): slam_centers[k] for k, i in enumerate(slam_ids)}
    gt_dict = {int(i): gt_centers[k] for k, i in enumerate(gt_ids)}

    # IDs intersection
    common_ids = sorted(set(slam_dict.keys()) & set(gt_dict.keys()))

    if len(common_ids) < 3:
        raise ValueError(f"There are not enough matched ids (only {len(common_ids)}).")

    slam_pts = np.array([slam_dict[i] for i in common_ids]).T   # (3, N)
    gt_pts = np.array([gt_dict[i] for i in common_ids]).T  # (3, N)

    return slam_pts, gt_pts, np.array(common_ids, dtype=int)