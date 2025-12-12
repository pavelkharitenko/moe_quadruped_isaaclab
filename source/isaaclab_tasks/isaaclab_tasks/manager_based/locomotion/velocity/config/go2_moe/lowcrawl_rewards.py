# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lowcrawl_base_height_exp(
    env: "ManagerBasedRLEnv",
    std: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward that exponentially prefers the base (torso) to be close to `target_height`.
    Use this to push the robot to a low-profile posture.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # base index assumed to be asset_cfg.body_ids (should contain base); supports batching
    base_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # shape (B, n_bodies)
    # if multiple bodies selected, we sum squared error across them (usually one)
    height_error = torch.sum(torch.square(base_height - target_height), dim=1)
    return torch.exp(-height_error / (std**2))


def lowcrawl_feet_clearance_penalty(
    env: "ManagerBasedRLEnv",
    std: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize (or negatively reward) foot positions that are higher than `target_height`.
    Returns a positive reward for feet close to or below target_height and smaller rewards when feet
    are lifted (higher clearance). Implemented as an exponential falloff of squared error.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    # feet heights: shape (B, n_feet)
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # compute squared distance above target (clamp negative values to 0 so being below target isn't penalized)
    above_target = torch.clamp(feet_height - target_height, min=0.0)
    feet_error = torch.sum(torch.square(above_target), dim=1)
    return torch.exp(-feet_error / (std**2))


def lowcrawl_feet_contact_fraction(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Reward proportional to the fraction of specified feet that are in contact with the ground.
    Uses ContactSensor.compute_first_contact over the current step to determine contacts.
    Returns values in [0, 1] per environment in batch.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # boolean tensor (B, n_bodies) indicating contact in the window
    recent_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # sum contacts and normalize by number of feet
    contact_count = torch.sum(recent_contact.float(), dim=1)
    n = len(sensor_cfg.body_ids) if hasattr(sensor_cfg, "body_ids") and sensor_cfg.body_ids else recent_contact.shape[1]
    # avoid division by zero — fallback to 1
    n = n if n > 0 else 1
    return contact_count / float(n)


def lowcrawl_feet_last_air_time_reward(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    max_air_time: float,
) -> torch.Tensor:
    """
    Reward that encourages short recent air-times for the selected feet.
    If feet have been in the air longer than `max_air_time`, the reward decreases (can be negative).
    This uses contact_sensor.data.last_air_time which holds last continuous air duration for each body.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # last_air_time shape (B, n_bodies)
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # clamp to max_air_time and compute penalty proportionally
    capped = torch.clamp(max_air_time - last_air_time, min=0.0)
    # sum normalized reward across feet
    reward = torch.sum(capped / (max_air_time + 1e-6), dim=1) / float(last_air_time.shape[1])
    return reward


def lowcrawl_crawl_posture_orientation_l2(
    env: "ManagerBasedRLEnv",
    target_gravity: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize deviation of the base-frame projected gravity vector from a provided `target_gravity`.
    For a low crawl you may want the robot to keep a roughly horizontal torso (e.g. [0, 0, 1] or small tilt),
    but this term can also be used to discourage large roll/pitch.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    target = torch.tensor(target_gravity, device=env.device)
    return torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=1)
