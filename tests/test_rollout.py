"""Tests for the action-evaluation rollout (executor): waypoints + MuJoCo run."""
import numpy as np
import pytest

from simpact.actions import ActionProposal, Descend, Grasp, Push, Release, Rotate
from simpact.executor.waypoints import proposal_to_waypoints


# ---- waypoint bridge (pure, no sim) ---------------------------------------- #
def test_waypoints_accumulate_primitives():
    prop = ActionProposal([Push(0.1, -0.05), Descend(0.04), Grasp(0.03)])
    wps = proposal_to_waypoints(prop, [0.40, 0.0, 0.30], z_min=0.0, default_duration=1.0)
    # initial + one per action
    assert len(wps) == 4
    assert wps[1].position[:2] == pytest.approx([0.50, -0.05])  # PUSH
    assert wps[2].position[2] == pytest.approx(0.26)            # DESCEND -0.04
    assert wps[3].gripper_width == 0.03                          # GRASP width (not dropped)


def test_descend_clamped_to_z_min():
    wps = proposal_to_waypoints(ActionProposal([Descend(1.0)]), [0.4, 0.0, 0.3], z_min=0.2)
    assert wps[-1].position[2] == pytest.approx(0.2)


def test_release_opens_and_rotate_changes_orientation():
    wps = proposal_to_waypoints(
        ActionProposal([Release(), Rotate(1.5708)]), [0.4, 0.0, 0.3], initial_orientation=(0, 1, 0, 0)
    )
    assert wps[1].gripper_width == 1.0
    assert wps[2].orientation != wps[0].orientation  # yaw applied


def test_grasp_width_alias_into_waypoint():
    # dict-form action using the LLM 'grasp_width' key must reach the waypoint
    wps = proposal_to_waypoints([{"type": "GRASP", "grasp_width": 0.02}], [0.4, 0, 0.3])
    assert wps[-1].gripper_width == 0.02


def test_optimizer_plan_actions_roll_out():
    # the optimizer's refined plan (move/gripper_control) rolls out like a proposal
    def move(dx, dy, dz):
        return {"type": "move", "delta_x": dx, "delta_y": dy, "delta_z": dz,
                "delta_roll": 0.0, "delta_pitch": 0.0, "delta_yaw": 0.0}
    plan = [move(0.05, -0.02, 0.0), {"type": "gripper_control", "width": 0.03},
            move(0.0, 0.0, -0.5)]
    wps = proposal_to_waypoints(plan, [0.40, 0.0, 0.30], z_min=0.10, default_duration=1.0)
    assert len(wps) == 4
    assert wps[1].position[:2] == pytest.approx([0.45, -0.02])  # MOVE x,y
    assert wps[2].gripper_width == 0.03                          # GRIPPER_CONTROL
    assert wps[3].position[2] == pytest.approx(0.10)            # MOVE -z clamped to z_min


def test_pag_initial_gripper_pose_ee_offset():
    # the committed home pose (mocap frame) lifts to the EE frame by 0.105 m along
    # the tool axis — the EE tip sits below the mocap body for a downward gripper.
    import numpy as np

    from simpact.executor.rollout import (
        HOME_GRIPPER_ORIENTATION,
        HOME_GRIPPER_POSITION,
    )
    from simpact.real2sim.convert_gripper_pose import ee_pose_from_matrix, ee_pose_to_matrix

    mocap = np.asarray(HOME_GRIPPER_POSITION)
    wxyz = np.asarray(HOME_GRIPPER_ORIENTATION)
    T_ee = ee_pose_to_matrix(mocap, wxyz)
    assert T_ee[2, 3] < mocap[2]  # EE tip below the mocap body (points down)
    # round-trips back to the mocap pose
    mocap_back, wxyz_back = ee_pose_from_matrix(T_ee)
    assert mocap_back == pytest.approx(mocap, abs=1e-6)
    assert wxyz_back == pytest.approx(wxyz, abs=1e-6)


def test_snap_to_mocap_teleports_hand_to_target(tmp_path):
    # the welded hand spawns at the XML default (origin, z=0.5); snap_to_mocap must
    # place it AT the target so it doesn't fly across the scene knocking objects over
    mujoco = pytest.importorskip("mujoco")
    trimesh = pytest.importorskip("trimesh")
    from simpact.executor.rollout import MuJoCoRollout
    from simpact.real2sim.mujoco_load_gripper import FloatingGripperController

    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(mesh)
    pose = np.eye(4); pose[:3, 3] = [0.45, 0.0, 0.04]
    roll = MuJoCoRollout([{"name": "box", "mesh_file": str(mesh), "pose": pose}],
                         xml_path=tmp_path / "scene.xml")
    ctrl = FloatingGripperController(roll.xml)
    model, data = ctrl.model, ctrl.data
    hid = ctrl.hand_body_id

    mujoco.mj_forward(model, data)
    assert data.xpos[hid][2] == pytest.approx(0.5, abs=1e-3)  # spawns at the origin

    target = np.array([0.40, -0.10, 0.30])
    ctrl.set_gripper_pose(target, np.array([1.0, 0.0, 0.0, 0.0]))
    ctrl.snap_to_mocap()
    mujoco.mj_forward(model, data)
    assert np.allclose(data.xpos[hid], target, atol=2e-3)  # now AT the target
    # and it stays put (weld already satisfied — no fly-in)
    for _ in range(20):
        mujoco.mj_step(model, data)
    assert np.allclose(data.xpos[hid], target, atol=5e-3)


# ---- full MuJoCo rollout ---------------------------------------------------- #
@pytest.fixture
def box_rollout(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    trimesh = pytest.importorskip("trimesh")
    from simpact.executor.rollout import MuJoCoRollout

    mesh = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.05, 0.05, 0.08)).export(mesh)
    pose = np.eye(4); pose[:3, 3] = [0.45, 0.0, 0.04]
    objs = [{"name": "box", "mesh_file": str(mesh), "pose": pose, "rgba": (0.8, 0.2, 0.2)}]
    roll = MuJoCoRollout(objs, xml_path=tmp_path / "scene.xml")
    # start the (closed) gripper just behind the box at its height, push +x into it
    res = roll.run(
        ActionProposal([Push(0.18, 0.0)]),
        initial_position=[0.34, 0.0, 0.045],
        initial_orientation=(0, 1, 0, 0),
        z_min=0.0, default_duration=1.0, initial_gripper=0.0, settle_steps=100,
        render=False,  # headless-safe; rendering exercised separately
    )
    return res, tmp_path


def test_mujoco_rollout_push_moves_object(box_rollout):
    res, _ = box_rollout
    assert res.object_names == ["box"]
    # initial + one transition + settled final
    assert len(res.snapshots) == 3
    assert res.snapshots[0]["waypoint_index"] == 0
    assert np.isfinite(res.final_poses["box"]["position"]).all()
    dx = res.final_poses["box"]["position"][0] - res.initial_poses["box"]["position"][0]
    assert dx > 0.02, f"box should move +x from the push, got dx={dx:.3f}"


def test_rollout_saves_pag_format_json(box_rollout):
    import json

    res, tmp_path = box_rollout
    path = res.save(tmp_path / "out", index=0, instruction="push the box")
    data = json.loads(open(path).read())
    assert path.endswith("rollout_00.json")
    for key in ("timestamp", "object_names", "waypoints", "snapshots", "instruction"):
        assert key in data
    assert data["proposal_index"] == 0
    snap = data["snapshots"][0]
    for key in ("waypoint_index", "gripper", "objects", "screenshot"):
        assert key in snap
    assert set(snap["gripper"]) == {"position", "orientation", "width"}
    # render=False -> screenshots are null but the record is complete
    assert snap["screenshot"] is None
