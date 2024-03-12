"""This file is a basic test to determine if the environment is properly configured.
"""


def test_scsc_environment() -> None:
    """If this runs without issue, then the configuration is acceptable.
    """
    try:
        # MDM / UHC
        from train.train_mdm_iterative_finetuning import main
        from train.training_loop import SelfConsumingTrainLoop

        # UHC
        from UniversalHumanoidControl.uhc.agents.agent_copycat import AgentCopycat
    except AssertionError as e:
        if "Path UniversalHumanoidControl/data/smpl" in str(e) and "does not exist!" in str(e):
            print(
                "Cannot find SMPL data for UHC. "
                "Please follow instructions at: "
                "https://github.com/ZhengyiLuo/UniversalHumanoidControl?tab=readme-ov-file#smpl-robot")
        else:
            raise e

    # Visualization
    # this will fail if OpenGL, EGL, Pyrender, etc. is not properly setup
    from body_visualizer.mesh.mesh_viewer import MeshViewer  # noqa
    from body_visualizer.tools.mesh_tools import rotateXYZ  # noqa
    from translation.mdm_to_amass import VposerSkinResult  # noqa


if __name__ == "__main__":
    test_scsc_environment()
    print("Python test_environment.py script threw no unexpected errors.")
