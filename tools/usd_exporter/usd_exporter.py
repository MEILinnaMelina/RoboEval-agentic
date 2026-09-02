"""
Requires the usd extra for mujoco

pip install mujoco[usd]
"""

import sys
from pathlib import Path
import tempfile

import mujoco
from dm_control import mjcf

from mujoco.usd.exporter import USDExporter

from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

from roboeval.demonstrations.demo import Demo
from roboeval.demonstrations.demo import DemoStep

from roboeval.roboeval_renderer import RoboEvalRenderer

import rich_click as click
from click_prompt import filepath_option
from rich.console import Console
from rich.progress import track

console = Console()


class CustomDemoRenderer(RoboEvalRenderer):
    def __init__(self, mojo):
        super().__init__(mojo)
        self.viewer = CustomOffScreenViewer(self.model, self.data)

    def _get_viewer(self, render_mode: str | None = None):
        self.viewer.make_context_current()
        return self.viewer

    def close(self):
        super().close()
        self.viewer.close()

    def set_demo_data(self, demo_info: DemoStep, actual_info: DemoStep):
        self._get_viewer().set_step_data(demo_info, actual_info)


class CustomOffScreenViewer(OffScreenViewer):
    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
    ):
        super().__init__(model, data)
        self.demo_info: DemoStep | None = None
        self.actual_info: DemoStep | None = None

    def set_step_data(self, demo_info: DemoStep, actual_info: DemoStep):
        self.demo_info = demo_info
        self.actual_info = actual_info

    def _create_overlay(self):
        return

    def render(self):
        return super().render("rgb_array")


def prepare_model_for_export(
    model: mujoco.MjModel,
    mjcf_model: "mjcf.RootElement",
) -> mujoco.MjModel:
    def remove_invalid_chars(s: str) -> str:
        s = s.replace(" ", "_")
        s = s.replace("//", "_")
        return s

    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = Path(tmpdir) / "scene.xml"
        # We need also to export the assets
        mjcf.export_with_assets(mjcf_model, out_dir=tmpdir, out_file_name=xml_path.name)
        spec = mujoco.MjSpec.from_file(str(xml_path))

        for obj_list in [
            spec.bodies,
            spec.geoms,
            spec.joints,
            spec.sites,
            spec.cameras,
            spec.lights,
            spec.meshes,
            spec.materials,
            spec.sensors,
        ]:
            for obj in obj_list:
                if obj.name:
                    obj.name = remove_invalid_chars(obj.name)

                if hasattr(obj, "material") and obj.material is not None:
                    obj.material = remove_invalid_chars(obj.material)

                if hasattr(obj, "meshname") and obj.meshname is not None:
                    obj.meshname = remove_invalid_chars(obj.meshname)

        # remove tags that are unnecessary for USD export
        for obj_list in [spec.excludes, spec.equalities, spec.tendons, spec.actuators]:
            for obj in obj_list:
                obj.delete()

        return spec.compile()


@click.command()
@filepath_option(
    "--demo-path",
    default="~/code/RobotOlympics/data/Cube/Bimanual Panda/00066fa2b1b24439bb66f5fc406c379b.safetensors",
    help="Recorded demo to load",
)
@filepath_option(
    "--output-path",
    default="./videos",
    prompt=False,
    help="Default output folder to store the rendered videos",
)
def cli(demo_path, output_path):
    """
    Exports a recording to USD
    """
    demo_path = Path(demo_path).expanduser().absolute()
    output_path = Path(output_path).expanduser().absolute()

    demo = Demo.from_safetensors(demo_path)

    frequency = 50
    env = demo.metadata.get_env(frequency, "human")
    robot_name = demo.metadata.environment_data.robot_name
    env_name = demo.metadata.environment_data.env_name
    demo_recorded_date = demo.metadata.date

    console.rule("Exporting to USD")

    if not output_path.exists():
        output_path.mkdir()

    if not output_path.is_dir():
        console.print("[red]Error[/red] output-path is not a folder")
        sys.exit(-1)

    console.print(f"[blue]Reading[/blue] demo from [gray]{demo_path}[/gray]")
    console.print(f"[blue]Writing[/blue] exporting USD to [gray]{output_path}[/gray]")

    console.print(
        f"Robot name: [gray]{robot_name}[/gray]. Env name: [gray]{env_name}[/gray]. Recording date: [gray]{demo_recorded_date}[/gray]"
    )

    # demo = DemoConverter.decimate(demo, frequency, robot=env.robot)
    demo_renderer = CustomDemoRenderer(env.mojo)
    env.mujoco_renderer = demo_renderer

    cam = demo_renderer.viewer.cam
    cam.distance = 2.5
    cam.elevation = -25
    cam.lookat[:] = [1.0, 0, 1.0]
    # cam.lookat[:] = [0.0, 0, 1.0]

    # reset the env and replay the demo
    env.reset(seed=int(demo.seed))

    console.rule("Converting demo to video")

    model: mujoco.MjModel = env.mojo.model
    data: mujoco.MjData = env.mojo.data

    model = prepare_model_for_export(model, env.mojo.root_element.mjcf)

    exp = USDExporter(
        model=model, output_directory_root=output_path, light_intensity=10
    )
    try:
        for timestep in track(demo.timesteps, console=console, description="Rendering"):
            # cam.azimuth = (cam.azimuth + 1) % 360
            actual_timestep = DemoStep(
                *env.step(timestep.executed_action), timestep.executed_action
            )
            demo_renderer.set_demo_data(demo_info=timestep, actual_info=actual_timestep)
            exp.update_scene(data=data)
    except ValueError as e:
        console.print("[red]Error[/red] while rendering images: ", e)
    except KeyboardInterrupt:
        console.print("[orange]Keyboard interrupt.[/orange]")
    finally:
        env.close()

    exp.save_scene(filetype="usd")

    console.rule("Done")


if __name__ == "__main__":
    cli()
