import omni.kit.app
import omni.usd


USD_PATH = "/home/autolab/AMMR/isaac_usd/mycobot_280_m5_adaptive_gripper_reimport/mycobot_280_m5_adaptive_gripper/mycobot_280_m5_adaptive_gripper.usda"


def main():
    app = omni.kit.app.get_app()
    usd_context = omni.usd.get_context()

    print(f"Opening myCobot stage: {USD_PATH}")
    if not usd_context.open_stage(USD_PATH):
        raise RuntimeError(f"Failed to open stage: {USD_PATH}")

    for _ in range(20):
        app.update()

    stage = usd_context.get_stage()
    if stage is None:
        raise RuntimeError("USD stage did not become available after open_stage().")

    root = stage.GetDefaultPrim()
    print(f"Opened stage default prim: {root.GetPath() if root else '<none>'}")

    try:
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(
            eye=[0.65, -0.85, 0.55],
            target=[0.0, 0.0, 0.16],
            camera_prim_path="/OmniverseKit_Persp",
        )
        print("Viewport camera moved to myCobot.")
    except Exception as exc:
        print(f"Could not set viewport camera automatically: {exc}")


main()
