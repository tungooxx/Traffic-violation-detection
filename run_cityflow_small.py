from mtmc.cityflow_adapter import CityFlowAdapter


def main():
    adapter = CityFlowAdapter("cityflow_2022_small/")
    print(adapter.summary())

    for scene in adapter.scenes:
        print(f"\nScene: {scene}")
        for camera in adapter.get_camera_configs(scene):
            print(f"  {camera['id']}")
            print(f"    video: {camera['video']}")
            print(f"    mask: {camera['mask']}")
            print(f"    position: {camera['position']}")


if __name__ == "__main__":
    main()
