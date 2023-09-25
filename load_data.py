import numpy as np
import os
import pickle


path_kdprain = './NJU_CPOL_kdpRain'
path_update = ('./NJU_CPOL_update2308/ZDR/1.0km/data_dir_000/')

path_features = os.listdir('./NJU_CPOL_update2308')
path_heights = os.listdir("./NJU_CPOL_update2308/dBZ")
path_processions = os.listdir("./NJU_CPOL_update2308/dBZ/1.0km")



processions = []
def load_all_data():
    os.makedirs('./data', exist_ok=True)
    for procession in path_processions:
            frames = os.listdir('./NJU_CPOL_update2308/dBZ/1.0km/' + str(procession))
            path = "./data/" + str(procession) + '.pkl'
            times = []
            for frame in frames:
                # os.makedirs("./data/" + str(procession), exist_ok=True)
                heights = []
                for height in path_heights:
                    features = []
                    for feature in path_features:
                        data = np.load("./NJU_CPOL_update2308/" + str(feature) + '/' + str(height) + '/' + str(procession) + '/' + str(frame))
                        print("./NJU_CPOL_update2308/" + str(feature) + '/' + str(height) + '/' + str(procession) + '/' + str(frame))
                        features.append(data)
                    heights.append(features)
                times.append(heights)
            with open(path, "wb") as f:
                pickle.dump(np.array(times), f)


# load_all_data()


def load_rain_data(root_path):
    dir_list = os.listdir(root_path)
    for dir_path in dir_list:
        path = root_path + "/" + dir_path
        frames = os.listdir(path)
        frame_list = []
        for frame in frames:
            frame_path = path + "/" + frame
            data = np.load(frame_path)
            frame_list.append(data)
        frame_list = np.array(frame_list)
        with open("./rain_data/" + dir_path + ".pkl", "wb") as f:
            pickle.dump(frame_list, f)
            # data.shape: time_steps, H, L


load_rain_data("./NJU_CPOL_kdpRain")
